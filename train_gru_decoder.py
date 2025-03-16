import time
import torch.optim
import torch.utils.data
import pandas as pd
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import ast
import numpy as np
import os

from my_datasets import CaptionDataset
from utils import *
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import configparser
import pickle
from nltk.translate.meteor_score import meteor_score
from statistics import mean
from torchsummary import summary

output_folder = "./"

# Data parameters
config = configparser.ConfigParser()
config.read(os.path.join("configs", "config_train_lstm_decoder.ini"))

for section in config.sections():
    print(f'[{section}]')
    for key, value in config.items(section):
        print(f'{key} = {value}')
    print()

paths = config["file_names_and_paths"]
data_folder = paths["dataset_folder"]
model_name = paths["model_name"]

# Encoder parameters
encoder_blocks = config.getint("encoder_parameters", "num_blocks")
encoded_ecg_size = config.getint("encoder_parameters", "encoded_ecg_size")
kernel_size_str = config.get("encoder_parameters", "kernel_size")
kernel_size = ast.literal_eval(kernel_size_str)
stem_kernel_size = config.getint("encoder_parameters", "stem_kernel_size")

# Decoder parameters
attention_dim = config.getint("decoder_parameters", "attention_dim")
encoder_dim = config.getint("decoder_parameters", "encoder_dim")
decoder_dim = config.getint("decoder_parameters", "decoder_dim")
emb_dim = config.getint("decoder_parameters", "emb_dim")
dropout = config.getfloat("decoder_parameters", "dropout")
layers = config.getint("decoder_parameters", "layers")

# Training parameters
max_caption_length = config.getint("training_parameters", "max_caption_length")
epochs = config.getint("training_parameters", "epochs")
batch_size = config.getint("training_parameters", "batch_size")
encoder_lr = config.getfloat("training_parameters", "encoder_lr")
decoder_lr = config.getfloat("training_parameters", "decoder_lr")
grad_clip = config.getfloat("training_parameters", "grad_clip")
alpha_c = config.getfloat("training_parameters", "alpha_c")
print_freq = config.getint("training_parameters", "print_freq")
early_stopping_trigger = config.getint("training_parameters", "early_stopping_trigger")
train_teacher_forcing_probability = config.getfloat("training_parameters", "train_teacher_forcing_probability")
val_teacher_forcing = config.getboolean("training_parameters", "val_teacher_forcing")
sanity_check = config.getboolean("training_parameters", "sanity_check")
normalize_ds = config.getboolean("training_parameters", "normalize_ds")
adjust_learning_rate_after = config.getint("training_parameters", "adjust_learning_rate_after")
debug_mode = config.getboolean("training_parameters", "debug_mode")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


################################################
#  Attention + GRU-based Decoder
################################################
class Attention(nn.Module):
    """
    Same attention mechanism used by the original LSTM-based code.
    """
    def __init__(self, attention_dim, encoder_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim) 
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        # encoder_out: (batch_size, num_pixels, encoder_dim)
        # decoder_hidden: (batch_size, decoder_dim)

        att1 = self.encoder_att(encoder_out)      # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)   # (batch_size, attention_dim)
        att2 = att2.unsqueeze(1)                  # (batch_size, 1, attention_dim)
        att = self.full_att(self.relu(att1 + att2)).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)                 # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    """
    GRU-based decoder with attention; 
    replaces LSTMCell layers and cell state references with GRU logic.
    """
    def __init__(self, attention_dim, embed_dim, decoder_dim, token_encoder,
                 encoder_dim, layers, dropout):
        super().__init__()

        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = token_encoder.vocab_size
        self.dropout = dropout
        self.encoder_dim = encoder_dim
        self.layers = layers

        self.embedding = nn.Embedding(self.vocab_size, embed_dim)
        self.attention = Attention(attention_dim, encoder_dim)

        # Replace LSTMCell with GRUCell. 
        # If you want multiple GRU layers, see below on how to stack them.
        self.decode_step = nn.GRUCell(embed_dim + encoder_dim, decoder_dim, bias=True)

        # Only one hidden init for GRU
        self.init_h = nn.Linear(encoder_dim, decoder_dim)

        # Optional gating scalar
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()

        # Classification layer
        self.fc = nn.Linear(decoder_dim, self.vocab_size)
        self.init_weights()

        self.dropout_layer = nn.Dropout(self.dropout)

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        """
        For a GRU, we only need one hidden state (no cell state).
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        return h

    def forward(self, encoder_out, encoded_captions, caption_lengths,
                teacher_forcing, token_encoder, max_caption_length):
        """
        Forward pass of the GRU-based decoder with attention.
        encoder_out: (batch_size, num_pixels, encoder_dim)
        """
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten/reshape the encoder output: (batch_size, num_pixels, encoder_dim)
        # num_pixels is "time" dimension from the CNN perspective
        # or whichever shape the dense blocks produce.
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embed the input captions
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize GRU hidden state
        h = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # Determine decoding lengths (exclude <end>)
        decode_lengths = (caption_lengths - 1).tolist()

        # Tensors to hold predictions and attention
        predictions = torch.zeros(batch_size, max_caption_length, vocab_size).to(device)
        alphas = torch.zeros(batch_size, max_caption_length, num_pixels).to(device)

        # Decode step by step
        for t in range(max_caption_length):
            batch_size_t = sum([l > t for l in decode_lengths])

            # Attention
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            # Gate
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding

            # Prepare Decoder input: embed + context (attention)
            gru_input = torch.cat([embeddings[:batch_size_t, t, :],
                                   attention_weighted_encoding], dim=1)

            # Single GRUCell update
            new_h = self.decode_step(gru_input, h[:batch_size_t])
            h = torch.cat([new_h, h[batch_size_t:]], dim=0)

            # Compute output token logits
            preds = self.fc(self.dropout_layer(h[:batch_size_t]))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind


############################################
# Minimal "Model" wrapper
############################################
class Model(nn.Module):
    def __init__(self, encoder, decoder):
        super(Model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, ecg, encoded_captions, caption_lengths, teacher_forcing,
                token_encoder, max_caption_length):
        encoder_out = self.encoder(ecg)
        outputs = self.decoder(
            encoder_out, encoded_captions, caption_lengths, teacher_forcing,
            token_encoder, max_caption_length
        )
        return outputs

def calculate_total_parameters(model):
    total_params = model.encoder.get_num_params() + model.decoder.parameters().__next__().numel()
    total_params = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad) \
                   + sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    return total_params

############################################################
############################################################
class DenseLayer1D(nn.Module):
    """
    Same as before, omitted for brevity. 
    """
    def __init__(self, in_channels, growth_rate, kernel_size=3):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, growth_rate, kernel_size=kernel_size,
                              stride=1, padding=padding, bias=False)

    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        return torch.cat([x, out], dim=1)

class DenseBlock1D(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, kernel_size=3):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(DenseLayer1D(in_channels, growth_rate, kernel_size))
            in_channels += growth_rate
        self.block = nn.Sequential(*layers)
        self.out_channels = in_channels

    def forward(self, x):
        return self.block(x)

class Transition1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(self.relu(self.bn(x)))
        x = self.pool(x)
        return x

class Encoder(nn.Module):
    """
    DenseNet1D-121 style Encoder.
    """
    def __init__(self, num_blocks, encoded_ecg_size, kernel_size, stem_kernel_size, in_chan=12):
        super().__init__()
        self.block_config = (6, 12, 24, 16)
        self.growth_rate = 32
        
        # Stem
        stem_padding = (stem_kernel_size - 1) // 2
        self.stem = nn.Sequential(
            nn.Conv1d(in_chan, 64, kernel_size=stem_kernel_size, stride=2,
                      padding=stem_padding, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        num_features = 64
        blocks = []
        for i, nb_layers in enumerate(self.block_config):
            block = DenseBlock1D(nb_layers, num_features, self.growth_rate, kernel_size=3)
            blocks.append(block)
            num_features = block.out_channels
            if i != len(self.block_config) - 1:
                out_features = num_features // 2
                trans = Transition1D(num_features, out_features)
                blocks.append(trans)
                num_features = out_features

        self.dense_blocks = nn.Sequential(*blocks)
        self.bn_final = nn.BatchNorm1d(num_features)
        self.conv_final = nn.Conv1d(num_features, encoded_ecg_size, kernel_size=1, bias=False)
        self.relu_final = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.stem(x)
        x = self.dense_blocks(x)
        x = self.relu_final(self.bn_final(x))
        x = self.conv_final(x)
        return x

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def main():
    print(device)
    global_start = time.time()

    print("############################################")
    print("trained model will be " + model_name)
    print("folder used is " + data_folder)
    print("############################################")

    train_ds = CaptionDataset(data_folder, 'TRAIN')
    val_ds = CaptionDataset(data_folder, 'VAL')
    test_ds = CaptionDataset(data_folder, 'TEST')

    # Adjust shape
    if train_ds.secgs.ndim == 2:
        train_ds.secgs = np.expand_dims(train_ds.secgs, 1)
        val_ds.secgs = np.expand_dims(val_ds.secgs, 1)
        test_ds.secgs = np.expand_dims(test_ds.secgs, 1)

    # Optional normalization
    if normalize_ds:
        train_ds_mean = train_ds.secgs.mean(axis=(0, 2), keepdims=True)
        train_ds_std = train_ds.secgs.std(axis=(0, 2), keepdims=True)
        train_ds.secgs = (train_ds.secgs - train_ds_mean) / train_ds_std
        val_ds.secgs = (val_ds.secgs - train_ds_mean) / train_ds_std
        test_ds.secgs = (test_ds.secgs - train_ds_mean) / train_ds_std

    # Read word map
    with open(os.path.join(data_folder, 'encoder.pkl'), 'rb') as f:
        token_encoder = pickle.load(f)

    # Instantiate GRU-based decoder
    decoder = DecoderWithAttention(
        attention_dim=attention_dim,
        embed_dim=emb_dim,
        decoder_dim=decoder_dim,
        token_encoder=token_encoder,
        encoder_dim=encoder_dim, 
        layers=layers,
        dropout=dropout
    )

    # Instantiate DenseNet1D-based encoder
    in_chan = train_ds.secgs.shape[1]   # typically 12 for 12-lead ECG
    ecg_length = train_ds.secgs.shape[2]

    encoder = Encoder(
        num_blocks=encoder_blocks,
        encoded_ecg_size=encoded_ecg_size,
        kernel_size=kernel_size,
        stem_kernel_size=stem_kernel_size,
        in_chan=in_chan
    )

    model = Model(encoder, decoder).to(device)

    print("Encoder (DenseNet1D-121) + GRU Decoder are initialized.")
    summary(model.encoder, input_size=(in_chan, ecg_length))
    total_params = calculate_total_parameters(model)
    print(f"Total number of parameters: {total_params/1e6:.2f}M")

    optimizer = torch.optim.Adam([
        {'params': filter(lambda p: p.requires_grad, encoder.parameters()), 'lr': encoder_lr},
        {'params': filter(lambda p: p.requires_grad, decoder.parameters()), 'lr': decoder_lr}
    ])
    criterion = nn.CrossEntropyLoss().to(device)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    epochs_since_improvement = 0
    best_meteor = -1
    meteor_str = "0"

    train_losses = []
    val_losses = []
    val_meteor_scores = []

    for epoch in range(epochs):
        if epochs_since_improvement >= early_stopping_trigger:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % adjust_learning_rate_after == 0:
            adjust_learning_rate(optimizer, 0.8)

        # TRAIN
        train_loss = train(
            train_loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            token_encoder=token_encoder,
            epoch=epoch,
            train_teacher_forcing_probability=train_teacher_forcing_probability
        )
        train_losses.append(train_loss.avg)

        # VALIDATE
        (recent_bleu4, recent_bleu1, recent_bleu2,
         recent_loss, recent_rouge1, recent_meteor) = validate(
            val_loader=val_loader,
            model=model,
            criterion=criterion,
            token_encoder=token_encoder,
            teacher_forcing=val_teacher_forcing
        )
        val_losses.append(recent_loss.avg)
        val_meteor_scores.append(recent_meteor)

        is_best = recent_meteor > best_meteor
        if not is_best:
            epochs_since_improvement += 1
            print(f"\nEpochs since last improvement: {epochs_since_improvement}\n")
        else:
            if epoch > 0:
                os.remove(os.path.join(output_folder,
                                       'BEST_checkpoint_' + model_name + "_METEOR-" + meteor_str + '.pth.tar'))
            epochs_since_improvement = 0
            best_meteor = recent_meteor
            meteor_str = str(round(best_meteor, 3))

        encoder_config_dict = dict(
            num_blocks=encoder_blocks,
            kernel_size=kernel_size,
            stem_kernel_size=stem_kernel_size,
            encoded_ecg_size=encoded_ecg_size,
            in_chan=in_chan,
        )
        decoder_config_dict = dict(
            attention_dim=attention_dim,
            embed_dim=emb_dim,
            decoder_dim=decoder_dim,
            encoder_dim=encoder_dim,
            layers=layers,
            dropout=dropout
        )

        save_checkpoint(
            output_folder, model_name, epoch, epochs_since_improvement,
            model, optimizer, encoder_config_dict, decoder_config_dict,
            recent_meteor, meteor_str, is_best
        )

        print(
            f"\nEpoch {epoch} complete. "
            f"Best METEOR so far: {best_meteor:.4f}"
        )

        if debug_mode:
            break

    # Load best model for final testing
    best_checkpoint = torch.load(
        os.path.join(output_folder, f'BEST_checkpoint_{model_name}_METEOR-{meteor_str}.pth.tar')
    )
    model.load_state_dict(best_checkpoint['model'])

    print("Testing model:")
    test_bleu4, test_bleu1, test_bleu2, test_loss, test_rouge1, test_meteor = validate(
        val_loader=test_loader,
        model=model,
        criterion=criterion,
        token_encoder=token_encoder,
        teacher_forcing=False
    )

    print("\n"
          f"TEST BLEU4-Score: {test_bleu4:.4f}\n"
          f"TEST BLEU1-Score: {test_bleu1:.4f}\n"
          f"TEST BLEU2-Score: {test_bleu2:.4f}\n"
          f"TEST ROUGE1: P - {test_rouge1['rouge1_precision']:.4f}, "
          f"R - {test_rouge1['rouge1_recall']:.4f}, "
          f"F - {test_rouge1['rouge1_f1_score']:.4f}\n"
          f"TEST METEOR-Score: {test_meteor:.4f}")

    os.rename(
        os.path.join(output_folder, f'BEST_checkpoint_{model_name}_METEOR-{meteor_str}.pth.tar'),
        os.path.join(output_folder, f'BEST_checkpoint_{model_name}_TEST_METEOR-{round(test_meteor, 3)}.pth.tar')
    )

    print("\n### Entire training runtime ###")
    print(time.strftime("%H:%M:%S", time.gmtime((time.time() - global_start))))

    save_results_to_csv(
        output_folder, epoch, train_losses, val_meteor_scores,
        test_bleu4, test_bleu1, test_bleu2, test_rouge1, test_meteor
    )


def train(train_loader, model, criterion, optimizer, token_encoder, epoch,
          train_teacher_forcing_probability):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()
    for i, (batch, caps, caplens) in enumerate(train_loader):
        data_time.update(time.time() - start)
        if sanity_check:
            batch.fill_(1)
        batch = batch.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        teacher_forcing = np.random.choice(
            [True, False],
            p=[train_teacher_forcing_probability,
               1 - train_teacher_forcing_probability]
        )
        scores, caps_sorted, decode_lengths, alphas, sort_ind = model(
            batch, caps, caplens, teacher_forcing, token_encoder, max_caption_length
        )

        targets = caps_sorted[:, 1:]
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        loss = criterion(scores, targets)
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()

        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        optimizer.step()

        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        if i % print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                  f'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'Top-5 Accuracy {top5accs.val:.3f} ({top5accs.avg:.3f})')

    return losses


@torch.inference_mode()
def validate(val_loader, model, criterion, token_encoder, teacher_forcing):
    model.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = []
    hypotheses = []

    with torch.inference_mode():
        for i, (batch, caps, caplens) in enumerate(val_loader):
            if sanity_check:
                batch = torch.ones((len(caps), 7663))
            batch = batch.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            scores, caps_sorted, decode_lengths, alphas, sort_ind = model(
                batch, caps, caplens, teacher_forcing, token_encoder, max_caption_length
            )

            targets = caps_sorted[:, 1:]
            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            loss = criterion(scores, targets)
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)
            start = time.time()

            if i % print_freq == 0:
                print(f'Validation: [{i}/{len(val_loader)}]\t'
                      f'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      f'Top-5 Accuracy {top5accs.val:.3f} ({top5accs.avg:.3f})\t')

            # Prepare references/hypotheses for BLEU, METEOR, etc.
            for j in range(caps_sorted.shape[0]):
                img_caps = caps_sorted[j].tolist()
                references.append(img_caps)

            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = []
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

            if debug_mode:
                break

        # Convert indices to text
        hypotheses = [token_encoder.decode(hyp, skip_special_tokens=True) for hyp in hypotheses]
        for i, reference in enumerate(references):
            references[i] = [token_encoder.decode(reference, skip_special_tokens=True)]

        pd.DataFrame({'hypotheses': hypotheses, 'references': references}
                     ).to_csv(os.path.join(output_folder, "tokenized_reports.csv"))

        print("predictions example:")
        print(hypotheses[0:5])
        if not teacher_forcing:
            with open('gru_tokenized_hypotheses.pkl', 'wb') as f:
                pickle.dump(hypotheses, f)
        print("references:")
        print(references[0:5])
        smoothing_function = SmoothingFunction().method2
        if not teacher_forcing:
            with open('gru_tokenized_references.pkl', 'wb') as f:
                pickle.dump(references, f)

        # Prepare for BLEU, ROUGE, METEOR
        tokenized_hypotheses = word_tokenize_caption_list(hypotheses)
        tokenized_references = []
        for ref in references:
            tokenized_references.append(word_tokenize_caption_list(ref))

        bleu4 = corpus_bleu(tokenized_references, tokenized_hypotheses,
                            smoothing_function=smoothing_function)
        bleu1 = corpus_bleu(tokenized_references, tokenized_hypotheses,
                            weights=[1, 0, 0, 0],
                            smoothing_function=smoothing_function)
        bleu2 = corpus_bleu(tokenized_references, tokenized_hypotheses,
                            weights=[0.5, 0.5, 0, 0],
                            smoothing_function=smoothing_function)
        rouge1 = evaluate_rouge_1(tokenized_references, tokenized_hypotheses)
        meteor = mean(meteor_score(ref, hyp)
                      for ref, hyp in zip(tokenized_references, tokenized_hypotheses))

        print(
            f'\n * LOSS - {losses.avg:.3f}, TOP-5 ACCURACY - {top5accs.avg:.3f}, '
            f'BLEU-1 - {bleu1:.4f}, BLEU-2 - {bleu2:.4f}, BLEU-4 - {bleu4:.4f}, '
            f'ROUGE1: P - {rouge1["rouge1_precision"]:.4f}, '
            f'R - {rouge1["rouge1_recall"]:.4f}, '
            f'F - {rouge1["rouge1_f1_score"]:.4f}, METEOR - {meteor:.4f}\n'
        )

    return bleu4, bleu1, bleu2, losses, rouge1, meteor


if __name__ == '__main__':
    main()
