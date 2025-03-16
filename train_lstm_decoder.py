import time
import torch.optim
import torch.utils.data
import pandas as pd
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import DecoderWithAttention, Model
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

def calculate_total_parameters(model):
    total_params = model.encoder.get_num_params() + model.decoder.get_num_params()
    return total_params

############################################
class DenseLayer1D(nn.Module):
    """
    A single 'dense' layer using 1D convolutions,
    with layout: BN -> ReLU -> Conv1d, then channel-wise concat.
    """
    def __init__(self, in_channels, growth_rate, kernel_size=3):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, growth_rate,
                              kernel_size=kernel_size,
                              stride=1,
                              padding=padding,
                              bias=False)

    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        # Concatenate existing features with newly generated features
        return torch.cat([x, out], dim=1)


class DenseBlock1D(nn.Module):
    """
    A dense block made of multiple dense layers.
    """
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
    """
    Transition down layer:
    BN -> ReLU -> 1x1 Conv -> AvgPool1d(stride=2)
    """
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
    DenseNet1D-121 style Encoder replacing the original ResNet-based 'Encoder'.
    By default: block_config = (6, 12, 24, 16), growth_rate=32.
    We do NOT perform a final global average pool, so time dimension is retained
    for attention-based decoding (i.e., shape is (batch, C, time)).
    """
    def __init__(self,
                 num_blocks=4,
                 encoded_ecg_size=512,
                 kernel_size=[3, 3, 3, 3],
                 stem_kernel_size=7,
                 in_chan=12):
        super().__init__()
        # Optional: tune the actual DenseNet1D configuration here
        self.block_config = (6, 12, 24, 16)  # typical DenseNet-121
        self.growth_rate = 32
        
        # Initial "stem" convolution
        stem_padding = (stem_kernel_size - 1) // 2
        self.stem = nn.Sequential(
            nn.Conv1d(in_chan, 64, kernel_size=stem_kernel_size, stride=2,
                      padding=stem_padding, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # Build Dense Blocks + Transitions
        num_features = 64
        blocks = []
        for i, nb_layers in enumerate(self.block_config):
            block = DenseBlock1D(nb_layers, num_features, self.growth_rate, kernel_size=3)
            blocks.append(block)
            num_features = block.out_channels
            if i != len(self.block_config) - 1:
                # Transition layer halves spatial dimension and reduces channels
                out_features = num_features // 2
                trans = Transition1D(num_features, out_features)
                blocks.append(trans)
                num_features = out_features

        self.dense_blocks = nn.Sequential(*blocks)
        
        # Final batch norm and reducing channels to 'encoded_ecg_size'
        self.bn_final = nn.BatchNorm1d(num_features)
        self.conv_final = nn.Conv1d(num_features, encoded_ecg_size, kernel_size=1, bias=False)
        self.relu_final = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Forward pass returning feature maps shaped (batch, encoded_ecg_size, time).
        """
        x = self.stem(x)              # (batch, 64, time//4)
        x = self.dense_blocks(x)      # (batch, num_features, time//(some_factor))
        x = self.relu_final(self.bn_final(x))
        x = self.conv_final(x)        # (batch, encoded_ecg_size, time//(some_factor))
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

    if train_ds.secgs.ndim == 2:
        train_ds.secgs = np.expand_dims(train_ds.secgs, 1)
        val_ds.secgs = np.expand_dims(val_ds.secgs, 1)
        test_ds.secgs = np.expand_dims(test_ds.secgs, 1)

    if normalize_ds:
        train_ds_mean = train_ds.secgs.mean(axis=(0, 2), keepdims=True)
        train_ds_std = train_ds.secgs.std(axis=(0, 2), keepdims=True)
        train_ds.secgs = (train_ds.secgs - train_ds_mean) / train_ds_std
        val_ds.secgs = (val_ds.secgs - train_ds_mean) / train_ds_std
        test_ds.secgs = (test_ds.secgs - train_ds_mean) / train_ds_std

    # Read word map
    with open(os.path.join(data_folder, 'encoder.pkl'), 'rb') as f:
        token_encoder = pickle.load(f)

    decoder = DecoderWithAttention(attention_dim=attention_dim,
                                   embed_dim=emb_dim,
                                   decoder_dim=decoder_dim,
                                   token_encoder=token_encoder,
                                   encoder_dim=encoder_dim,  # must match final channels of Encoder
                                   layers=layers,
                                   dropout=dropout)

    in_chan = train_ds.secgs.shape[1]   # 12 leads => 12 channels if shape is (batch,12,length)
    ecg_length = train_ds.secgs.shape[2]

    # Instantiate our new DenseNet1D-based encoder
    encoder = Encoder(num_blocks=encoder_blocks,
                      encoded_ecg_size=encoded_ecg_size,
                      kernel_size=kernel_size,
                      stem_kernel_size=stem_kernel_size,
                      in_chan=in_chan)

    model = Model(encoder, decoder)
    model.to(device)

    print("Encoder (DenseNet1D-121) and Decoder are initialized")
    summary(model.encoder, input_size=(in_chan, ecg_length))

    total_params = calculate_total_parameters(model)
    print(f"Total number of parameters: {total_params/1e6:.2f}M")

    optimizer = torch.optim.Adam(
        params=[
            {'params': filter(lambda p: p.requires_grad, encoder.parameters()),
             'lr': encoder_lr,
             'finetune': True},
            {'params': filter(lambda p: p.requires_grad, decoder.parameters()),
             'lr': decoder_lr,
             'finetune': True}
        ]
    )

    model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    epochs_since_improvement = 0
    best_bleu4 = 0
    best_bleu1 = 0
    best_bleu2 = 0
    best_loss = AverageMeter()
    best_rouge1 = {
        'rouge1_precision': 0,
        'rouge1_recall': 0,
        'rouge1_f1_score': 0
    }
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

        train_loss = train(train_loader=train_loader,
                           model=model,
                           criterion=criterion,
                           optimizer=optimizer,
                           token_encoder=token_encoder,
                           epoch=epoch,
                           train_teacher_forcing_probability=train_teacher_forcing_probability)

        train_losses.append(train_loss.avg)

        (recent_bleu4, recent_bleu1, recent_bleu2,
         recent_loss, recent_rouge1, recent_meteor) = validate(val_loader=val_loader,
                                                               model=model,
                                                               criterion=criterion,
                                                               token_encoder=token_encoder,
                                                               teacher_forcing=val_teacher_forcing)

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
            best_bleu4 = recent_bleu4
            best_bleu1 = recent_bleu1
            best_bleu2 = recent_bleu2
            best_loss = recent_loss
            best_rouge1 = recent_rouge1
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

        save_checkpoint(output_folder, model_name, epoch, epochs_since_improvement,
                        model, optimizer, encoder_config_dict, decoder_config_dict, recent_meteor, meteor_str, is_best)

        print(
            "Best Validation: \n"
            f"BLEU1-Score: {best_bleu1:.4f}--\t"
            f"BLEU2-Score: {best_bleu2:.4f}--\t"
            f"BLEU4-Score: {best_bleu4:.4f}\n"
            f"ROUGE1: P - {best_rouge1['rouge1_precision']:.4f}, "
            f"R - {best_rouge1['rouge1_recall']:.4f}, F - {best_rouge1['rouge1_f1_score']:.4f}\n"
            f"METEOR-Score: {best_meteor:.4f}\n"
            f"Loss: {best_loss.val:.3f} ({best_loss.avg:.3f})\n"
        )

        if debug_mode:
            break

    best_checkpoint = torch.load(os.path.join(output_folder, 'BEST_checkpoint_' +
                                              model_name + "_METEOR-" + meteor_str + '.pth.tar'))
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
          f"TEST METEOR-Score: {test_meteor:.4f}\n")

    os.rename(os.path.join(output_folder, 'BEST_checkpoint_' + model_name + "_METEOR-" + meteor_str + '.pth.tar'),
              os.path.join(output_folder, 'BEST_checkpoint_' + model_name + "_TEST_METEOR-" +
                           str(round(test_meteor, 3)) + '.pth.tar'))

    print("###entire training runtime:###")
    print(time.strftime("%H:%M:%S", time.gmtime((time.time() - global_start))))

    save_results_to_csv(output_folder, epoch, train_losses, val_meteor_scores, test_bleu4,
                        test_bleu1, test_bleu2, test_rouge1, test_meteor)


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

        teacher_forcing = np.random.choice([True, False],
                                           p=[train_teacher_forcing_probability,
                                              1 - train_teacher_forcing_probability])
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
            with open('lstm_tokenized_hypotheses.pkl', 'wb') as f:
                pickle.dump(hypotheses, f)
        print("references:")
        print(references[0:5])
        smoothing_function = SmoothingFunction().method2
        if not teacher_forcing:
            with open('lstm_tokenized_references.pkl', 'wb') as f:
                pickle.dump(references, f)

        # Prepare for BLEU, ROUGE, METEOR
        tokenized_hypotheses = word_tokenize_caption_list(hypotheses)
        tokenized_references = []
        for ref in references:
            tokenized_references.append(word_tokenize_caption_list(ref))

        bleu4 = corpus_bleu(tokenized_references, tokenized_hypotheses,
                            smoothing_function=smoothing_function)
        bleu1 = corpus_bleu(tokenized_references, tokenized_hypotheses, weights=[1, 0, 0, 0],
                            smoothing_function=smoothing_function)
        bleu2 = corpus_bleu(tokenized_references, tokenized_hypotheses, weights=[0.5, 0.5, 0, 0],
                            smoothing_function=smoothing_function)
        rouge1 = evaluate_rouge_1(tokenized_references, tokenized_hypotheses)
        meteor = mean(
            meteor_score(ref, hyp)
            for ref, hyp in zip(tokenized_references, tokenized_hypotheses)
        )

        print(
            f'\n * LOSS - {losses.avg:.3f}, TOP-5 ACCURACY - {top5accs.avg:.3f}, '
            f'BLEU-1 - {bleu1}, BLEU-2 - {bleu2}, BLEU-4 - {bleu4}, '
            f'ROUGE1: P - {rouge1["rouge1_precision"]:.4f}, '
            f'R - {rouge1["rouge1_recall"]:.4f}, '
            f'F - {rouge1["rouge1_f1_score"]:.4f}, METEOR - {meteor:.4f}\n'
        )

    return bleu4, bleu1, bleu2, losses, rouge1, meteor


if __name__ == '__main__':
    main()
