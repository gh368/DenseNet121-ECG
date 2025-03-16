import time
import numpy as np
import torch
from my_datasets import TransformerDataset
import pickle
import configparser
import ast
import os
import torch.nn.functional as F
from models import Encoder
from GPT import GPT, GPTConfig
import tqdm
from utils import adjust_learning_rate, save_checkpoint, clip_gradient, word_tokenize_caption_list, \
    save_results_to_csv
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from utils import AverageMeter, evaluate_rouge_1
import random
from nltk.translate.meteor_score import meteor_score
from statistics import mean


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors

output_folder = "./"


class Model(torch.nn.Module):
    def __init__(self, gpt_model_args, ecg_encoder, token_encoder, ecg_input_length):
        super().__init__()
        gptconf = GPTConfig(**gpt_model_args)
        self.token_encoder = token_encoder
        self.ecg_encoder = ecg_encoder  # (B, C, N) -> (B, D)
        self.ecg_input_length = ecg_input_length
        self.decoder = GPT(gptconf,
                           ecg_emb_dim=self.ecg_encoder.embedding_size,
                           padding_idx=token_encoder.vocab['<|pad|>'])  # (B, K+1, D) -> (B, K+1, D)

    def forward(self, input, targets=None, ecg_emb=None):
        ECG, input_tokens = input
        if ecg_emb is None: 
            ecg_emb = self.ecg_encoder(ECG)  # (B, n_chan, D)
        logits = self.decoder(ecg_emb, input_tokens)  # (B, K+1, V)
        if targets is not None:
            if (self.ecg_input_length-1) > 0:
                padding = torch.tensor([self.token_encoder.vocab["<|pad|>"]], dtype=torch.int).unsqueeze(1). \
                    repeat(input_tokens.size()[0], (self.ecg_input_length-1)).to(device)
                targets = torch.cat([padding, targets], dim=1)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.reshape(-1), ignore_index=self.token_encoder.vocab['<|pad|>'])
            return logits, targets, loss
        return logits, targets

    @torch.no_grad()
    def generate(self, input, max_new_tokens, eos_idx, temperature=1.0, top_k=None):
        """
        Take an ecg signal idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        ecg = input[0]
        sentence_tokens = torch.tensor([], dtype=torch.int).to(ecg.device)
        ecg_emb = self.ecg_encoder(ecg.unsqueeze(0))  # (B, 1, D)
        for _ in range(max_new_tokens):
            # forward the model to get the logits for the index in the sequence
            logits, targets = self(input, ecg_emb=ecg_emb)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            sentence_tokens = torch.cat((sentence_tokens, idx_next), dim=1)
            if idx_next[0][0] == eos_idx:
                return sentence_tokens
            input[1] = torch.cat((input[1], idx_next), dim=1)

        return sentence_tokens


def calculate_total_parameters(model):
    total_params = model.ecg_encoder.get_num_params() + model.decoder.get_num_params()
    return total_params


def main():
    global_start = time.time()
    global print_freq, sanity_check, debug_mode

    config = configparser.ConfigParser()
    config.read(os.path.join("configs", "config_train_transformer_decoder.ini"))

    # Print the contents of the config file
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
    kernel_size_str = config.get("encoder_parameters", "kernel_size")
    kernel_size = ast.literal_eval(kernel_size_str)
    stem_kernel_size = config.getint("encoder_parameters", "stem_kernel_size")
    ecg_input_length = config.getint("encoder_parameters", "encoded_ecg_size")

    # Decoder parameters
    dropout = config.getfloat("decoder_parameters", "dropout")
    start_token = config.getboolean("decoder_parameters", "start_token")
    n_layer = config.getint("decoder_parameters", "num_layers")
    n_head = config.getint("decoder_parameters", "num_heads")
    bias = config.getboolean("decoder_parameters", "bias")
    embed_dim = config.getint("decoder_parameters", "emb_dim")

    # Training parameters
    start_epoch = config.getint("training_parameters", "start_epoch")
    epochs = config.getint("training_parameters", "epochs")
    batch_size = config.getint("training_parameters", "batch_size")
    encoder_lr = config.getfloat("training_parameters", "encoder_lr")
    decoder_lr = config.getfloat("training_parameters", "decoder_lr")
    grad_clip = config.getfloat("training_parameters", "grad_clip")
    remove_start_token = config.getboolean("training_parameters", "remove_start_token")
    print_freq = config.getint("training_parameters", "print_freq")
    top_k = config.getint("training_parameters", "top_k")
    early_stopping_trigger = config.getint("training_parameters", "early_stopping_trigger")
    debug_mode = config.getboolean("training_parameters", "debug_mode")
    sanity_check = config.getboolean("training_parameters", "sanity_check")
    normalize_ds = config.getboolean("training_parameters", "normalize_ds")
    adjust_learning_rate_after = config.getint("training_parameters", "adjust_learning_rate_after")


    print("********************************************************************")
    print("output model name is: " + model_name)
    print("********************************************************************")

    print(device)

    with open(os.path.join(data_folder, 'encoder.pkl'), 'rb') as f:
        token_encoder = pickle.load(f)


    train_ds = TransformerDataset(data_folder, 'TRAIN')
    val_ds = TransformerDataset(data_folder, 'VAL')
    test_ds = TransformerDataset(data_folder, 'TEST')

    if train_ds.secgs.ndim == 2:
        train_ds.secgs = np.expand_dims(train_ds.secgs, 1)
        val_ds.secgs = np.expand_dims(val_ds.secgs, 1)
        test_ds.secgs = np.expand_dims(test_ds.secgs, 1)

    if normalize_ds == True:
        # compute mean and std along first and 3 dimensions (batch and time), and keep mean of each channel. then normalize every channel by its std and mean.
        train_ds_mean = train_ds.secgs.mean(axis=(0, 2), keepdims=True)
        train_ds_std =  train_ds.secgs.std(axis=(0, 2), keepdims=True)
        train_ds.secgs = (train_ds.secgs - train_ds_mean) / train_ds_std
        val_ds.secgs = (val_ds.secgs - train_ds_mean) / train_ds_std
        test_ds.secgs = (test_ds.secgs - train_ds_mean) / train_ds_std

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    in_chan = train_ds.secgs.shape[1]
    ecg_encoder = Encoder(num_blocks=encoder_blocks, encoded_ecg_size=ecg_input_length,
                          kernel_size=kernel_size, stem_kernel_size=stem_kernel_size, in_chan=in_chan)
    ecg_encoder.to(device)

    block_size = train_ds.captions.shape[1] + ecg_input_length - 1
    gpt_model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=embed_dim, block_size=block_size,
                      bias=bias, vocab_size=token_encoder.vocab_size,
                      dropout=dropout)  # start with model_args from command line

    model = Model(gpt_model_args, ecg_encoder, token_encoder, ecg_input_length)
    model.to(device)

    total_params = calculate_total_parameters(model)
    print(f"Total number of parameters: {total_params/1e6:.2f}M")
    optim = torch.optim.Adam(
        params=[
            {'params': filter(lambda p: p.requires_grad, model.ecg_encoder.parameters()),
             'lr': encoder_lr,
             'finetune': True,},
            {'params': filter(lambda p: p.requires_grad, model.decoder.parameters()),
             'lr': decoder_lr,
             'finetune': True}
        ]
    )

    epochs_since_improvement = 0
    best_bleu4 = 0
    best_bleu1 = 0
    best_bleu2 = 0
    best_rouge1 = {
        'rouge1_precision': 0,
        'rouge1_recall': 0,
        'rouge1_f1_score': 0
    }
    best_meteor = -1
    meteor_str = "0"
    best_loss = AverageMeter()
    train_losses = []
    val_losses = []
    val_meteor_scores = []
    for epoch in range(start_epoch, epochs):
        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == early_stopping_trigger:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % adjust_learning_rate_after == 0:
            adjust_learning_rate(optim, 0.8)
        # One epoch's training
        train_loss = train(train_loader=train_loader,
                           model=model,
                           optim=optim,
                           epoch=epoch,
                           grad_clip=grad_clip,
                           start_token=start_token,
                           remove_start_token=remove_start_token
                           )
        train_losses.append(train_loss.avg)

        # One epoch's validation
        recent_bleu4, recent_bleu1, recent_bleu2, recent_loss, recent_rouge1, recent_meteor = validate(val_loader=val_loader,
                model=model, 
                token_encoder=token_encoder, 
                block_size=block_size, top_k=top_k,
                start_token=start_token, remove_start_token=remove_start_token)

        val_losses.append(recent_loss.avg)
        val_meteor_scores.append(recent_meteor)

        is_best = recent_meteor > best_meteor
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            if epoch > 0:
                os.remove(os.path.join(output_folder, 'BEST_checkpoint_' + model_name + "_METEOR-" + meteor_str + '.pth.tar'))
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
            encoded_ecg_size=ecg_input_length,
            in_chan=in_chan
        )
        decoder_config_dict = dict(
            n_layer=n_layer,
            n_head=n_head,
            n_embd=embed_dim,
            block_size=block_size,
            bias=bias,
            vocab_size=token_encoder.vocab_size,
            dropout=dropout
        )
        # Save checkpoint
        save_checkpoint(output_folder, model_name, epoch, epochs_since_improvement, model,
                        optim, encoder_config_dict, decoder_config_dict, recent_meteor, meteor_str, is_best)

        print("Best Validation: \n"
            "BLEU1-Score: {bleu1:.4f}--\t"
            "BLEU2-Score: {bleu2:.4f}--\t"
            "BLEU4-Score: {bleu4:.4f}\n"
            "ROUGE1: P - {rouge1_p:.4f}, R - {rouge1_r:.4f}, F - {rouge1_f:.4f}\n"
            "METEOR-Score: {meteor:.4f}\n"
            'Loss: {loss.val:.3f} ({loss.avg:.3f})\n'.format(bleu1=best_bleu1,
                                                                          bleu4=best_bleu4,
                                                                          bleu2=best_bleu2,
                                                                          rouge1_p=best_rouge1['rouge1_precision'],
                                                                          rouge1_r=best_rouge1['rouge1_recall'],
                                                                          rouge1_f=best_rouge1['rouge1_f1_score'],
                                                                          meteor=best_meteor,
                                                                          loss=best_loss))
        if debug_mode:
            break
    best_checkpoint = torch.load(
        os.path.join(os.path.join(output_folder, 'BEST_checkpoint_' + model_name + "_METEOR-" + meteor_str + '.pth.tar')))
    model.load_state_dict(best_checkpoint['model'])

    print("Testing model:")
    test_bleu4, test_bleu1, test_bleu2, test_loss, test_rouge1, test_meteor = validate(val_loader=test_loader,
            model=model,
            token_encoder=token_encoder,
            block_size=block_size, top_k=top_k,
            start_token=start_token, remove_start_token=remove_start_token)

    print("\n"
          "TEST BLEU4-Score: {bleu4:.4f}\n"
          "TEST BLEU1-Score: {bleu1:.4f}\n"
          "TEST BLEU2-Score: {bleu2:.4f}\n"
          "ROUGE1: P - {rouge1_p:.4f}, R - {rouge1_r:.4f}, F - {rouge1_f:.4f}\n"
          "TEST METEOR-Score: {meteor:.4f}\n".format(bleu4=test_bleu4, 
                                                     bleu1=test_bleu1, 
                                                     bleu2=test_bleu2,
                                                     rouge1_p=test_rouge1['rouge1_precision'],
                                                     rouge1_r=test_rouge1['rouge1_recall'],
                                                     rouge1_f=test_rouge1['rouge1_f1_score'],
                                                     meteor=test_meteor))
    os.rename(os.path.join(output_folder, 'BEST_checkpoint_' + model_name + "_METEOR-" + meteor_str + '.pth.tar'),
              os.path.join(output_folder, 'BEST_checkpoint_' + model_name + "_TEST_METEOR-" + str(round(test_meteor, 3)) + '.pth.tar'))

    print("###entire training runtime:###")
    print(time.strftime("%H:%M:%S", time.gmtime((time.time() - global_start))))

    save_results_to_csv(output_folder, epoch, train_losses, val_meteor_scores, test_bleu4, test_bleu1,
                        test_bleu2, test_rouge1, test_meteor)

def train(train_loader,
          model, optim,
          epoch,
          grad_clip,
          start_token, remove_start_token
          ):

    model.train()
    model.ecg_encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)

    start = time.time()
    for i, ((ecgs, captions), targets, caplens) in enumerate(tqdm.tqdm(train_loader)):
        data_time.update(time.time() - start)
        if sanity_check:
            ecgs.fill_(1)
        ecgs = ecgs.to(device)
        captions = captions.to(device)
        targets = targets.to(device)
        caplens = caplens.to(device)
        if start_token and remove_start_token:
            captions = captions[:, 1:]
            targets = targets[:, 1:]
        logits, output_targets, loss = model((ecgs, captions), targets)
        loss.backward()
        # Clip gradients
        if grad_clip is not None:
            clip_gradient(optim, grad_clip)

        optim.step()
        optim.zero_grad(set_to_none=True)

        losses.update(loss.item(), sum(caplens).item())
        batch_time.update(time.time() - start)

        if i % print_freq == 0:
            print('\nEpoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(str(epoch), str(i), str(len(train_loader)),
                                                                batch_time=batch_time,
                                                                data_time=data_time,
                                                                loss=losses))
        start = time.time()
    return losses
def validate(val_loader, model, token_encoder, block_size, top_k, start_token, remove_start_token):
    model.eval()
    model.ecg_encoder.eval()

    losses = AverageMeter()
    batch_time = AverageMeter()

    start = time.time()

    eos_idx = token_encoder.vocab["<|endoftext|>"]
    if start_token and not remove_start_token:
        sos_idx = token_encoder.vocab["<|startoftext|>"]
    hypotheses = []
    references_list = []
    with torch.no_grad():
        for i, ((ecgs, captions), targets, caplens) in tqdm.tqdm(enumerate(val_loader)):
            if sanity_check:
                ecgs = torch.ones((len(captions), 1, 7663)) # for sanity check
            ecgs = ecgs.to(device)
            captions = captions.to(device)
            targets = targets.to(device)
            if start_token and not remove_start_token:
                input = [[ecg, torch.tensor([sos_idx]).unsqueeze(1).to(device)] for ecg in ecgs]
                block_size = block_size-1
            else:
                input = [[ecg, torch.tensor([], dtype=torch.int).to(device)] for ecg in ecgs]
            y = [model.generate(sample, block_size, eos_idx, top_k=top_k).squeeze(0) for sample in input]
            logits, targets, loss = model((ecgs, captions), targets)
            losses.update(loss.item(), sum(caplens).item())
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('\nValidation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i, len(val_loader),
                                                                                batch_time=batch_time,
                                                                                loss=losses))

            for caption in captions:
                cur_cap = []
                for token in caption:
                    if token != token_encoder.vocab['<|pad|>']:
                        cur_cap.append(token)
                    else:
                        break
                references_list.append([token_encoder.decode(cur_cap, skip_special_tokens=True)])
            for caption in y:
                cur_cap = []
                for token in caption:
                    if token != token_encoder.vocab['<|pad|>']:
                        cur_cap.append(token)
                    else:
                        break
                hypotheses.append(token_encoder.decode(cur_cap, skip_special_tokens=True))


        tokenized_hypotheses = word_tokenize_caption_list(hypotheses, list(token_encoder.special_tokens.keys()))
        tokenized_references = [word_tokenize_caption_list(reference, list(token_encoder.special_tokens.keys())) for reference in references_list]
        print("predictions example:")
        print(tokenized_hypotheses[0:5])
        with open('transformer_tokenized_hypotheses.pkl', 'wb') as f:
            pickle.dump(tokenized_hypotheses, f)
        print("references:")
        print(tokenized_references[0:5])
        with open('transformer_tokenized_references.pkl', 'wb') as f:
            pickle.dump(tokenized_hypotheses, f)
        smoothing_function = SmoothingFunction().method2
        bleu4 = corpus_bleu(tokenized_references, tokenized_hypotheses, smoothing_function=smoothing_function)
        bleu1 = corpus_bleu(tokenized_references, tokenized_hypotheses, weights=[1, 0, 0, 0],
                            smoothing_function=smoothing_function)
        bleu2 = corpus_bleu(tokenized_references, tokenized_hypotheses, weights=[0.5, 0.5, 0, 0],
                            smoothing_function=smoothing_function)
        rouge1 = evaluate_rouge_1(tokenized_references, tokenized_hypotheses)
        meteor = mean(meteor_score(cur_reference, cur_hypothesis) for cur_reference, cur_hypothesis
                  in zip(tokenized_references, tokenized_hypotheses))


        print(
                '\n * BLEU-1 - {bleu1:.4f}, BLEU-2 - {bleu2:.4f}, BLEU-4 - {bleu4:.4f}, ' 
            'ROUGE1: P - {rouge1_p:.4f}, R - {rouge1_r:.4f}, F - {rouge1_f:.4f} METEOR - {meteor:.4f}\n'.format(
                bleu1=bleu1,
                bleu4=bleu4,
                bleu2=bleu2,
                rouge1_p=rouge1['rouge1_precision'],
                rouge1_r=rouge1['rouge1_recall'],
                rouge1_f=rouge1['rouge1_f1_score'],
                meteor=meteor))

    return bleu4, bleu1, bleu2, losses, rouge1, meteor


if __name__ == '__main__':
    main()
    random.seed(10)
    torch.manual_seed(10)
    np.random.seed(10)
