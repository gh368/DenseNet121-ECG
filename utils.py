import os

import numpy as np
import h5py

import pandas as pd
import sklearn.model_selection
import torch
import re
from collections import Counter
import pickle
from rouge_score import rouge_scorer
import wfdb
import ast
from easynmt import EasyNMT

class Word_encoder():
    def __init__(self, word_map, special_tokens):
        self.vocab = word_map
        self.rev_word_map = {v: k for k, v in word_map.items()}
        self.special_tokens = {token: word_map[token] for token in special_tokens}
        self.vocab_size = len(word_map)

    def decode(self, ids, skip_special_tokens=False):
        if skip_special_tokens:
            decoded_caption = [self.rev_word_map[int(token_id)] for token_id in ids
                               if token_id not in self.special_tokens.values()]
        else:
            decoded_caption = [self.rev_word_map[int(token_id)] for token_id in ids]

        return ' '.join(decoded_caption)

    def encode(self, caption, pad_to_length=-1, add_special_tokens=True):
        tokenized_caption = word_tokenize_caption(caption)
        encoded_caption = [self.vocab[word] if word in self.vocab else self.vocab['<|unk|>'] for word in tokenized_caption]
        if add_special_tokens:
            encoded_caption = ([self.special_tokens['<|startoftext|>']] +
                                 encoded_caption +
                                 [self.special_tokens['<|endoftext|>']])
        caplen = len(encoded_caption)
        if pad_to_length > 0:
            encoded_caption = encoded_caption[:pad_to_length]
            encoded_caption += [self.special_tokens['<|pad|>']] * (pad_to_length - len(encoded_caption))
            return encoded_caption, caplen

def replace_med_abbr(df, column_name, new_column_name):
    """
    Replaces the medical abbreviations from the input df column and writes them into the new_column_name

    Args:
        df - the dataframe
        column_name - column name of the column to clean
        new_column_name - the output column name
    Returns:
        df - where the column has been updated
    """

    # all lower case
    df[new_column_name] = df[column_name].apply(lambda x: x.lower())

    df[new_column_name] = df[new_column_name].apply(
        lambda x: x.replace("@", " at ")
        .replace("+", " and ")
        .replace("&", " and "))

    df[new_column_name] = df[new_column_name].apply(
        lambda x: re.sub(r"(\bpacs?\b|\bsves?\b|\bapcs?\b)",
                         " premature atrial contraction ", str(x),
                         )
    )
    df[new_column_name] = df[new_column_name].apply(
        lambda x: re.sub(r"(\bpvcs?\b|\bvpcs?\b|\bves\b)",
                         " premature ventricular contraction ", str(x),
                         )
    )
    df[new_column_name] = df[new_column_name].apply(
        lambda x: re.sub(r"(\bectopics?\b|\bectopy\b)",
                         " premature contraction ", str(x),
                         )
    )
    df[new_column_name] = df[new_column_name].apply(
        lambda x: re.sub(r"(\bcontractions?\b)",
                         " contraction ", str(x),
                         )
    )
    df[new_column_name] = df[new_column_name].apply(
        lambda x: re.sub(r"(\bbrady?\b)",
                         " bradycardia ", str(x)
                         )
    )
    df[new_column_name] = df[new_column_name].apply(
        lambda x: re.sub(r"(\bsb\b)",
                         " sinus bradycardia ", str(x))
    )
    df[new_column_name] = df[new_column_name].apply(
        lambda x: re.sub(r"(\btachy?\b)",
                         " tachycardia ", str(x)
                         )
    )
    df[new_column_name] = df[new_column_name].apply(
        lambda x: re.sub(r"(\bst\b)",
                         " sinus tachycardia ", str(x))
    )
    df[new_column_name] = df[new_column_name].apply(
        lambda x: re.sub(r"(\bsvt\b)",
                         " supraventricular tachycardia ", str(x))
    )
    df[new_column_name] = df[new_column_name].apply(
        lambda x: re.sub(r"(\bnsvt\b)",
                         " nonsustained ventricular tachycardia ", str(x),
                         )
    )
    df[new_column_name] = df[new_column_name].apply(
        lambda x: re.sub(r"(\bsr\b)",
                         " sinus rhythm ", str(x))
    )
    df[new_column_name] = df[new_column_name].apply(
        lambda x: re.sub(r"(\bnsr\b)",
                         " normal sinus rhythm ", str(x))
    )
    df[new_column_name] = df[new_column_name].apply(
        lambda x: re.sub(r"(\baf\b|\ba[- ]?fib\b)",
                         " atrial fibrillation ", str(x))
    )
    df[new_column_name] = df[new_column_name].apply(
        lambda x: re.sub(r"(\ba[- ]?fl\b|\ba[- ]?flutter\b)",
                         " atrial flutter ", str(x))
    )
    df[new_column_name] = df[new_column_name].apply(
        lambda x: re.sub(r"(\bcw\b)",
                         " continuous wave ", str(x))
    )
    df[new_column_name] = df[new_column_name].apply(
        lambda x: re.sub(r"(\brvr\b)",
                         " rapid ventricular rate ", str(x)
                         )
    )
    df[new_column_name] = df[new_column_name].apply(
        lambda x: re.sub(r"(\bppm\b)",
                         " permanent pacemaker ", str(x))
    )
    df[new_column_name] = df[new_column_name].apply(
        lambda x: re.sub(r"(bpm\b)",
                         " beats per minute ", str(x))
    )
    df[new_column_name] = df[new_column_name].apply(
        lambda x: re.sub(r"(\bpa?t\b)",
                         " patient ", str(x))
    )
    df[new_column_name] = df[new_column_name].apply(
        lambda x: re.sub(r"(bts\b)",
                         " beats ", str(x))
    )
    df[new_column_name] = df[new_column_name].apply(
        lambda x: re.sub(r"(\bw/?o\b)",
                         " without ", str(x))
    )
    df[new_column_name] = df[new_column_name].apply(
        lambda x: re.sub(r"(\bw\b/?)",
                         " with ", str(x))
    )
    df[new_column_name] = df[new_column_name].apply(
        lambda x: re.sub(r"(\bhr\b)",
                         " heart rate ", str(x))
    )
    df[new_column_name] = df[new_column_name].apply(
        lambda x: re.sub(r"(\bavb\b)",
                         " atrioventricular block ", str(x))
    )
    df[new_column_name] = df[new_column_name].apply(
        lambda x: re.sub(r"(  )", " ", str(x))
    )

    df[new_column_name] = df[new_column_name].apply(lambda x: x.strip())

    return df



def word_tokenize_caption(comment, exceptions=None):
    processed_comment = str.replace(comment, "'", "")
    if exceptions is None:
        split_comment = list(
            filter(None, re.split("(\\W|\\d+\\.\\d+|\\d+)", processed_comment)))
    else:
        exceptions = [re.escape(exception) for exception in exceptions]
        split_comment = list(filter(None, re.split("(" + '|'.join(exceptions) + ")|(\\W|\\d+\\.\\d+|\\d+)", processed_comment)))
    # split_comment = re.split("(\W|\d+\.\d+|\d+)", processed_comment)
    tokenized_comment = [word for word in split_comment if word != ' ' and word != '']
    # final_comments = [[sub_item.lower() for item in split_comments] for sub_item in item if sub_item != ' ' and sub_item != '']

    return tokenized_comment


def word_tokenize_caption_list(caption_list, exceptions=None):
    return [word_tokenize_caption(comment, exceptions) for comment in caption_list]


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        if param_group['finetune']:
            param_group['lr'] = param_group['lr'] * shrink_factor
            print("The new learning rate is %f\n" % (param_group['lr'],))


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def save_checkpoint(folder, model_name, epoch, epochs_since_improvement, model, optimizer,
                    encoder_config_dict, decoder_config_dict, meteor, meteor_str, is_best):
    """
    Saves model checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param meteor: validation meteor score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'meteor': meteor,
             'model': model.state_dict(),
             'optimizer': optimizer,
             'encoder_config': encoder_config_dict,
             'decoder_config': decoder_config_dict
             }
    filename = os.path.join(folder, 'checkpoint_' + model_name + '.pth.tar')
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        filename = os.path.join(folder, 'BEST_checkpoint_' + model_name + "_METEOR-" + meteor_str + '.pth.tar')
        torch.save(state, filename)

def save_results_to_csv(output_folder, epochs, train_losses, val_meteor_scores, test_bleu4, test_bleu1, test_bleu2,
                        test_rouge1, test_meteor):
    # Create a DataFrame with the results
    # Create DataFrame for training results
    results_df = pd.DataFrame({
        "Epoch": range(epochs + 1),
        "Train Loss": train_losses,
        "Validation METEOR": val_meteor_scores
    })

    # Add columns for test metric names and values, and set them to NaN for the epoch rows
    test_metric_names = ["Test BLEU-1", "Test BLEU-2", "Test BLEU-4", "Test ROUGE-1 P", "Test ROUGE-1 R",
                         "Test ROUGE-1 F", "Test METEOR"]
    test_metric_values = [test_bleu1, test_bleu2, test_bleu4, test_rouge1['rouge1_precision'],
                          test_rouge1['rouge1_recall'], test_rouge1['rouge1_f1_score'], test_meteor]

    # Fill the first rows with the test metrics
    results_df[""] = np.nan
    results_df["Test Metric"] = pd.Series([np.nan] * len(results_df), dtype=object)
    results_df["Test Value"] = pd.Series([np.nan] * len(results_df), dtype=object)

    # If the number of epochs (rows) is less than the number of test metrics, pad with NaN rows
    if len(results_df) < len(test_metric_names):
        missing_rows = len(test_metric_names) - len(results_df)
        padding_df = pd.DataFrame({
            "Epoch": [np.nan] * missing_rows,
            "Train Loss": [np.nan] * missing_rows,
            "Validation METEOR": [np.nan] * missing_rows,
            "": [np.nan] * missing_rows,
            "Test Metric": [np.nan] * missing_rows,
            "Test Value": [np.nan] * missing_rows
        })
        results_df = pd.concat([results_df, padding_df], ignore_index=True)

    # Fill the first rows with the test metrics
    results_df.iloc[:len(test_metric_names), results_df.columns.get_loc("Test Metric")] = test_metric_names
    results_df.iloc[:len(test_metric_values), results_df.columns.get_loc("Test Value")] = test_metric_values

    # Save to CSV
    csv_file = os.path.join(output_folder, "training_and_test_results.csv")
    results_df.to_csv(csv_file, index=False)

    print(f"Training and test results saved to {csv_file}")


def evaluate_rouge_1(references, hypotheses):
    assert len(references) == len(hypotheses), "The number of references and hypotheses should be the same"

    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    total_precision = []
    total_recall = []
    total_f1_score = []

    for ref_list, hypo_tokens in zip(references, hypotheses):
        # Join tokens to create a single string
        hypo_text = ' '.join(hypo_tokens)
        # Evaluate against all references and take the best score
        best_scores = None
        for ref_tokens in ref_list:
            ref_text = ' '.join(ref_tokens)
            scores = scorer.score(ref_text, hypo_text)['rouge1']
            if best_scores is None or scores.fmeasure > best_scores.fmeasure:
                best_scores = scores

        total_precision.append(best_scores.precision)
        total_recall.append(best_scores.recall)
        total_f1_score.append(best_scores.fmeasure)

    # Calculate average scores
    avg_precision = np.mean(total_precision)
    avg_recall = np.mean(total_recall)
    avg_f1_score = np.mean(total_f1_score)

    return {
        'rouge1_precision': avg_precision,
        'rouge1_recall': avg_recall,
        'rouge1_f1_score': avg_f1_score
    }


def load_ptb_xl_raw_data(df, sampling_rate, path):
    data = []
    for f in df.filename_lr if sampling_rate == 100 else df.filename_hr:
        signal, _ = wfdb.rdsamp(path + f)
        data.append(signal)
    return np.array(data)


def duplicate_Y(row_Y):
    if row_Y.superclass_len > 1:
        df_Y = pd.DataFrame([row_Y] * row_Y.superclass_len, columns=row_Y.index).reset_index(drop=True)
        df_Y['superclass'] = df_Y.apply(lambda row: [row_Y.superclass[row.name % row_Y.superclass_len]], axis=1)
        df_Y['superclass_len'] = 1
        return df_Y
    return pd.DataFrame([row_Y])
    # return row_Y


def duplicate_X(row_Y, row_X):
    if row_Y.superclass_len > 1:
        return np.tile(np.expand_dims(row_X, axis=0), (row_Y.superclass_len, 1, 1))
    return np.expand_dims(row_X, axis=0)

def translate_textlist(text_list):
    model = EasyNMT('opus-mt')
    return model.translate(text_list, target_lang='en', source_lang='de', show_progress_bar=True)


def create_decoder_input_files(data_folder, output_folder, sampling_rate, replace_abbr_text, min_word_freq, vocab_size,
                               max_len, translate_comments, split_method, debug_mode):

    assert sampling_rate in [100, 500]
    assert split_method in ["official", "random"]

    Y = pd.read_csv(data_folder + 'ptbxl_database.csv', index_col='ecg_id')
    if debug_mode:
        Y = Y.sample(100)
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    if translate_comments:
        translated_comments = translate_textlist(Y.report.reset_index(drop=True))
        Y.loc[:, 'report'] = translated_comments
    X = load_ptb_xl_raw_data(Y, sampling_rate, data_folder)
    X = X.transpose(0, 2, 1)

    if replace_abbr_text:
       Y = replace_med_abbr(Y, "report", "report")

    tokenized_comments = word_tokenize_caption_list(Y.report)
    tokens = [word for comment in tokenized_comments for word in comment]

    tokens_freq = Counter(tokens)

    SOS = "<|startoftext|>"
    PAD = "<|pad|>"
    EOS = "<|endoftext|>"
    UNK = "<|unk|>"
    if len(tokens_freq) > vocab_size:
        # remove the least frequent words
        tokens_freq = dict(sorted(tokens_freq.items(), key=lambda item: (-item[1], item[0]))[:vocab_size])
    print("tokens freq:")
    print(tokens_freq)
    print("vocabulary size")
    print(len(tokens_freq))

    # Create word map
    words = [w for w in tokens_freq.keys() if tokens_freq[w] >= min_word_freq]
    print("vocabulary size with min_word_freq")
    print(len(words))
    word_map = {k: v for v, k in enumerate(words)}

    word_map[UNK] = len(word_map)
    word_map[SOS] = len(word_map)
    word_map[EOS] = len(word_map)
    word_map[PAD] = len(word_map)

    enc = Word_encoder(word_map, [SOS, PAD, EOS, UNK])

    assert len(tokenized_comments) == X.shape[0]

    os.makedirs(output_folder, exist_ok=True)
    with open(os.path.join(output_folder, 'encoder.pkl'), "wb") as f:
        pickle.dump(enc, f)

    # prepare train/val/test datasets
    Y.report = word_tokenize_caption_list(Y.report, [SOS, PAD, EOS, UNK])
    def process_comment(comment, max_len, word_map, SOS, EOS, UNK, PAD):
        if len(comment) > (max_len - 2):
            comment = comment[:(max_len - 2)]
        comment = [SOS] + comment + [EOS]
        enc_c = [word_map.get(word, word_map[UNK]) for word in comment] + \
                [word_map[PAD]] * (max_len - len(comment))
        enc_c = enc_c[:max_len]
        return enc_c

    Y['encoded_comments'] = Y['report'].apply(lambda x: process_comment(x, max_len, word_map, SOS, EOS, UNK, PAD))
    Y['comment_lengths'] = [min(max_len, len(rep) + 2) for rep in Y.report]

    ecg_length = 10*sampling_rate

    pad_width = ((0, 0), (0, 12 - X.shape[1]), (0, ecg_length - X.shape[2]))
    X = np.pad(X, pad_width, mode='constant', constant_values=0)

    if split_method == "official":
        Y_train_records = Y[Y.strat_fold < 9]
        X_train_records = X[Y.strat_fold < 9]
        Y_val_records = Y[Y.strat_fold == 9]
        X_val_records = X[Y.strat_fold == 9]
        Y_test_records = Y[Y.strat_fold == 10]
        X_test_records = X[Y.strat_fold == 10]
        test_train_overlap = Y_test_records.patient_id.isin(
            Y_train_records.patient_id).any() or Y_test_records.patient_id.isin(Y_val_records.patient_id).any()
        assert ~test_train_overlap

    elif split_method == "random":
        (X_test_records, X_train_val_records,
         Y_test_records, Y_train_val_records) = (sklearn.model_selection.
                                                 train_test_split(X, Y, train_size=0.2))

        (X_train_records, X_val_records,
         Y_train_records, Y_val_records) = (sklearn.model_selection.
                                            train_test_split(X_train_val_records, Y_train_val_records,
                                                             test_size=0.2))

    print("patient overlap between test and train:")
    print(sum(Y_test_records.patient_id.isin(Y_train_records.patient_id)))
    print("patient overlap between test and val:")
    print(sum(Y_test_records.patient_id.isin(Y_val_records.patient_id)))


    for split_X, split_Y, split in [(X_train_records, Y_train_records, 'TRAIN'),
                                    (X_val_records, Y_val_records, 'VAL'),
                                    (X_test_records, Y_test_records, 'TEST')]:

        with (h5py.File(os.path.join(output_folder, split + '_DATASET.hdf5'), 'a') as h1):
            # Create dataset inside HDF5 file to store images
            h1.create_dataset('secgs', data=split_X)
            h1.create_dataset('captions', data=np.stack(split_Y['encoded_comments']))
            h1.create_dataset('caplens', data=split_Y['comment_lengths'].values)

    print('---DONE CREATING FILE---')
