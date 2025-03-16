import torch  
import torch.nn as nn  
import DenseNet1D
import numpy as np
import torchvision.models as models

class Model(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, batch, caps=None, caplens=None, teacher_forcing=True, token_encoder=None, max_caption_len=None):
        x = self.encoder(batch)
        y = self.decoder(x, caps, caplens, teacher_forcing, token_encoder, max_caption_len)

        return y
       
class Encoder(nn.Module):
    def __init__(self, num_blocks, encoded_ecg_size, kernel_size, stem_kernel_size, in_chan):
        super(Encoder, self).__init__()
        
        # If you're using a custom 1D version of DenseNet, make sure
        # to define `densenet121` inside your DenseNet1D module.
        # Otherwise, use the built-in torchvision method:
        if num_blocks == 121:
            # Use the built-in PyTorch model (2D DenseNet for images)
            # Turn off pretrained if you don't want ImageNet weights
            self.model = models.densenet121(weights=False)
            self.model.features.conv0 = nn.Conv1d(
                in_channels=12,   # <--- changed from 3
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            )
        else:
            raise ValueError("Unsupported number of num_blocks; only 121 is allowed.")


    def forward(self, x):
        return self.densenet(x)
        
        # Remove linear and pool layers (since we're not doing classification)
        self.embedding_size = densenet.fc.in_features
        densenet.avgpool = torch.nn.Identity()
        densenet.fc = torch.nn.Identity()
        self.model = densenet
        self.adaptive_pool = torch.nn.AdaptiveAvgPool1d(encoded_ecg_size)

        print("number of Encoder parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, batch):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.model(batch)  # (batch_size, ecg_size) --> (batch_size, number of output channels, DenseNet output size)
        out = self.adaptive_pool(out)  # (batch_size, number of output channels, encoded_ecg_size)
        out = out.permute(0, 2, 1)  # (batch_size, encoded_ecg_size, number of output channels)
        return out


class Attention(torch.nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = torch.nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = torch.nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = torch.nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, downsampled_signal_length, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, embed_length)
        alpha = self.softmax(att)  # (batch_size, downsampled_signal_length)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class DecoderWithAttention(torch.nn.Module):
    """
    Decoder.
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, token_encoder, encoder_dim, layers, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = token_encoder.vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = torch.nn.Embedding(self.vocab_size, embed_dim)  # embedding layer
        self.dropout = torch.nn.Dropout(p=self.dropout)
        self.decode_step = torch.nn.LSTM(embed_dim + encoder_dim, decoder_dim, num_layers=layers, bias=True)  # decoding LSTMCell
        self.init_h = torch.nn.Linear(encoder_dim, layers * decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = torch.nn.Linear(encoder_dim, layers * decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = torch.nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = torch.nn.Sigmoid()
        self.fc = torch.nn.Linear(decoder_dim, self.vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

        print("number of LSTM-decoder parameters: %.2fM" % (self.get_num_params() / 1e6,))
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.

        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = torch.nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1) # later we will aggregate with alphas, for now we do a simple mean
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions=None, caption_lengths=None, teacher_forcing=True, token_encoder=None,
                max_length=None):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :param teacher_forcing: boolean whether in each step to feed next ground truth or next predicted token
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """
        device = encoder_out.device

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        if teacher_forcing:
            # Sort input data by decreasing lengths; why? apparent below
            caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
            encoder_out = encoder_out[sort_ind]

            # Initialize LSTM state
            h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)
            h = h.reshape(batch_size, -1, self.decoder_dim).permute(1, 0, 2).contiguous()
            c = c.reshape(batch_size, -1, self.decoder_dim).permute(1, 0, 2).contiguous()
            encoded_captions = encoded_captions[sort_ind]
            # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
            # So, decoding lengths are actual lengths - 1
            decode_lengths = (caption_lengths - 1).tolist()
            # Embedding
            embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)
            # Create tensors to hold word prediction scores and alphas
            predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
            alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)
            for t in range(max(decode_lengths)):
                batch_size_t = sum([l > t for l in decode_lengths])
                h_pooled = h.sum(0)[:batch_size_t]
                attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                    h_pooled)  # (batch_size_t, encoder_dim), (batch_size_t, num_pixels)
                gate = self.sigmoid(self.f_beta(h_pooled))  # gating scalar, (batch_size_t, encoder_dim)
                attention_weighted_encoding = gate * attention_weighted_encoding
                h_last, (h, c) = self.decode_step(
                    torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1).unsqueeze(0),
                    (h[:, :batch_size_t].contiguous(),
                     c[:, :batch_size_t].contiguous()))  # (batch_size_t, decoder_dim)
                preds = self.fc(self.dropout(h_last.squeeze(0)))  # (batch_size_t, vocab_size)
                predictions[:batch_size_t, t, :] = preds
                alphas[:batch_size_t, t, :] = alpha
        else:
            # Initialize LSTM state
            h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)
            h = h.reshape(batch_size, -1, self.decoder_dim).permute(1, 0, 2).contiguous()
            c = c.reshape(batch_size, -1, self.decoder_dim).permute(1, 0, 2).contiguous()

            # Create tensors to hold word prediction scores and alphas
            predictions = torch.zeros(batch_size, max_length, vocab_size).to(device)
            alphas = torch.zeros(batch_size, max_length, num_pixels).to(device)

            prev_words = torch.LongTensor([[token_encoder.special_tokens['<|startoftext|>']]] * batch_size).to(
                device)
            is_seq_complete = np.full(len(encoder_out), False)
            decode_lengths = torch.ones(len(encoder_out))
            step = 0
            while step < max_length:
                embeddings = self.embedding(prev_words).squeeze(1)  # (s, embed_dim)
                h_pooled = h.sum(0)
                attention_weighted_encoding, alpha = self.attention(encoder_out,
                                                                    h_pooled)  # (s, encoder_dim), (s, num_pixels)
                alpha = alpha.view(-1, encoder_out.size(1))  # (s, enc_image_size, enc_image_size)

                gate = self.sigmoid(self.f_beta(h_pooled))  # gating scalar, (s, encoder_dim)
                attention_weighted_encoding = gate * attention_weighted_encoding

                h_last, (h, c) = self.decode_step(
                    torch.cat([embeddings, attention_weighted_encoding], dim=1).unsqueeze(0),
                    (h, c))  # (s, decoder_dim)
                preds = self.fc(self.dropout(h_last.squeeze(0)))  # (batch_size_t, vocab_size)
                predictions[:, step, :] = preds
                alphas[:, step, :] = alpha

                next_word_inds = torch.argmax(preds, 1)

                # Which sequences are incomplete (didn't reach <end>)?
                cur_incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                                       next_word != token_encoder.special_tokens['<|endoftext|>']]
                # all complete sequences (previous + new)
                cur_complete_inds = list(set(range(len(next_word_inds))) - set(cur_incomplete_inds))
                # only the sequences that reached <|end|> in this iteration
                new_complete_inds = list(
                    set(cur_complete_inds) - set(i for i, is_complete in enumerate(is_seq_complete) if is_complete))

                if len(new_complete_inds) > 0:
                    is_seq_complete[new_complete_inds] = True
                    decode_lengths[
                        new_complete_inds] = step + 1  # lengths is the sequence with start but witout end token

                prev_words = next_word_inds.unsqueeze(1)

                if np.all(is_seq_complete):
                    break

                step += 1

            if not np.all(is_seq_complete):
                incomplete_ids = [i for i, is_complete in enumerate(is_seq_complete) if not is_complete]
                decode_lengths[incomplete_ids] = max_length - 1  # -1 because we don't include end token

            decode_lengths, sort_ind = decode_lengths.sort(descending=True)
            if encoded_captions is not None:
                encoded_captions = encoded_captions[sort_ind]
            predictions = predictions[sort_ind]
            alphas = alphas[sort_ind]
            decode_lengths = decode_lengths.int().tolist()
        return predictions, encoded_captions, decode_lengths, alphas, sort_ind

