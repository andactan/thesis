import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from scipy.stats import spearmanr


VOCAB_SIZE = 264

def lstm_helper(sequences, lengths, lstm):
    if len(sequences) == 1:
        output, (hidden, _) = lstm(sequences)
        return output, hidden[-1]

    ordered_len, ordered_idx = lengths.sort(0, descending=True)
    ordered_sequences = sequences[ordered_idx]
    # remove zero lengths
    try:
        nonzero = list(ordered_len).index(0)
    except ValueError:
        nonzero = len(ordered_len)

    sequences_packed = pack_padded_sequence(
        ordered_sequences[:nonzero], ordered_len[:nonzero],
        batch_first=True)
    output_nonzero, (hidden_nonzero, _) = lstm(sequences_packed)
    output_nonzero = pad_packed_sequence(output_nonzero, batch_first=True)[0]
    max_len = sequences.shape[1]
    max_len_true = output_nonzero.shape[1]
    output = torch.zeros(len(sequences), max_len, output_nonzero.shape[-1])
    output_final = torch.zeros(
        len(sequences), max_len, output_nonzero.shape[-1])
    output[:nonzero, :max_len_true, :] = output_nonzero
    hidden = torch.zeros(len(sequences), hidden_nonzero.shape[-1])
    hidden_final = torch.zeros(len(sequences), hidden_nonzero.shape[-1])
    hidden[:nonzero, :] = hidden_nonzero[-1]
    output_final[ordered_idx] = output
    hidden_final[ordered_idx] = hidden
    return output_final.cuda(), hidden_final.cuda()

class ImgEnc(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.encoder = nn.Sequential(
            nn.Conv2d(3, args.n_channels, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(args.n_channels, args.n_channels, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(args.n_channels, args.n_channels, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(4*4*args.n_channels, args.img_enc_size),
            nn.Linear(args.img_enc_size, args.img_enc_size),
        )

    def forward(self, x):
        return self.encoder(x)

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.img_enc_r = ImgEnc(args)
        self.img_enc_l = ImgEnc(args)
        self.img_enc_c = ImgEnc(args)

        # trajectory encoder
        self.traj_encoder = nn.LSTM(
            3 * args.img_enc_size, 
            args.img_enc_size, 
            batch_first=True, 
            num_layers=args.num_layers)

        # language encoder
        # self.embedding = nn.Embedding(VOCAB_SIZE, args.lang_enc_size)
        # self.descr_encoder = nn.LSTM(args.lang_enc_size, args.lang_enc_size, batch_first=True, num_layers=args.num_layers)

        # linear layers
        self.linear1 = nn.Linear(args.img_enc_size + args.lang_enc_size, args.classifier_size)
        self.linear2 = nn.Linear(args.classifier_size, 1)


    def forward(self, traj_r, traj_l, traj_c, traj_len):
        traj_r_enc = self.img_enc_r(traj_r.view(-1, *traj_r.shape[-3:]))
        traj_r_enc = traj_r_enc.view(*traj_r.shape[:2], -1)
        traj_l_enc = self.img_enc_l(traj_l.view(-1, *traj_l.shape[-3:]))
        traj_l_enc = traj_l_enc.view(*traj_l.shape[:2], -1)
        traj_c_enc = self.img_enc_c(traj_c.view(-1, *traj_c.shape[-3:]))
        traj_c_enc = traj_c_enc.view(*traj_c.shape[:2], -1)

        traj_enc = torch.cat([traj_r_enc, traj_l_enc, traj_c_enc], dim=-1)
        _, traj_enc = lstm_helper(traj_enc, traj_len, self.traj_encoder)

        # lang_emb = self.embedding(lang)
        # _, lang_enc = lstm_helper(lang_emb, lang_len, self.descr_encoder)

        # traj_lang = torch.cat([traj_enc, lang_enc], dim=-1)
        # pred = F.relu(self.linear1(traj_lang))
        # pred = self.linear2(pred)
        return traj_enc


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-channels', type=int, default=256)
    parser.add_argument('--img-enc-size', type=int, default=512)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--lang-enc-size', type=int, default=512)
    parser.add_argument('--classifier-size', type=int, default=1024)
    args = parser.parse_args()

    m = Model(args=args)
    x = torch.rand(1, 2, 3, 50, 50)
    print(m(x, x, x, 1).shape)