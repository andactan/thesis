import torch
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def _lstm_forward(sequences, lengths, lstm):
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
        ordered_sequences[:nonzero], ordered_len[:nonzero], batch_first=True
    )
    output_nonzero, (hidden_nonzero, _) = lstm(sequences_packed)
    output_nonzero = pad_packed_sequence(output_nonzero, batch_first=True)[0]
    max_len = sequences.shape[1]
    max_len_true = output_nonzero.shape[1]
    output = torch.zeros(len(sequences), max_len, output_nonzero.shape[-1])
    output_final = torch.zeros(len(sequences), max_len, output_nonzero.shape[-1])
    output[:nonzero, :max_len_true, :] = output_nonzero
    hidden = torch.zeros(len(sequences), hidden_nonzero.shape[-1])
    hidden_final = torch.zeros(len(sequences), hidden_nonzero.shape[-1])
    hidden[:nonzero, :] = hidden_nonzero[-1]
    output_final[ordered_idx] = output
    hidden_final[ordered_idx] = hidden
    return output_final.cuda(), hidden_final.cuda()


class ImageEncoderBlock(torch.nn.Module):
    def __init__(self, input_shape=3, out_channels=3, hidden_dim=64) -> None:
        super().__init__()

        self.input_shape = input_shape
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, out_channels, 3, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(self.out_channels, self.out_channels, 3, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(self.out_channels, self.out_channels, 3, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Flatten(),
            torch.nn.Linear(4 * 4 * self.out_channels, self.hidden_dim),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class ImageEncoder(torch.nn.Module):
    def __init__(self, hidden_dim=64, num_lstm_layers=4) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_lstm_layers = num_lstm_layers

        # encoder blocks
        self.image_enc_left = ImageEncoderBlock()
        self.image_enc_center = ImageEncoderBlock()
        self.image_enc_right = ImageEncoderBlock()

        self.lstm = torch.nn.LSTM(
            3 * self.hidden_dim, self.hidden_dim, batch_first=True, num_layers=self.num_lstm_layers
        )

    def forward(self, x_left, x_center, x_right, traj_length):
        # print('shape', x_left.shape)
        # TODO: dont forget batch size
        # left view
        out_left = self.image_enc_left(x_left.view(-1, *x_left.shape[-3:]))
        out_left = out_left.view(*x_left.shape[:2], -1)

        # center view
        out_center = self.image_enc_center(x_center.view(-1, *x_center.shape[-3:]))
        out_center = out_center.view(*x_center.shape[:2], -1)

        # right view
        out_right = self.image_enc_right(x_right.view(-1, *x_right.shape[-3:]))
        out_right = out_right.view(*x_right.shape[:2], -1)

        # concatenate encoded views
        out_concat = torch.cat([out_left, out_center, out_right], dim=-1)
        _, encoded = _lstm_forward(out_concat, traj_length, self.lstm)

        return encoded


class LanguageEncoder(torch.nn.Module):
    def __init__(self, hidden_dim=64, num_lstm_layers=2) -> None:
        super().__init__()

        self.vocab_size = 264
        self.hidden_dim = hidden_dim
        self.num_lstm_layers = num_lstm_layers

        # embedding layer
        self.embedding = torch.nn.Embedding(self.vocab_size, self.hidden_dim)
        self.lstm = torch.nn.LSTM(
            self.hidden_dim, self.hidden_dim, batch_first=True, num_layers=self.num_lstm_layers
        )

    def forward(self, x, x_length):
        emb = self.embedding(x)
        _, encoded = _lstm_forward(emb, x_length, self.lstm)

        return encoded


class Encoder(torch.nn.Module):
    def __init__(self, img_hidden_dim=64, lang_hidden_dim=64, output_dim=64, img_lstm_layers=2, lang_lstm_layers=2) -> None:
        super().__init__()

        self.img_hidden_dim = img_hidden_dim
        self.lang_hidden_dim = lang_hidden_dim
        self.output_dim = output_dim
        self.img_lstm_layers = img_lstm_layers
        self.lang_lstm_layers = lang_lstm_layers

        # encoders
        self.img_encoder = ImageEncoder(hidden_dim=self.img_hidden_dim, num_lstm_layers=self.img_lstm_layers)
        self.lang_encoder = LanguageEncoder(hidden_dim=self.lang_hidden_dim, num_lstm_layers=self.lang_lstm_layers)

        # linear layers
        self.linear1 = torch.nn.Linear(self.img_hidden_dim+self.lang_hidden_dim, self.output_dim)
        self.linear2 = torch.nn.Linear(self.output_dim, 1)

    def forward(self, img_left, img_center, img_right, lang, traj_length, lang_length):
        img_encoded = self.img_encoder(img_left, img_center, img_right, traj_length)
        lang_encoded = self.lang_encoder(lang, lang_length)

        encoded_cat = torch.cat([img_encoded, lang_encoded], dim=-1)
        out = F.relu(self.linear1(encoded_cat))
        out = self.linear2(out)

        return out, lang_encoded

m = ImageEncoder()
# batch_size, sequence_length, channels, height, width
x = torch.rand(1, 1, 3, 50, 50)
print(m(x, x, x, 1).shape)
