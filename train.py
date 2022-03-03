import pickle
import spacy
import re
import os
import numpy as np
import torch
import time

from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator, vocab
from torchtext.data.utils import get_tokenizer
from collections import Counter
from models.transformer.model import Transformer
from torch.utils.tensorboard import SummaryWriter


def tokenize(sentence, nlp):
    # replace special characters with space
    sentence = re.sub(r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(sentence))
    sentence = re.sub(r"[ ]+", " ", sentence)
    sentence = re.sub(r"\!+", "!", sentence)
    sentence = re.sub(r"\,+", ",", sentence)
    sentence = re.sub(r"\?+", "?", sentence)
    sentence = sentence.lower()

    # tokenize
    return [token.text for token in nlp.tokenizer(sentence) if token.text != " "]


def count(sentences, tokenizer):
    counter = Counter()
    tokenized_sentences = []
    for i, sentence in enumerate(sentences):
        tokens = tokenizer(sentence)

        tokenized_sentences.append(tokens)
        counter.update(tokens)

    return counter, tokenized_sentences


# nlp_en = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])
# nlp_fr = spacy.load("fr_core_news_sm", disable=["tagger", "parser", "ner"])

# tokenize_en = lambda sentence: tokenize(sentence, nlp_en)
# tokenize_fr = lambda sentence: tokenize(sentence, nlp_fr)

# # english sentences
# data_path = os.path.join(os.path.dirname(__file__), "data")
# with open(os.path.join(data_path, "english.txt"), "r") as f:
#     en = f.readlines()

# with open(os.path.join(data_path, "french.txt"), "r") as f:
#     fr = f.readlines()

# counter_en, sentences_en = count(en, tokenize_en)
# counter_fr, sentences_fr = count(fr, tokenize_fr)

# # save pickles
# with open('sentences_en.pkl', 'wb') as f:
#     print('saving en')
#     pickle.dump(sentences_en, f)

# with open('sentences_fr.pkl', 'wb') as f:
#     print('saving fr')
#     pickle.dump(sentences_fr, f)

# # load pickle files
# with open('sentences_en.pkl', 'rb') as f:
#     sentences_en = pickle.load(f)

# with open('sentences_fr.pkl', 'rb') as f:
#     sentences_fr = pickle.load(f)

# # split dataset 80-10-10
# total = len(sentences_en)
# train_len = int(total * 0.8)
# val_len = (total - train_len) // 2
# test_len = total - train_len - val_len

# remaining_indices = list(range(total))
# train_indices = numpy.random.choice(remaining_indices, train_len, replace=False)

# remaining_indices = list(set(remaining_indices) - set(train_indices))
# val_indices = numpy.random.choice(remaining_indices, val_len, replace=False)

# test_indices = list(set(remaining_indices) - set(val_indices))
# print(len(train_indices), len(val_indices), len(test_indices))


class Vocabulary:
    def __init__(self, min_freq=5) -> None:
        self.itos = {0: "<PAD>", 1: "<BOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.min_freq = min_freq

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentences):
        counter = Counter()
        idx = 4

        for sentence in sentences:
            counter.update(sentence)

        for key, val in dict(counter).items():
            if key in self.stoi.keys():
                continue

            if val >= self.min_freq:
                self.itos[idx] = key
                self.stoi[key] = idx
                idx += 1


class TranslationDataset(Dataset):
    def __init__(self, root_dir_src, root_dir_trgt, min_freq=5) -> None:
        super().__init__()

        self.root_dir_src = root_dir_src
        self.root_dir_trgt = root_dir_trgt
        self.min_freq = min_freq

        # read tokenized sentences from a .pkl file
        with open(self.root_dir_src, "rb") as f:
            self.sentences_src = pickle.load(f)
        
        with open(self.root_dir_trgt, "rb") as f:
            self.sentences_trgt = pickle.load(f)

        # initialize and build a vocabulary
        self.vocab_src = Vocabulary()
        self.vocab_src.build_vocabulary(self.sentences_src)

        self.vocab_trgt = Vocabulary()
        self.vocab_trgt.build_vocabulary(self.sentences_trgt)

    def __len__(self):
        return len(self.sentences_src)

    def __getitem__(self, index):
        src = self.sentences_src[index]
        trgt = self.sentences_trgt[index]

        stoi_src = [self.vocab_src.stoi['<BOS>'], *self._stoi(src, self.vocab_src), self.vocab_src.stoi['<EOS>']]
        stoi_trgt = [self.vocab_trgt.stoi['<BOS>'], *self._stoi(trgt, self.vocab_trgt), self.vocab_trgt.stoi['<EOS>']]

        return torch.tensor(stoi_src), torch.tensor(stoi_trgt)

    def _stoi(self, sentence, vocabulary):
        return [
            vocabulary.stoi[token] if token in vocabulary.stoi else vocabulary.stoi['<UNK>']
            for token in sentence
        ]

def collate_fn(batch, padding_value):
    sources = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # pad sequences, trying to match the longest sequence in the batch
    sources = pad_sequence(sources, padding_value=padding_value)
    targets = pad_sequence(targets, padding_value=padding_value)

    return sources, targets


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        

dataset = TranslationDataset(root_dir_src='sentences_en.pkl', root_dir_trgt='sentences_fr.pkl')

padding_value = dataset.vocab_src.stoi['<PAD>']
collate_batch = lambda batch: collate_fn(batch, padding_value=padding_value)

dataset_indices = torch.arange(len(dataset))

# shuffle the indices and split
np.random.shuffle(dataset_indices)
split_idx = int(np.floor(0.2 * len(dataset)))
train_indices, test_indices = dataset_indices[split_idx:], dataset_indices[:split_idx]

# create random samplers
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = DataLoader(
    dataset=dataset,
    sampler=train_sampler,
    batch_size=128,
    shuffle=False,
    collate_fn=collate_batch
)

test_loader = DataLoader(
    dataset=dataset,
    sampler=test_sampler,
    batch_size=32,
    shuffle=False,
    collate_fn=collate_batch
)


# setup tensorboard
writer = SummaryWriter('runs/loss_plot')

# create the model
model = Transformer(
    src_vocab_size=len(dataset.vocab_src),
    trgt_vocab_size=len(dataset.vocab_trgt),
    device=device
).to(device)

num_epochs = 5

criterion = torch.nn.CrossEntropyLoss(ignore_index=dataset.vocab_src.stoi['<PAD>'])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    print(f'Epoch {epoch+1} / {num_epochs}')

    for batch_idx, batch in enumerate(train_loader):
        print(f'Processing {batch_idx+1} / {len(train_loader)}')
        source = batch[0].transpose(0, 1).to(device)
        target = batch[1].transpose(0, 1).to(device)

        # forward prop
        output = model(target[:, :-1], source)
        output = output.reshape(-1, output.shape[2])
        target = target[:, 1:].reshape(-1)

        start = time.time()
        optimizer.zero_grad()

        loss = criterion(output, target)
        loss.backward()

        optimizer.step()
        end = time.time()

        writer.add_scalar('Training Loss', loss, global_step=batch_idx*(epoch+1))
        print(f'Loss: {loss} -- took {end-start} sec(s)')