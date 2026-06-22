from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import torch
from collections import Counter

class AGNewsDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=128):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def tokenize(self, text):
        tokens = text.lower().split()
        ids = [self.vocab.get(t, self.vocab["<unk>"]) for t in tokens]
        if len(ids) < self.max_len:
            ids += [self.vocab["<pad>"]] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]
        return ids

    def __getitem__(self, idx):
        ids = self.tokenize(self.texts[idx])
        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


def build_vocab(texts, max_vocab=20000):
    counter = Counter()
    for text in texts:
        counter.update(text.lower().split())
    vocab = {"<pad>": 0, "<unk>": 1}
    for word, _ in counter.most_common(max_vocab - 2):
        vocab[word] = len(vocab)
    return vocab


def load_agnews(batch_size=64, max_len=128):
    dataset = load_dataset("ag_news")
    train_texts = dataset["train"]["text"]
    train_labels = dataset["train"]["label"]
    test_texts  = dataset["test"]["text"]
    test_labels  = dataset["test"]["label"]
    vocab = build_vocab(train_texts)
    train_loader = DataLoader(AGNewsDataset(train_texts, train_labels, vocab, max_len),
                              batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(AGNewsDataset(test_texts,  test_labels,  vocab, max_len),
                              batch_size=batch_size, shuffle=False)
    print(f"Train: {len(train_texts)} | Test: {len(test_texts)} | Vocab: {len(vocab)}")
    return train_loader, test_loader, vocab