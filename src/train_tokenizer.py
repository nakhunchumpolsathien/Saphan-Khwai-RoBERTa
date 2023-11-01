import os
import torch
from glob import glob
from transformers import AdamW
from pprint import pprint
from tokenizers import ByteLevelBPETokenizer
from transformers import RobertaTokenizer
from transformers import RobertaConfig
from transformers import RobertaForMaskedLM
from tqdm import tqdm


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        # store encodings internally
        self.encodings = encodings

    def __len__(self):
        # return the number of samples
        return self.encodings['input_ids'].shape[0]

    def __getitem__(self, i):
        # return dictionary of input_ids, attention_mask, and labels for index i
        return {key: tensor[i] for key, tensor in self.encodings.items()}


def get_texts():
    files = []
    dirs = glob("data/processed_texts/*/", recursive=True)
    for dir in dirs:
        texts = glob(os.path.join(dir, '*.txt'), recursive=True)
        for text in texts:
            files.append(text)
    return files


def mlm(tensor):
    rand = torch.rand(tensor.shape)
    mask_arr = (rand < 0.15) * (tensor > 2)
    for i in range(tensor.shape[0]):
        selection = torch.flatten(mask_arr[i].nonzero()).tolist()
        tensor[i, selection] = 4
    return tensor


def train_tokenizer():
    paths = get_texts()
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=paths, vocab_size=25000, min_frequency=2,
                    special_tokens=['<s>', '<pad>', '</s>', '<unk>', '<mask>'])
    os.mkdir('saphankwai-base')
    tokenizer.save_model('saphankwai-base')

if __name__ == '__main__':
    train_tokenizer()