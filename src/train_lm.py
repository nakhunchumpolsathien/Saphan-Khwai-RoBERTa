import warnings
warnings.filterwarnings("ignore")

import os
import torch
from glob import glob
from transformers import LineByLineTextDataset
from tokenizers import ByteLevelBPETokenizer
from transformers import RobertaTokenizer
from transformers import RobertaConfig
from transformers import RobertaForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
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
    for dir in dirs[:5]:
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
    tokenizer.train(files=paths, vocab_size=30_522, min_frequency=2,
                    special_tokens=['<s>', '<pad>', '</s>', '<unk>', '<mask>'])
    os.mkdir('saphankwai')
    tokenizer.save_model('saphankwai')


if __name__ == '__main__':
    config = RobertaConfig(
        vocab_size=25_000,
        hidden_size=768,
        max_position_embeddings=514,
        num_attention_heads=12,
        num_hidden_layers=12,
        type_vocab_size=1)

    tokenizer = RobertaTokenizer.from_pretrained("/root/saphankwai-tokenizer", max_len=512)
    model = RobertaForMaskedLM(config=config)
    print(f'Number of parameter {model.num_parameters()}')

    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path="/root/training_dataset.txt",
        block_size=128)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir="saphankwai-base",
        overwrite_output_dir=True,
        num_train_epochs=30,
        per_device_train_batch_size=128,
        save_steps=10000,
        save_total_limit=2,
        prediction_loss_only=True,
        fp16=True,
        load_best_model_at_end=False,
        logging_dir='/root/saphankwai-base/logs'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    trainer.train()
