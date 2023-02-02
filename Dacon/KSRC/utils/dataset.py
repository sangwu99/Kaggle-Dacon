from sklearn.model_selection import train_test_split
from tokenizers import Tokenizer
import torch


def tokenize(train):
    train_dataset, eval_dataset = train_test_split(train, test_size=0.2, shuffle=True, stratify=train['label'])
    
    tokenized_train = tokenizer(
        list(train_dataset['premise']),
        list(train_dataset['hypothesis']),
        return_tensors="pt",
        max_length=300, # Max_Length = 190
        padding=True,
        truncation=True,
        add_special_tokens=True
        )

    tokenized_eval = tokenizer(
        list(eval_dataset['premise']),
        list(eval_dataset['hypothesis']),
        return_tensors="pt",
        max_length=300,
        padding=True,
        truncation=True,
        add_special_tokens=True
        )
    
    return tokenized_train, tokenized_eval


class BERTDataset(torch.utils.data.Dataset):
    def __init__(self, pair_dataset, label):
        self.pair_dataset = pair_dataset
        self.label = label

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
        item['label'] = torch.tensor(self.label[idx])
        
        return item

    def __len__(self):
        return len(self.label)
    
def label_to_num(label):
    label_dict = {"entailment": 0, "contradiction": 1, "neutral": 2, "answer": 3}
    num_label = []

    for v in label:
        num_label.append(label_dict[v])

    return num_label