import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import numpy as np
import os
import pandas as pd
from transformers import BertTokenizer
from stanza.server import CoreNLPClient

import logging

current_path = os.path.dirname(__file__)
DATA_DIR = current_path + "/../data/qqp"
TRAIN_DATA = DATA_DIR + "/ner_qqp_train.csv"
TEST_DATA = DATA_DIR + "/ner_qqp_dev.csv"

MAX_LEN = 64
PAD_ID = 0
SEP_ID = 102

class PawsDataset(Dataset):

    def __init__(self, datafile):
        self.datafile = datafile
        self.df = pd.read_csv(self.datafile, sep=",")
        self.len = self.df.shape[0]
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        logging.info("数据集大小: %d", self.len)


    def __len__(self):
        return self.len

    def __getitem__(self, i):
        row = self.df.iloc[i % self.len]
        ids, seg_ids = self.get_token_ids(row)
        return ids, seg_ids, torch.tensor(row.label), str(row.sentence1)+"\n"+str(row.sentence2)

    def pad_slice(self, tokens, length):
        if len(tokens) <= length:
            return tokens
        else:
            return tokens[:length]

    def get_seg_ids(self, ids):
        seg_ids = []
        tag = 0
        for x in ids:
            seg_ids += [tag]
            if x == SEP_ID:
                tag += 1
        return seg_ids 

    def get_token_ids(self, row):
        # 拼接 句子token + 结尾标志
        #tokens_1 = self.tokenizer.tokenize(row.question1)
        #tokens_2 = self.tokenizer.tokenize(row.question2)
        #ids_1 = self.tokenizer.convert_tokens_to_ids(tokens_1) 
        #ids_2 = self.tokenizer.convert_tokens_to_ids(tokens_2) 
        #ids_pair = self.tokenizer.build_inputs_with_special_tokens(ids_1, ids_2)
        if row.sentence1 is None or row.sentence2 is None:
            return None, None
        if type(row.sentence1) != type("a") or type(row.sentence2) != type('a'):
            return None, None
        if len(row.sentence1) == 0 or len(row.sentence2) == 0:
            return None, None
        s1 = add_ner(row.sentence1)
        s2 = add_ner(row.sentence2)
        ids_pair = self.tokenizer.encode(s1, s2, max_length=MAX_LEN, add_special_tokens=True)
        seg_ids  = self.get_seg_ids(ids_pair)
        #padding
        if len(ids_pair) < MAX_LEN:
            ids_pair += [PAD_ID] * (MAX_LEN- len(ids_pair))
        #totensor
        ids_pair = torch.tensor(ids_pair)[:MAX_LEN]
        #ids_pair = ids_pair.unsqueeze(0)

        if len(seg_ids) < MAX_LEN:
            seg_ids += [PAD_ID] * (MAX_LEN - len(seg_ids))
        seg_ids = torch.tensor(seg_ids)[:MAX_LEN]
        #seg_ids = seg_ids.unsqueeze(0)
        return ids_pair, seg_ids


def add_ner(sentence):
    L = sentence.split('\t')
    L[0] = L[0].split()
    L[1] = L[1].split()
    s = ""
    names = ""
    for i in range(len(L[0])):
        if L[0][i] == 'O':
            continue
        else:
            names += " " + L[1][i]
            L[1][i] = L[0][i]
             
    for i in range(len(L[1])):
        s += " " + L[1][i]
    
    return s + "#" +  names


def my_collate_fn(batch):
    # 过滤为None的数据
    batch = list(filter(lambda x:x[0] is not None, batch))
    if len(batch) == 0: return torch.Tensor()
    return default_collate(batch) # 用默认方式拼接过滤后的batch数据


def get_loader(df, batch_size=64, is_train=True):
    ds_df = PawsDataset(df)
    loader = torch.utils.data.DataLoader(ds_df, collate_fn=my_collate_fn, batch_size=batch_size, shuffle=is_train, num_workers=8, drop_last=is_train)
    loader.num = len(ds_df)
    return loader


def get_train_loader():
    return get_loader(TRAIN_DATA)

def get_test_loader():
    return get_loader(TEST_DATA, is_train=False)


if __name__ == '__main__':    
    text = "O O MISC MISC MISC O O O     Why are African - Americans so beautiful ?"
    print(add_ner(text))
