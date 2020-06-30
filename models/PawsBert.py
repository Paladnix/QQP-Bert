import torch 
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertForPreTraining
import torch.nn.functional as F

import os


current_path = os.path.dirname(__file__)
BERT_MODEL_PATH = current_path + "/../../_model/bert-base/"

class PawsBertModel(nn.Module):

    def __init__(self, n_classes=1):
        super(PawsBertModel, self).__init__()
        self.model_name = 'PawsBertModel'

        #self.bert_model = BertForPreTraining.from_pretrained(BERT_MODEL_PATH, from_tf=True)
        
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')

        self.fcc= nn.Sequential(nn.Linear(768, n_classes))


    def forward(self, ids, seg_ids):
        #last seq 是最后一层的输出
        outputs = self.bert_model(ids)
        out = self.fcc(outputs[0][:,0,:]).sigmoid()
        return out



if __name__ == '__main__':
    model = PawsBertModel()
    print(model)
