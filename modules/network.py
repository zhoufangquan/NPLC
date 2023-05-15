import torch
import torch.nn as nn
from torch.nn.functional import normalize

from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer, AutoConfig


BERT_CLASS = {
    "AgNews": "./pre_trained/STC/AgNews/",  # path of pre-trained model on AgNews
    "StackOverflow": "./pre_trained/STC/StackOverflow",
    "SearchSnippets": "./pre_trained/STC/SearchSnippets",
    "Biomedical": "./pre_trained/STC/Biomedical",
    "GooglenewsTS": "./pre_trained/STC/GooglenewsTS",
    "GooglenewsS": "./pre_trained/STC/GooglenewsS",
    "GooglenewsT": "./pre_trained/STC/GooglenewsT",
    "Tweet": "./pre_trained/STC/Tweet"
}

def get_bert(bert_name):
    """_summary_

    Args:
        bert_name (_type_): _description_

    Returns:
        _type_: _description_
    """

    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/distilbert-base-nli-stsb-mean-tokens', use_fast=True)
    model = AutoModel.from_pretrained(BERT_CLASS[bert_name])

    return model, tokenizer


class Network(nn.Module):
    def __init__(self, bert, feature_dim, class_num, P=None, alpha=1.0):
        """_summary_

        Args:
            bert (_type_): _description_
            feature_dim (_type_): _description_
            class_num (_type_): _description_
        """

        super(Network, self).__init__()
        self.bert = bert
        self.hidden_dim = self.bert.config.hidden_size
        self.feature_dim = feature_dim
        self.cluster_num = class_num
        # self.P = P


        self.instance_projector = nn.Sequential(
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.feature_dim),
        )
        # Clustering head
        self.cluster_projector = nn.Sequential(
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.cluster_num),
            nn.Softmax(dim=1)
        )
        nn.init.trunc_normal_(self.cluster_projector[2].weight, std=0.02)  # 正态分布式 截断。  不在指定区间的数据会被重新分配
        nn.init.trunc_normal_(self.cluster_projector[5].weight, std=0.02)

    def get_mean_embeddings(self, input_ids, attention_mask):
        bert_output = self.bert.forward(input_ids=input_ids, attention_mask=attention_mask)
        attention_mask = attention_mask.unsqueeze(-1)
        mean_output = torch.sum(bert_output[0]*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        return mean_output

    def forward(self, x, x_i, x_j):

        h   = self.get_mean_embeddings(x['input_ids'], attention_mask=x['attention_mask'])
        h_i = self.get_mean_embeddings(x_i['input_ids'], attention_mask=x_i['attention_mask'])
        h_j = self.get_mean_embeddings(x_j['input_ids'], attention_mask=x_j['attention_mask'])

        # 将所有的特征 映射 到一个超球体的表面
        z_i = normalize(self.instance_projector(h_i), dim=1)
        z_j = normalize(self.instance_projector(h_j), dim=1)

        c = self.cluster_projector(h)

        return z_i, z_j, c

    def forward_c(self, x):
        h = self.get_mean_embeddings(x['input_ids'], attention_mask=x['attention_mask'])
        c = self.cluster_projector(h)
        return h, c

