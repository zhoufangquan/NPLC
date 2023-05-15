import os
import re
import pandas as pd


import torch
from torch.utils import data
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer, AutoConfig


class MyDataset(Dataset):
    
    def __init__(self, data_dir, data_name, max_len, tokenizer, is_train) -> None:
        """_summary_

        Args:
            data_dir (_type_): _description_
            data_name (_type_): _description_
            max_len (_type_): _description_
            tokenizer (_type_): _description_
            is_train (bool): _description_
        """

        self.is_train = is_train
        data_path = os.path.join(data_dir, data_name+'.csv')
        df_data = pd.read_csv(data_path)     #.iloc[:100, :]
        text = df_data['text'].fillna('.').tolist()
        if is_train:
            text1 = df_data['text1'].fillna('.').tolist()
            text2 = df_data['text2'].fillna('.').tolist()
        self.label = df_data['label'].astype(int).tolist()  # 数据标签从零开始
    
        tokenized_text = self.get_token(tokenizer, text, max_len)
        df_tmp = pd.DataFrame.from_dict(tokenized_text, orient="index").T
        self.tokenized_text = df_tmp.to_dict(orient="records")
        
        if is_train:
            tokenized_text1 = self.get_token(tokenizer, text1, max_len)
            df_tmp = pd.DataFrame.from_dict(tokenized_text1, orient="index").T
            self.tokenized_text1 = df_tmp.to_dict(orient="records")

            tokenized_text2 = self.get_token(tokenizer, text2, max_len)
            df_tmp = pd.DataFrame.from_dict(tokenized_text2, orient="index").T
            self.tokenized_text2 = df_tmp.to_dict(orient="records")

    def get_token(self, tokenizer, text, max_length):
        token_feat = tokenizer.batch_encode_plus(
            text,
            max_length=max_length,
            # return_tensors='pt',
            padding='max_length',
            truncation=True
        )
        return token_feat

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        if self.is_train:
            return {'text': self.tokenized_text[index],
                    'text1': self.tokenized_text1[index],
                    'text2': self.tokenized_text2[index],
                    'label': self.label[index],
                    'index': index}
        else:
            return {'text': self.tokenized_text[index],
                    'label': self.label[index]}

def collate_fn(batch):
    max_len = 0
    for X in batch:
        max_len = max(
            max_len,
            sum(X['text']['attention_mask']),
            sum(X['text1']['attention_mask']),
            sum(X['text2']['attention_mask']),
        )
        
    texts = dict()
    for name in ['text', 'text1', 'text2']:
        all_input_ids = torch.tensor([x[name]['input_ids'][:max_len] for x in batch])
        # all_token_type_ids = torch.tensor([x[name]['token_type_ids'][:max_len] for x in batch])
        all_attention_mask = torch.tensor([x[name]['attention_mask'][:max_len] for x in batch])
        texts[name] = {
            'input_ids': all_input_ids,
            'attention_mask': all_attention_mask,
            # 'token_type_ids': all_token_type_ids
        }
    
    all_labels = torch.tensor([x['label'] for x in batch])
    index = torch.tensor([x['index'] for x in batch])

    return (
        texts['text'],  # 原始数据
        texts['text1'], # 数据增强 1
        texts['text2'], # 数据增强 2
        all_labels,     # 数据的类别标签
        index
    )

def collate_fn1(batch):
    max_len = 0
    for X in batch:
        max_len = max(
            max_len,
            sum(X['text']['attention_mask']),
        )
        
    texts = dict()
    all_input_ids = torch.tensor([x['text']['input_ids'][:max_len] for x in batch])
    # all_token_type_ids = torch.tensor([x['text']['token_type_ids'][:max_len] for x in batch])
    all_attention_mask = torch.tensor([x['text']['attention_mask'][:max_len] for x in batch])
    texts['text'] = {
        'input_ids': all_input_ids,
        'attention_mask': all_attention_mask,
        # 'token_type_ids': all_token_type_ids
    }
    
    all_labels = torch.tensor([x['label'] for x in batch])

    return (
        texts['text'],  # 原始数据
        all_labels      # 数据的类别标签
    )

def get_train_dataloader(args, tokenizer):
    dataset = MyDataset(args.data_dir, args.data_name, args.max_len, tokenizer, True)
    dataloader =  data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )

    print(f"train loader size: {dataset.__len__()}")

    return dataloader

def get_eval_dataloader(args, tokenizer):
    dataset = MyDataset(args.data_dir, args.data_name, args.max_len, tokenizer, False)
    dataloader =  data.DataLoader(
        dataset,
        batch_size=1000,
        collate_fn=collate_fn1,
        shuffle=False,
        num_workers=args.workers,
    )

    print(f"eval loader size: {dataset.__len__()}")

    return dataloader

