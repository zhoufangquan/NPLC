import random
import numpy as np
from collections import Counter


import torch
import torch.nn as nn

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


def detect_outlier(embeddings, init_labels, num_clu):

    outliers_fraction = 0.3  # 离群因子
    P = random.uniform(0.75, 0.95)
    avgclusternum = init_labels.shape[0] // num_clu

    # 将每个簇类的标签放在一个列表中
    clu_idx = [[] for i in range(num_clu)]
    for i, la in enumerate(init_labels):
        clu_idx[la].append(i)
    
    outlier_idx = []  # 用于存放 离群索引数据 的 索引
    for idx in clu_idx:
        clf = IsolationForest()
        # clf = LocalOutlierFactor()
        clf.fit(embeddings[idx, :])
        res = clf.predict(embeddings[idx, :])
        outlier_idx += [idx[i] for i, f in enumerate(res) if f != 1]

#     idx_without_outlierP = [[] for i in range(num_clu)]
#     for i, la in enumerate(init_labels):
#         if i not in outlier_idx: 
#             idx_without_outlierP[la].append(i)

#     for idx in idx_without_outlierP:
#         num_threshold = int(avgclusternum * P)
#         if len(idx) <= num_threshold: continue
#         temp = np.random.choice(idx, len(idx) - num_threshold, replace=False)
#         outlier_idx += temp.tolist()
    
    init_labels[outlier_idx] = -1

    print((init_labels != -1).sum(), init_labels.shape[0])

    return init_labels


@torch.no_grad()
def get_pseudo_l(bert, eval_loader, device, args, embeddings=None):
    if embeddings is None:
        embeddings = []
        bert.to(device)
        bert.eval()
        for step, (text, _) in enumerate(eval_loader):

            for x in text.keys():
                text[x] = text[x].to(device)

            input_ids = text['input_ids']
            attention_mask = text['attention_mask']

            bert_output = bert.forward(input_ids=input_ids, attention_mask=attention_mask)
            attention_mask = attention_mask.unsqueeze(-1)
            mean_output = torch.sum(bert_output[0]*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)

            embeddings.append(mean_output)

        embeddings = torch.cat(embeddings, dim=0)

    num_clu = args.class_num

    if num_clu < 50:
        model = KMeans(
            n_clusters=num_clu, 
            init="k-means++", 
            n_init=10, 
            max_iter=300, 
            tol=1e-4, 
            verbose=0, 
            random_state=None, 
            copy_x=True, 
            algorithm="auto")
    else:
        model = AgglomerativeClustering(
            n_clusters=num_clu,
            affinity="cosine",
            linkage="average",
            memory=None,
            connectivity=None,
            compute_full_tree="auto",
            distance_threshold=None,
            compute_distances=False)

    model.fit(embeddings.cpu().detach().numpy())

    init_labels = model.labels_

    pseudo_labels = detect_outlier(embeddings.cpu().detach().numpy(), init_labels, num_clu)

    return pseudo_labels

