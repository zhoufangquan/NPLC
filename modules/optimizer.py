
import torch


def get_optimizer(model, bert_lr, head_lr, weight_decay=0):

    optimizer = torch.optim.Adam(
        [
            {'params': model.bert.parameters()},
            {'params': model.MLP.parameters(), 'lr': head_lr}
        ],
        weight_decay=weight_decay,
        lr=bert_lr
    )

    return optimizer


def get_optimizer_old(model, bert_lr, head_lr, weight_decay=0):

    optimizer = torch.optim.Adam(
        [
            {'params': model.bert.parameters()},
            {'params': model.instance_projector.parameters(), 'lr': head_lr},
            {'params': model.cluster_projector.parameters(), 'lr': head_lr}
        ],
        weight_decay=weight_decay,
        lr=bert_lr
    )

    return optimizer
