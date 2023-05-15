import sys
import math
import time
import datetime
import numpy as np

import torch
import torch.nn as nn
from torch._six import inf
import torch.nn.functional as F


from utils.utils import logger, save_best_model, save_last_model
from utils.pseudo_labels_AL import *
from modules.loss import InstanceLossBoost, ClusterLossBoost
from metrics.metrics import cluster_metric, get_adjusted_l

from sklearn.cluster import KMeans, AgglomerativeClustering

class TextCluTrainer(nn.Module):

    def __init__(self, train_loader, eval_loader, model, optimize, pseudo_labels, device, args) -> None:
        """_summary_

        Args:
            train_loader (_type_): 训练集
            eval_loader (_type_): 测试集
            model (_type_): 模型
            optimize (_type_): 优化器
            device (_type_): 训练模型使用的设备
            args (_type_): 其他参数
        """

        super(TextCluTrainer, self).__init__()
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.model = model
        self.optimize = optimize
        self.pseudo_labels = torch.LongTensor(pseudo_labels).to(device)
        self.device = device
        self.args = args

        self.criterion_ins = InstanceLossBoost(
            temperature=args.instance_temperature,
            cluster_num=args.class_num,
            device=device
        ).to(device)
        self.criterion_clu = ClusterLossBoost(
            args.class_num, 
            device
        ).to(device)

        # 判断能否使用自动混合精度
        self.enable_amp = True if "cuda" in device.type else False
        # 在训练最开始之前实例化一个GradScaler对象
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.enable_amp)

        self.total_loss = []
        self.total_acc = []
        self.total_nmi = []
        self.total_ari = []

    def train_one_epoch(self, epoch):
        loss_epoch = 0
        self.model.train(True)
        self.optimize.zero_grad()

        print_feq = max(len(self.train_loader) // 5, 1)

        for step, (antor, x_i, x_j, _, index) in enumerate(self.train_loader):
            for x in x_i.keys():
                antor[x] = antor[x].to(self.device)
                x_i[x] = x_i[x].to(self.device)
                x_j[x] = x_j[x].to(self.device)

            self.model.train(True)
            with torch.cuda.amp.autocast(enabled=self.enable_amp):
                z_i, z_j, c = self.model(antor, x_i, x_j)
                loss_ins = self.criterion_ins(z_i, z_j, self.pseudo_labels[index].to(self.device)) / self.args.accumulation_steps
                loss_clu = self.criterion_clu(c, self.pseudo_labels.to(self.device), index) / self.args.accumulation_steps
                # loss = self.args.eta*loss_ins + loss_clu
                loss = (10+5*np.cos(self.gstep/self.all_iter*np.pi)) * loss_ins + loss_clu
                loss = loss

            loss_ins_value = loss_ins.item()
            loss_clu_value = loss_clu.item()

            if not math.isfinite(loss_ins_value) and not math.isfinite(loss_clu_value):
                print(f"Loss is {loss_ins_value}, {loss_clu_value}, stopping training")
                sys.exit(1)

            if (step+1) % print_feq == 0:
                logger.info(f"Epoch:{epoch:3}| Step [{step+1:3}/{len(self.train_loader)}] |  loss_instance: {loss_ins_value:2.5} loss_cluster: {loss_clu_value:2.5}")
            # 1、Scales loss.  先将梯度放大 防止梯度消失
            self.scaler.scale(loss).backward()

            if (step+1) % self.args.accumulation_steps == 0:

                self.scaler.unscale_(self.optimize)

                # 2、scaler.step()   再把梯度的值unscale回来.
                # 如果梯度的值不是 infs 或者 NaNs, 那么调用optimizer.step()来更新权重,
                # 否则，忽略step调用，从而保证权重不更新（不被破坏）
                self.scaler.step(self.optimize)

                # 3、准备着，看是否要增大scaler
                self.scaler.update()

                self.optimize.zero_grad()

            loss_epoch += loss.item()
            self.gstep += 1

        return loss_epoch

    def train(self, ):
        # train
        logger.info(self.optimize)
        logger.info(f"{self.args.data_name}: Start training for {self.args.epochs} epochs")
        start_time = self.eval(0)['ACC']
        max_accuracy = 0.0
        self.gstep = 0
        self.all_iter = self.args.epochs * len(self.train_loader)
        for epoch in range(self.args.start_epoch, self.args.epochs):

            loss_epoch = self.train_one_epoch(epoch+1)

            res = self.eval(epoch+1)
            if res['ACC'] > max_accuracy:
                max_accuracy = res['ACC']
                save_best_model(self.args.check_point_path, self.model, True)

            logger.info(
                f"Epoch [{(epoch+1):3}/{self.args.epochs:3}]| Loss: {loss_epoch / len(self.train_loader):2.5}| NMI: {res['NMI']:2.4}| ARI: {res['ARI']:2.4}| ACC: {res['ACC']:2.3%}| MAX_ACC: {max_accuracy:2.3%}")

            save_last_model(self.args.check_point_path, self.model, epoch, True)
            self.total_loss.append(loss_epoch)
            self.total_acc.append(res['ACC'])
            self.total_nmi.append(res['NMI'])
            self.total_ari.append(res['ARI'])

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info("Training time {}!".format(total_time_str))

    @torch.no_grad()
    def eval(self, epoch):
        self.model.eval()
        embeddings = []
        pred_vector = []
        labels_vector = []
        for step, (texts, labels) in enumerate(self.eval_loader):

            for x in texts.keys():
                texts[x] = texts[x].to(self.device)

            # compute output
            with torch.cuda.amp.autocast(enabled=self.enable_amp):
                h, prod = self.model.forward_c(texts)
                preds = torch.argmax(prod, dim=1)

            pred_vector.extend(preds.cpu().detach().numpy())
            labels_vector.extend(labels.numpy())
            embeddings.append(h)
            
        embeddings = torch.cat(embeddings, dim=0)
        pred_vector = np.array(pred_vector)
        labels_vector = np.array(labels_vector)

        nmi, ari, f, acc, _ = cluster_metric(labels_vector, pred_vector)
        logger.info((nmi, ari, f, acc))

        if epoch % 4 == 0 and self.args.epochs > epoch:  # update pseudo labels
            pseudo_labels_new = get_pseudo_l(self.model.bert, self.eval_loader, self.device, self.args, embeddings)  # 两次为标签的对应有待修改
            adjust_labels = get_adjusted_l(self.pseudo_labels.cpu().detach().numpy()+1, pseudo_labels_new+1) - 1
            self.pseudo_labels = torch.LongTensor(adjust_labels).to(self.device)

        if self.args.class_num > 20:
            model = AgglomerativeClustering(
                n_clusters=self.args.class_num,
                affinity="cosine",
                linkage="average",
                memory=None,
                connectivity=None,
                compute_full_tree="auto",
                distance_threshold=None,
                compute_distances=False
            )
            model.fit(embeddings.cpu().detach().numpy())
            nmi, ari, f, acc, _ = cluster_metric(labels_vector, model.labels_)
        else:
            nmi, ari, f, acc, _ = cluster_metric(labels_vector, pred_vector)

        logger.info((nmi, ari, f, acc))
        return {
            'ACC': acc,
            'NMI': nmi,
            'ARI': ari
        }
