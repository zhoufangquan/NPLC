import os
import time
import json
import torch

# 自定义的函数包
from utils import init_args
from utils.utils import init_logger, seed_everything, logger
from utils.pseudo_labels_AL import *
from dataloader import get_train_dataloader, get_eval_dataloader
from modules.network import get_bert, Network
from modules.optimizer import get_optimizer, get_optimizer_old
from TextCluTrainer import TextCluTrainer

from sklearn.cluster import KMeans


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def setup_logpath(args):
    resPath = args.log_dir + '/'

    resPath += "{}".format(time.strftime("%Y-%m-%d %Hh%Mmin", time.localtime()))
    resPath += f'.{args.bert_name}'
    resPath += f'.lr{args.bert_lr}'
    resPath += f'.head_lr{args.head_lr}'
    resPath += f'.bz{args.batch_size}'
    resPath += f'.seed{args.seed}'
    resPath += ".log"

    logger.info(f'log path: {resPath}')
    return resPath


def run(args, device):

    # 加载bert
    bert, tokenize = get_bert(args.bert_name)

    # 加载数据集
    train_loader = get_train_dataloader(args, tokenize)
    eval_loader = get_eval_dataloader(args, tokenize)

    # 构建模型
    device_ids = [0, 1, 2, 3] # 可用GPU
    model = Network(bert, args.feature_dim, args.class_num)
    # model = torch.nn.DataParallel(model, device_ids=device_ids)
    model = model.to(device)
    if args.use_noise:
        model.to('cpu')
        for name, para in model.bert.named_parameters():
            model.bert.state_dict()[name][:] += (torch.rand(para.size()) - 0.5)*args.noise_lambda*torch.std(para)
        model.to(device)
    if args.resume:
        model_fp = os.path.join(args.check_point_path, "checkpoint_last_train.tar")
        checkpoint = torch.load(model_fp)
        model.load_state_dict(checkpoint['net'])
        args.start_epoch = checkpoint['epoch']


    pseudo_labels = get_pseudo_l(bert, eval_loader, device, args)
    # 构建优化器
    optimize = get_optimizer_old(model, args.bert_lr, args.head_lr, args.weight_decay)
    # 创建训练器
    trainer = TextCluTrainer(train_loader, eval_loader, model, optimize, pseudo_labels, device, args)
    # 训练
    trainer.train()

    # # 画图
    # trainer.plot_()


if __name__ == '__main__':
    # 获得代码运行的相关参数
    args = init_args.get_args()
    # 使用训练模型使用  CPU   or   GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 设置保存断点的地方
    args.check_point_path = os.path.join(args.check_point_path, args.data_name)
    if not os.path.exists(args.check_point_path):
        os.makedirs(args.check_point_path)

    # 设置保存图片的地方
    args.pic_dir = os.path.join(args.pic_dir, args.data_name)
    if not os.path.exists(args.pic_dir):
        os.makedirs(args.pic_dir)

    # 设置保存日志的地方
    args.log_dir = os.path.join(args.log_dir, args.data_name)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    init_logger(log_file=setup_logpath(args))

    # 固定随机种子
    seed_everything(args.seed)
    # 将参数输出
    logger.info(
        json.dumps(
            vars(args),
            sort_keys=True,
            indent=4,
            separators=(', ', ': '),
            ensure_ascii=False
        )
    )

    run(args, device)

    logger.info(f"{args.data_name}: train Over!!!")
