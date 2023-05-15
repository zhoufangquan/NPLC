import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", default=800,        type=int, help="训练批次的大小")
    parser.add_argument("--epochs",     default=100,        type=int, help="训练的轮次")
    parser.add_argument("--data_name",  default="StackOverflow",   type=str, help="")
    parser.add_argument("--class_num",  default=20,          type=int, help="")
    parser.add_argument("--max_len",    default=32,         type=int, help="文本的最大长度")
    parser.add_argument("--resume",     default=0,          type=int, help="是否接着上次进行训练")
    parser.add_argument("--bert_name",  default='StackOverflow',   type=str, help="pretrained SBERT based on StackOverflow")


    parser.add_argument("--seed", default=2023, type=int, help="随机种子的设置")
    parser.add_argument("--workers", default=4, type=int, help="")
    parser.add_argument("--log_dir", default="./log/", type=str, help="日志的保存地方")
    parser.add_argument("--pic_dir", default="./pic/", type=str, help="图片的保存地方")

    parser.add_argument("--data_dir", default="../datasets/", type=str, help="")

    parser.add_argument("--accumulation_steps", default=1, type=int, help="")
    parser.add_argument("--start_epoch", default=0, type=int, help="")

    parser.add_argument("--feature_dim", default=128, type=int, help="对比学习中，每个数据的维度数")

    parser.add_argument("--use_noise", default=0, type=int, help="是否给预训练模型加噪声")
    parser.add_argument("--noise_lambda", default=0.15, type=float, help="增加噪声时的超参数")

    parser.add_argument("--check_point_path", default="./save/", type=str, help="断点的存储地方")

    parser.add_argument("--bert_lr", default=0.000005, type=float, help="预训练模型的学习率")
    parser.add_argument("--head_lr", default=0.0005, type=float, help="MLP的学习率")
    parser.add_argument("--weight_decay", default=0.0001, type=float, help="")
    parser.add_argument("--instance_temperature", default=0.5, type=float, help="")
    parser.add_argument("--cluster_temperature", default=1.0, type=float, help="")

    parser.add_argument("--eta", default=1.0, type=float, help="")

    args = parser.parse_args()
    return args
