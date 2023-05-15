# NPLC: Non-Outlier Pseudo-Labeling for Short Text Clustering

This repository is an implementation of "Non-Outlier Pseudo-Labeling for Short Text Clustering". 

## Install requirements
~~~
conda install --yes --file requirements.txt
~~~

## Data

We release the data of stackoverflow now. The pretrained SentenceBERT based on stackoverflow is in pre_trained/STC/StackOverflow.

download the other original datastes from https://github.com/rashadulrakib/short-text-clustering-enhancement/tree/master/data

## pretrain

you can refer to [link](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py) for the pretraining code.

## Run an example


~~~
./run_script/train.sh
~~~

