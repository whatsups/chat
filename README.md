# chat
**这是一个基于seq2seq模型的聊天系统，使用LSTM/GRU+注意力机制。使用开源框架pytorch。**
项目过程中的学习记录见[record.pdf](record.pdf)。

## 环境
- python 3.6
- pytorch 0.4
- 其他python库


## 语料
本项目语料使用了20万句电影对话语料，去除低频词汇、过长句子后剩余约14万句，构建词典大小13000词。
训练此模型，首先在```main_train.py```中指定你的语料路径。并且，语料格式须满足：每个句子占一行，每两行为一个对话语句对。如：
```
Nice to meet you.
Nice to meet you, too.

I am sorry.
You are welcome.
```
一些公开语料：[[语料]](https://github.com/rkadlec/ubuntu-ranking-dataset-creator)

## 模型
基于sequence to sequence模型，项目分别实现了LSTM和GRU的模型构建，并实现了注意力机制。通过```main_train.py```指定相关参数选择使用LSTM或GRU，以及是否使用注意力机制。

## 训练
执行```python main_train.py```以训练模型。相关参数解释如下：
- corpus：语料路径
- batch_size
- n_iteration：迭代次数
- learning_rate：学习率
- n_layers：层数
- hidden_size:隐藏层维度
- print_every：每多少次打印损失
- save_every：每多少次保存模型
- load_pretrain:与训练模型路径
- voc：词典
- pairs:训练语句对
- bidirectional：是否双向
- dropout：dropout失活概率
- use_ATTN：是否使用注意力机制
- rnn_type：单元类型，'LSTM'或'GRU'


## 测试
```evaluate.py```执行测试。
