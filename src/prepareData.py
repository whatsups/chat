import os
import random
import itertools
import torch
import torch.nn as nn
from config import SOS_token,EOS_token,PAD_token,REVERSE,save_dir

# 生成二进制矩阵Matrix，Matrix[i,j]=0表示第i时间步第j样例位置的单词是填充的
def binaryMatrix(output_batch):
	matrix=[]
	for i,seq in enumerate(output_batch):
		matrix.append([])
		for word in seq:
			if word == PAD_token:
				matrix[i].append(0)
			else:
				matrix[i].append(1)
	return matrix				

# 句子转换成index列表，添加EOS
def sentence2index(voc,sentence):
	return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token];

# PAD填充，按时间步排列
def padding(l,fillValue=PAD_token):
	padded_input = list(itertools.zip_longest(*l,fillvalue=fillValue))
	return padded_input

# 准备输入
def prepareInput(voc,input_batches):

	# 转成index
	input_indexes = [sentence2index(voc,sentence) for sentence in input_batches]
	# 记录长度
	input_len = [len(sentence_indexes) for sentence_indexes in input_indexes]
	# 填充
	padded_input = padding(input_indexes)
	# 转成tensor
	input_batches_tensor = torch.LongTensor(padded_input)
	return input_batches_tensor,input_len

# 准备输出
def prepareOutput(voc,output_batches):

	# 转index
	output_indexes = [sentence2index(voc,sentence) for sentence in output_batches]
	# 记录最大长度
	output_maxlen = max([len(sentence_indexes) for sentence_indexes in output_indexes])
	# 填充
	padded_output = padding(output_indexes)
	# 记录填充位置矩阵
	mask = binaryMatrix(padded_output)
	mask = torch.ByteTensor(mask)
	# 转Tensor
	output_batches_tensor = torch.LongTensor(padded_output)

	return output_batches_tensor,mask,output_maxlen

# 准备一个mini-batch的训练数据
def batch2TrainData(voc,pair_batch):

	# pair_batch按句子长短排序，长在前
	pair_batch.sort(key=lambda x:len(x[0].split(" ")),reverse = True)
	# 输入batch和输出batch
	input_batch = []
	output_batch = []
	# 分离输入输出
	for i in pair_batch:
		input_batch.append(i[0])
		output_batch.append(i[1])

	# 准备输入和输出tensor		
	input_batch_tensor,input_len = prepareInput(voc,input_batch)
	output_batch_tensor,mask,max_target_len = prepareOutput(voc,output_batch)

	return input_batch_tensor,input_len,output_batch_tensor,max_target_len,mask

"""
	加载训练数据
	参数：
		voc：词典，Voc类对象
		pairs：训练语句对
		corpus:语料
		batch_size：mini-batch大小
		n_iterations：迭代次数，迭代多少个mini-batch
	返回：	
		pair_batches：n_iterations个mini-batch，每个mini-batch包括5项：
		input_batch_tensor,input_len,output_batch_tensor,mask,max_target_len
"""
def loadTraingingData(voc,pairs,corpus,batch_size,n_iterations):

	corpus_name = corpus.split('/')[-1].split('.')[0];
	traing_batches = None

	try:
		print("INFO:Start loading all_training_batches...")
		pair_batches = torch.load(os.path.join(save_dir,'training_data',corpus_name,
						'{}_{}_{}.tar'.format(n_iterations,\
							'all_training_batches',batch_size)))
	except FileNotFoundError:

		print("INFO:All_traning_batches have not prepared! Start preparing training_batches...")
	# 准备n_iterations个训练mini-batch数据，每个大小为batch_size
	# 且迭代次数是n_iterations，因而每次迭代是一个mini-batch
		pair_batches = [batch2TrainData(voc,[random.choice(pairs) for _ in range(batch_size)])
							 for _ in range(n_iterations)]
		torch.save(pair_batches,os.path.join(save_dir,'training_data',corpus_name,
					'{}_{}_{}.tar'.format(n_iterations,\
						'all_training_batches',batch_size)))
                                            

	# 返回所有训练数据
	print("INFO:End preparing training_batches!")
	print("\nINFO:End process data!\n")
	return pair_batches

"""
	测试脚本

	from PrepareData import loadTraingingData
	from LoadData import loadPareparedData
	import os

	corpus = os.path.join(os.path.dirname(os.getcwd()),'corpus/conservation.txt')
	voc,pairs = loadPareparedData(corpus)
	loadTraingingData(voc,pairs,corpus,64,2)
"""