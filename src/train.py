import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import logging
import random
import math
import os
from tqdm import tqdm
from model import EncoderRNN,DecoderRNN
from config import SOS_token,EOS_token,PAD_token,MAXLEN,teacher_forcing_ratio,save_dir

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

# 一轮迭代
#input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding,
#         encoder_optimizer, decoder_optimizer, batch_size, max_length=MAX_LENGTH
def train_iteration(input_batch_tensor,input_len,target_batch_tensor,max_target_len,mask,encoder,decoder,embedding,
	encoder_optimizer,decoder_optimizer,batch_size,rnn_type,use_ATTN,max_length=MAXLEN):
	
	# 梯度清零
	encoder_optimizer.zero_grad()
	decoder_optimizer.zero_grad()

	input_batch_tensor = input_batch_tensor.to(device)
	target_batch_tensor = target_batch_tensor.to(device)
	mask = mask.to(device)

	loss = 0
	print_losses = []
	n_totals = 0

	if rnn_type == 'LSTM':
		encoder_outputs,(encoder_h,encoder_c) = encoder(input_batch_tensor,input_len,None)
		# 初始输入
		decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
		decoder_input = decoder_input.to(device)
		decoder_hidden = (encoder_h[:decoder.n_layers],encoder_c[:decoder.n_layers])
		#decoder(encoder_outputs,decoder_hidden)
		use_teach_forcing = True if random.random() < teacher_forcing_ratio else False
		# 强制
		if use_teach_forcing:
			for t in range(max_target_len):
				if use_ATTN:
					decoder_output,decoder_hidden,_ = decoder(encoder_outputs,decoder_input,decoder_hidden)
				else:
					decoder_output,decoder_hidden = decoder(encoder_outputs,decoder_input,decoder_hidden)
				decoder_input = target_batch_tensor[t].view(1,-1)#相当于添加0维度
				#print(decoder_output.size())
				#print(target_batch_tensor[t])
				#print(target_batch_tensor[t].size())
				loss += F.cross_entropy(decoder_output,target_batch_tensor[t], ignore_index=EOS_token)					
		# 不强制
		else:
			for t in range(max_target_len):
				if use_ATTN:					
					decoder_output,decoder_hidden,_ = decoder(encoder_outputs,decoder_input,decoder_hidden)
				else:
					decoder_output,decoder_hidden = decoder(encoder_outputs,decoder_input,decoder_hidden)					
				_,topi = decoder_output.topk(1)					
				decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
				decoder_input = decoder_input.to(device)
				loss += F.cross_entropy(decoder_output, target_batch_tensor[t], ignore_index=EOS_token)

		loss.backward()

		# 梯度裁剪
		gradient_clip = 50.0
		_ = torch.nn.utils.clip_grad_norm_(encoder.parameters(),gradient_clip)
		_ = torch.nn.utils.clip_grad_norm_(decoder.parameters(),gradient_clip)

		encoder_optimizer.step()
		decoder_optimizer.step()

		return loss.item()/max_target_len

	# 使用GRU
	else:
		encoder_outputs,encoder_h = encoder(input_batch_tensor,input_len,None)
		# 初始输入
		decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
		decoder_input.to(device)
		decoder_hidden = encoder_h[:batch_size]
		#decoder(encoder_outputs,decoder_hidden)
		use_teach_forcing = True if random.random() < teacher_forcing_ratio else False
		# 强制
		if use_teach_forcing:
			for t in range(max_target_len):
				if use_ATTN:
					decoder_output,decoder_hidden,_ = decoder(encoder_outputs,decoder_input,decoder_hidden)
				else:
					decoder_output,decoder_hidden = decoder(encoder_outputs,decoder_input,decoder_hidden)
				decoder_input = target_batch_tensor[t].view(1,-1)#相当于添加0维度
				loss += F.cross_entropy(decoder_output,target_batch_tensor[t], ignore_index=EOS_token)					
		# 不强制
		else:
			for t in range(max_target_len):
				if use_ATTN:					
					decoder_output,decoder_hidden,_ = decoder(encoder_outputs,decoder_input,decoder_hidden)
				else:
					decoder_output,decoder_hidden = decoder(encoder_outputs,decoder_input,decoder_hidden)					
				_,topi = decoder_output.topk(1)					
				decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
				decoder_input = decoder_input.to(device)
				loss += F.cross_entropy(decoder_output, target_batch_tensor[t], ignore_index=EOS_token)

		loss.backward()

		# 梯度裁剪
		gradient_clip = 50.0
		_ = torch.nn.utils.clip_grad_norm_(encoder.parameters(),gradient_clip)
		_ = torch.nn.utils.clip_grad_norm_(decoder.parameters(),gradient_clip)

		encoder_optimizer.step()
		decoder_optimizer.step()

		return loss.item()/max_target_len


	



# 训练脚本
def train(load_pretrain,voc,pair_batches,n_iteration,learning_rate,batch_size,n_layers,hidden_size,print_every,
			save_every,dropout,rnn_type='LSTM',bidirectional=False,use_ATTN=False,decoder_learning_ratio=5.0):
	"""
		参数：
		voc:词典对象
		pair_batches：准备的训练集，每个batch中是： input_batch_tensor,input_len,output_batch_tensor,max_target_len,mask

	"""
	checkpoint = None
	# 构建模型
	embedding = nn.Embedding(voc.n_words,hidden_size)
	# self,embedding,input_size,hidden_size,n_layers=1,bidirectional=False,dropout=0,rnn_type='LSTM'
	print("INFO:Building Encoder and Decoder ... ")
	encoder = EncoderRNN(embedding,hidden_size,hidden_size,n_layers,bidirectional=bidirectional,dropout=dropout,rnn_type=rnn_type)
	# 构建DeocderRNN
	# self,embedding,hidden_size,output_size,n_layers=1,rnn_type='LSTM',use_ATTN=False,dropout = 0.1
	decoder = DecoderRNN(embedding,hidden_size,voc.n_words,n_layers,rnn_type=rnn_type,use_ATTN=use_ATTN,dropout=dropout)
	
	#加载预训练
	if load_pretrain!=None:
		checkpoint = torch.load(load_pretrain)
		encoder.load_state_dict(checkpoint['en'])
		decoder.load_state_dict(checkpoint['de'])


	print(encoder)
	print(decoder)
	# 放到GPU上
	encoder = encoder.to(device)
	decoder = decoder.to(device)

	print("INFO:Building optimizers ...")
	encoder_optimizer = optim.Adam(encoder.parameters(),lr=learning_rate)
	decoder_optimizer = optim.Adam(decoder.parameters(),lr=learning_rate*decoder_learning_ratio)

	# 加载预训练
	if load_pretrain!=None:
		encoder_optimizer.load_state_dict(checkpoint['en_opt'])
		decoder_optimizer.load_state_dict(checkpoint['de_opt'])


	print("INFO:Initializing...")

	start_iteration = 1
	perplexity = []
	print_loss = 0

	logger = logging.getLogger()
	logger.setLevel(level = logging.INFO)
	handler = logging.FileHandler("log.txt")
	handler.setLevel(logging.INFO)
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	handler.setFormatter(formatter)
	logger.addHandler(handler)

	if load_pretrain!=None:
		start_iteration = checkpoint['iteration'] + 1
		perplexity = checkpoint['plt']

	# 迭代
	for iteration in tqdm(range(start_iteration,n_iteration+1)):
		training_batch = pair_batches[iteration-1]
		input_batch_tensor,input_len,output_batch_tensor,max_target_len,mask = training_batch

		#input_batch_tensor = input_batch_tensor.to(device)
		#output_batch_tensor = output_batch_tensor.to(device)
		#mask = mask.to(device)

		#voc,pair_batches,n_iteration,learning_rate,batch_size,n_layers,hidden_size,print_every,
		#	save_every,dropout,rnn_type='LSTM',bidirectional=False,use_ATTN=False,decoder_learning_ratio=5.0)

		#input_batch_tensor,input_len.target_batch_tensor,max_target_len,mask,encoder,decoder,embedding,
			#encoder_optimizer,decoder_optimizer,batch_size,rnn_type,use_ATTN,max_length=MAXLEN
		#
		loss = train_iteration(input_batch_tensor,input_len,output_batch_tensor,max_target_len,mask,encoder,
			decoder,embedding,encoder_optimizer,decoder_optimizer,batch_size,rnn_type=rnn_type,use_ATTN=use_ATTN)

		print_loss += loss
		perplexity.append(loss)

		if iteration % print_every == 0:
			print_loss_average = math.exp(print_loss/print_every)
			print('INFO:loss %d %d%% %.4f' % (iteration, iteration / n_iteration * 100, print_loss_average))
			logger.info('INFO:loss %d %d%% %.4f' % (iteration, iteration / n_iteration * 100, print_loss_average))
			print_loss = 0.0

		# 保存模型
		if (iteration % save_every == 0):
			directory = os.path.join(save_dir,'model_param','{}_{}_{}'.format(n_layers,n_layers,hidden_size))
			if not os.path.exists(directory):
				os.makedirs(directory)
			torch.save(
				{
					'iteration':iteration,
					'encoder':encoder.state_dict(),
					'decoder':decoder.state_dict(),
					'encoder_optim':encoder_optimizer.state_dict(),
					'decoder_optim':decoder_optimizer.state_dict(),
					'loss':loss,
					'plt':perplexity
				},os.path.join(directory,'{}_{}.tar'.format(n_iteration,'seq2seq_bidir_model'))
				)
