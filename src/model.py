import torch
import torch.nn as nn
import torch.nn.functional as F

# 使用CUDA
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

# Attention机制
class AttentionModel(nn.Module):

	def __init__(self,hidden_size):
		super(AttentionModel,self).__init__()
		self.hidden_size = hidden_size
		self.attn = nn.Linear(self.hidden_size*2,self.hidden_size)
		self.v = nn.Parameter(torch.FloatTensor(1,self.hidden_size))

	# encoder_outputs是Encoder的输出，last_output是Decoder的上一步输出，以此为根据计算权重
	def forward(self,encoder_outputs,rnn_hidden):
		
		# 根据注意力机制，Encoder的seq_len，即encoder_seq_len，就是注意力权重的维度
		# 再加上batch_size
		encoder_seq_len = encoder_outputs.size(0)
		batch_size = rnn_hidden.size(1)

		energy = torch.zeros(batch_size,encoder_seq_len)
		energy = energy.to(device)

		# 计算
		for b in range(batch_size):
			for i in range(encoder_seq_len):
				# 512 + 512 = 1024
				tmp = torch.cat((rnn_hidden[:,b],encoder_outputs[i,b].unsqueeze(0)),1)
				#print('rnn_hidden_Size:',rnn_hidden[:b].size())
				#print('encoder_outputs_size:',encoder_outputs[i,b].size())
				#print('tmp_size:',tmp.size())
				# 转成512
				tmp = self.attn(tmp)
				# 计算内积
				energy[b,i] = self.v.squeeze(0).dot(tmp.squeeze(0))

		# 归一化
		return F.softmax(energy,dim=1).unsqueeze(1) #最后添加一个维度，是为了和encoder_output进行bmm相乘

# EncoderRNN
class EncoderRNN(nn.Module):
	def __init__(self,embedding,input_size,hidden_size,n_layers=1,bidirectional=False,dropout=0,rnn_type='LSTM'):
		super(EncoderRNN, self).__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size

		self.n_layers = n_layers

		self.bidirectional = bidirectional
		self.dropout = dropout
		self.rnn_type = rnn_type

		self.embedding = embedding
		
		# 建立RNN层
		if self.rnn_type == 'LSTM':
			self.lstm = nn.LSTM(input_size,hidden_size,n_layers,bidirectional=self.bidirectional,
				dropout=(0 if n_layers==1 else dropout))
		elif self.rnn_type == 'GRU':
			self.gru = nn.GRU(hidden_size,hidden_size,n_layers,
				dropout=(0 if self.n_layers==1 else dropout),bidirectional=self.bidirectional)
		
	# 输入按时间步排好的词向量
	# 返回:LSTM返回：Tensor([seq_len,batch_size,hidden_size]),
	#					(Tensor[n_layers,batch_size,hidden_size],Tensor[n_layers,batch_size,hidden_size])
	# GRU返回：Tensor([seq_len,batch_size,hidden_size]),Tensor[n_layers,batch_size,hidden_size]
	def forward(self,input_seq,input_lengths,hidden=None):
		#print("what the fuck!")
		# 取词向量，input_seq：longTensor,[numT,batch_size,embedding_size]
		# input_lengths [numT]
		#print("rnn_type:",self.rnn_type)
		#print(self.gru)
		embedded = self.embedding(input_seq)
		# 转packed_sequence
		packed = torch.nn.utils.rnn.pack_padded_sequence(embedded,input_lengths)
		# 输入进LSTM或者GRU
		if self.rnn_type == 'LSTM':
			output,(h,c) = self.lstm(packed,None)
			# 再使用pad_packed_sequence返回成填充的(填充的全0)，按时间步排列的输出供使用，h和c不用管
			# 即逆torch.nn.utils.rnn.pack_packed_sequence变换
			output,_ = torch.nn.utils.rnn.pad_packed_sequence(output)
			# output维度： [seq_len,batch_size,hidden_size*n_directions]
			# 如果是双向的，返回两个方向的加和，为了方便处理
			if self.bidirectional == True:
				output = output[:,:,:self.hidden_size] + output[:,:,self.hidden_size:]
				# 还少了一步吧？双向的话应该把h,c的双向也加起来
			return output,(h,c)

		elif self.rnn_type == 'GRU':
			output,hidden = self.gru(packed,hidden)
			print(hidden.size())
			output,_ = torch.nn.utils.rnn.pad_packed_sequence(output)
			if self.bidirectional == True:
				output = output[:,:,:self.hidden_size] + output[:,:,self.hidden_size:]
			return output,hidden


# DecoderRNN
class DecoderRNN(nn.Module):
	# 参数：词向量层，输出维度一定是和词典维度一样，用一个全连接转换
	def __init__(self,embedding,hidden_size,output_size,n_layers=1,rnn_type='LSTM',use_ATTN=False,dropout = 0.1):
		super(DecoderRNN,self).__init__()
		self.use_ATTN = use_ATTN
		self.embedding = embedding
		self.rnn_type = rnn_type
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.n_layers = n_layers
		self.dropout = dropout

		# RNN
		if self.rnn_type == 'LSTM':
			self.lstm = nn.LSTM(self.hidden_size,self.hidden_size,
				self.n_layers,dropout=(0 if n_layers == 1 else self.dropout))
		elif self.rnn_type == 'GRU':
			self.gru = nn.GRU(self.hidden_size,self.hidden_size,
				self.n_layers,dropout=(0 if n_layers == 1 else self.dropout))

		# 使用扩展的注意力机制，如果ATTN=True，则RNN的输出和注意力向量连接torch.cat一下，然后再转成hidden_size，再转成output_szie
		if self.use_ATTN == True:
			self.attn_layer = AttentionModel(hidden_size)
			self.attn_linear = nn.Linear(self.hidden_size*2,self.hidden_size)
		# 没有注意力机制			
		self.outLayer = nn.Linear(self.hidden_size,self.output_size)

	# input_seq就是batch_size个SOS
	# 注意last_hidden！如果是GRU就传进来Tensor，否则传进来(h0,c0)
	def forward(self,encoder_outputs,input_seq,last_hidden):

		SOS_embed = self.embedding(input_seq)
		if SOS_embed.size(0) != 1:
			raise ValueError("Decoder Start seq_len should be 1 !")
		
		# 计算RNN前向传播
		if self.rnn_type == 'LSTM':
			outputs,(h,c) = self.lstm(SOS_embed,last_hidden) #小心这里可能出错 是个元组
			#print("LSTM_outpus size:",outputs.size())
			#print(outputs)
			# 注意力机制
			if self.use_ATTN == True:
				# 计算注意力权重
				attn_weights = self.attn_layer(encoder_outputs,outputs)
				# 计算上下文 [batch_size,1,hidden_size]
				context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
				#print("LSTM输出维度：",outputs.size())
				# 维度对齐
				outputs = outputs.squeeze(0) #[batch_size,hidden_size]
				context = context.squeeze(1) #[batch_size,hidden_size]
				#print("LSTM压缩之后：",outputs.size())
				# 连接
				tmp = torch.cat((outputs,context),1) # [batch_size,hidden_size*2]
				# 维度
				outputs_tmp = self.attn_linear(tmp) # [batch_size,hidden_size]
				outputs_tmp = torch.tanh(outputs_tmp)
				outputs_final = self.outLayer(outputs_tmp) # [batch_size,output_size]
				#print("输出维度：",outputs_final.size())
				return outputs_final,(h,c),attn_weights			
			else:
				outputs = outputs.squeeze(0)
				outputs = self.outLayer(outputs) # 转到output_size
				return outputs,(h,c)
		# 使用GRU
		else:
			outputs,h = self.gru(SOS_embed,last_hidden)
			# 注意力机制
			if self.use_ATTN == True:
				# 计算注意力权重
				attn_weights = self.attn_layer(encoder_outputs,outputs)
				# 计算上下文 [batch_size,1,hidden_size]
				context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
				# 维度对齐
				outputs = outputs.squeeze(0) #[batch_size,hidden_size]
				context = context.squeeze(1) #[batch_size,hidden_size]
				# 连接
				tmp = torch.cat((outputs,context),1) # [batch_size,hidden_size*2]
				# 维度
				outputs_tmp = self.attn_linear(tmp) # [batch_size,hidden_size]
				outputs_tmp = torch.tanh(outputs_tmp)
				outputs_final = self.outLayer(outputs_tmp) # [batch_size,output_size]
				return outputs_final,h,attn_weights
			
			else:
				outputs = outputs.squeeze(0)
				outputs = self.outLayer(outputs) # 转到output_size
				return outputs,h