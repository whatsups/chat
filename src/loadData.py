"""
	读取语料，生成字典和训练语句对
	保存
"""
import torch
import re
import os
import unicodedata
from config import SOS_token,EOS_token,PAD_token,MAXLEN,save_dir

MIN_COUNT = 8
# 词典类
class Voc:
	def __init__(self,name):
		self.name = name
		self.word2index = {}
		self.index2word = {0:"SOS",1:"EOS",2:"PAD"}
		self.word2count = {}
		self.n_words = 3 #SOS,EOS,PAD

	def addSentence(self,sentence):
		for word in sentence.split(" "):
			self.addWord(word)

	def addWord(self,word):
		if word not in self.word2index:
			self.word2index[word] = self.n_words
			self.word2count[word] = 1
			self.index2word[self.n_words] = word
			self.n_words  = self.n_words+1
		else:
			self.word2count[word] = self.word2count[word]+1

# 引用自：github.com/
# 去空，小写，分离标点
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

# 引用自：http://stackoverflow.com/a/518232/2809427
# unicode编码转Acsii
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# 判断句子对长度是否小于规定长度
def judgeLen(pair):
	if len(pair[0].split(" ")) < MAXLEN and len(pair[1].split(" "))<MAXLEN:
		return True
	return False	

# 返回所有符合长度要求的句子
def filterMaxLenPairs(pairs):
	pairs_new = [pair for pair in pairs if judgeLen(pair)]
	return pairs_new

# 过滤字典，频率太少的单词去掉
def filterVoc(voc,corpus_name):
	voc_new = Voc(corpus_name)

	for i in range(voc.n_words):
		if i == 0 or i == 1 or i == 2:
			continue
		if voc.word2count[voc.index2word[i]] >= MIN_COUNT:
			voc_new.word2index[voc.index2word[i]] = voc_new.n_words
			voc_new.word2count[voc.index2word[i]] = voc.word2count[voc.index2word[i]]
			voc_new.index2word[voc_new.n_words] = voc.index2word[i]
			voc_new.n_words = voc_new.n_words + 1

	return voc_new

def noUNK(pair,voc):
	p1 = pair[0].split(' ')
	p2 = pair[1].split(' ')

	for i in range(len(p1)):
		if p1[i] not in voc.word2index:
			return False
	for i in range(len(p2)):
		if p2[i] not in voc.word2index:
			return False
	return True			

# 新的voc,删除有UNK的句子
def filterUNKpairs(pairs,voc):
	pairs_new = [[pair[0],pair[1]] for pair in pairs if noUNK(pair,voc)]
	return pairs_new

def prepareData(corpus,corpus_name):
	voc = Voc(corpus_name)
	# 打开文本文件，读行存于pairs并统计voc
	with open(corpus) as f:
		lines = f.readlines() # readlines返回list，占内存
		#f.close()
	# 去空	
	lines = [line.strip() for line in lines]
	it = iter(lines)
	pairs = [[x,next(it)] for x in it]
	print('INFO:Read %d pairs in the corpus file!'%len(pairs))
	# 去掉过长的句子
	pairs = filterMaxLenPairs(pairs)
	# 标准化处理
	pairs = [[normalizeString(pair[0]),normalizeString(pair[1])] for pair in pairs]
	print('INFO:After filter, Remain %d pairs in the corpus file!'%len(pairs))
	# 建立词典
	for pair in pairs:
		voc.addSentence(pair[0])
		voc.addSentence(pair[1])
	print("INFO:Vocabulary Size:", voc.n_words)		
	print("INFO:End Build vocabulary!")		

	directory = os.path.join(save_dir,'training_data',corpus_name)
	if not os.path.exists(directory):
		os.makedirs(directory)

	# 保存起来
	torch.save(voc,os.path.join(directory,'{!s}.tar'.format('vocabulary')))    
	torch.save(pairs,os.path.join(directory,'{!s}.tar'.format('training_pairs')))

	return voc,pairs

# 加载预处理数据
def loadPreparedData(corpus):
	corpus_name = corpus.split('/')[-1].split('.')[0]
	# print("INFO:Corpus route: %s , name: " %(corpus,corpus_name))

	try:
		print('INFO:Start load prepared training data and vocabulary...')
		voc = torch.load(os.path.join(save_dir,'training_data',corpus_name,'vocabulary.tar'))
		pairs = torch.load(os.path.join(save_dir,'training_data',corpus_name,'training_pairs.tar'))
	except FileNotFoundError:
		print("INFO:Training_data have not prepared,Start prepare training data and vocabulary...")
		voc,pairs = prepareData(corpus,corpus_name)

	print("INFO:End prepare training data and vocabulary!")

	voc_new = filterVoc(voc,corpus_name)
	print('INFO:单词总数：',voc.n_words)
	print('INFO:过滤之后还剩下单词：',voc_new.n_words)
	pairs_new = filterUNKpairs(pairs,voc_new)
	print('INFO:语句对总数:',len(pairs))
	print('INFO:过滤之后还剩下语句:',len(pairs_new))
	#print(pairs_new)
	return voc_new,pairs_new


"""
	预处理测试脚本：
	import os
	corpus = os.path.join(os.path.dirname(os.getcwd()),'corpus/conservation.txt')
	print(corpus)
	voc,p = loadPareparedData(corpus)
	print(voc.n_words）
"""