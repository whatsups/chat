# 测试对话
from evaluate import runTest

if __name__ == __main__():
	# 模型路径
	modelFile = "model_path"
	# 参数与训练模型保持一致
	n_layers = 4
	hidden_size = 512
	beam_size = 4 #beam search大小
	# 语料路径
	corpus = "corpus_path"
	# 另外需要注意测试时的模型结构需和训练时一致，比如使用注意力，双向LSTM
	runTest(n_layers, hidden_size,modelFile,beam_size,corpus)