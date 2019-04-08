from loadData import loadPreparedData
from prepareData import loadTraingingData
from train import train

if __name__ == '__main__':
	
    # 这里添加语料文件路径
    corpus = "corpus_path" 
    batch_size = 256
    n_iteration = 100
    learning_rate = 0.0001
    n_layers = 4
    hidden_size = 512
    print_every = 5
    save_every = 20
    voc,pairs = loadPreparedData(corpus)
    train_batches = loadTraingingData(voc,pairs,corpus,batch_size,n_iteration)
    print("INFO:Start ...")
    dropout = 0.1
    load_pretrain = '1000_seq2seq_bidir_model.tar'
    # 训练
    train(load_pretrain,voc,train_batches,n_iteration,learning_rate,batch_size,n_layers,hidden_size,print_every,
        save_every,dropout,use_ATTN=True,decoder_learning_ratio=5.0)
    print("INFO:End ...")
