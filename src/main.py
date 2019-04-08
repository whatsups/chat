import argparse
from loadData import loadPreparedData
from prepareData import loadTraingingData
from train import train

#from evaluate import runTest

def parse():
    parser = argparse.ArgumentParser(description='Attention Seq2Seq Model')
    parser.add_argument('-tr', '--train', help='Train the model with corpus') #-tr,arg.train输入的是语料的路径
    parser.add_argument('-te', '--test', help='Test the saved model')
    parser.add_argument('-l', '--load', help='Load the model and train') #加载预训练过的模型，可以是别人的也可以是自己的
    parser.add_argument('-c', '--corpus', help='Test the saved model with vocabulary of the corpus')

    parser.add_argument('-r', '--reverse', action='store_true', help='Reverse the input sequence')

    parser.add_argument('-f', '--filter', action='store_true', help='Filter to small training data set')
    parser.add_argument('-i', '--input', action='store_true', help='Test the model by input the sentence')
    parser.add_argument('-it', '--iteration', type=int, default=10000, help='Train the model with it iterations')
    parser.add_argument('-p', '--print', type=int, default=100, help='Print every p iterations')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='Batch size') #batch_size
    parser.add_argument('-la', '--layer', type=int, default=1, help='Number of layers in encoder and decoder') #层数 
    parser.add_argument('-hi', '--hidden', type=int, default=256, help='Hidden size in encoder and decoder') # hidden维度
    parser.add_argument('-be', '--beam', type=int, default=1, help='Hidden size in encoder and decoder')
    parser.add_argument('-s', '--save', type=int, default=500, help='Save every s iterations')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('-d', '--dropout', type=float, default=0.1, help='Dropout probability for rnn and dropout layers')
    parser.add_argument('-at', '--use_ATTN', type=int, default=0, help='Use Attention or not')

    args = parser.parse_args()
    return args

def parseFilename(filename, test=False):
    filename = filename.split('/')
    dataType = filename[-1][:-4] # remove '.tar'
    parse = dataType.split('_')
    reverse = 'reverse' in parse
    layers, hidden = filename[-2].split('_')
    n_layers = int(layers.split('-')[0])
    hidden_size = int(hidden)
    return n_layers, hidden_size, reverse

def cmd():

    # 控制参数
    corpus = args.train
    n_iteration = args.iteration
    print_every = args.print
    save_every = args.save
    learning_rate = args.learning_rate
    n_layers = args.layer
    hidden_size = args.hidden
    barch_size = args.batch_size
    beam_size = args.beam
    inputs = args.input
    dropout = args.dropout
    use_ATTN = args.use_ATTN

    if args.train and not args.load: #训练模型
        # 准备数据
        voc,pairs = loadPreparedData(corpus)
        train_batches = loadTraingingData(voc,pairs,corpus,batch_size,n_iteration)
        print("INFO:start training ...")
        # 训练
        train(voc,train_batches,n_iteration,learning_rate,batch_size,n_layers,hidden_size,print_every,
            save_every,dropout,use_ATTN=use_ATTN,decoder_learning_ratio=5.0);

#train脚本参数：voc,pair_batches,n_iteration,learning_rate,batch_size,n_layers,hidden_size,print_every,
#            save_every,dropout,rnn_type='LSTM',bidirectional=False,use_ATTN=False,decoder_learning_ratio=5.0

#loadData参数：corpus 语料路径 返回 voc,pairs

#loadTraingingData参数：voc,pairs,corpus,batch_size,n_iterations 返回：pair_batches

	

if __name__ == __main__():
	cmd()
	pass
