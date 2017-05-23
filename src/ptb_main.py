import sys,argparse
import numpy as np
import cPickle as pickle
from tqdm import tqdm

# Parse arguments
parser=argparse.ArgumentParser(description='Main script using CIFAR-10')
parser.add_argument('--seed',default=333,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--filename_in',default='../dat/ptb.pkl',type=str,required=False,help='(default=%(default)s)')
parser.add_argument('--batch_size',default=20,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--num_epochs',default=40,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--bptt',default=35,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--learning_rate',default=20,type=float,required=False,help='(default=%(default)f)')
parser.add_argument('--clip_norm',default=0.25,type=float,required=False,help='(default=%(default)f)')
parser.add_argument('--anneal_factor',default=2.0,type=float,required=False,help='(default=%(default)f)')
args=parser.parse_args()
print '*'*100,'\n',args,'\n','*'*100

# Import pytorch stuff
import torch
import torchvision

# Set random seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)
else: print '[CUDA unavailable]'; sys.exit()

# Model import
import ptb_model

########################################################################################################################
# Load data
########################################################################################################################

print 'Load data...'

# Load numpy data
data_train,data_valid,data_test,vocabulary_size=pickle.load(open(args.filename_in,'rb'))

# Make it pytorch
data_train=torch.LongTensor(data_train.astype(np.int64))
data_valid=torch.LongTensor(data_valid.astype(np.int64))
data_test=torch.LongTensor(data_test.astype(np.int64))

# Make batches
num_batches=data_train.size(0)//args.batch_size         # Get number of batches
data_train=data_train[:num_batches*args.batch_size]     # Trim last elements
data_train=data_train.view(args.batch_size,-1)          # Reshape
num_batches=data_valid.size(0)//args.batch_size
data_valid=data_valid[:num_batches*args.batch_size]
data_valid=data_valid.view(args.batch_size,-1)
num_batches=data_test.size(0)//args.batch_size
data_test=data_test[:num_batches*args.batch_size]
data_test=data_test.view(args.batch_size,-1)


########################################################################################################################
# Inits
########################################################################################################################

print 'Init...'

# Instantiate and init the model, and move it to the GPU
model=ptb_model.BasicRNNLM(vocabulary_size).cuda()

# Define loss function
criterion=torch.nn.CrossEntropyLoss(size_average=False)

# Define optimizer
optimizer=torch.optim.SGD(model.parameters(),lr=args.learning_rate)

########################################################################################################################
# Train/test routines
########################################################################################################################

def train(data,model,criterion,optimizer):

    # Set model to training mode (we're using dropout)
    model.train()
    # Get initial hidden and memory states
    states=model.get_initial_states(data.size(0))

    # Loop sequence length (train)
    for i in tqdm(range(0,data.size(1)-1,args.bptt),desc='> Train',ncols=100,ascii=True):

        # Get the chunk and wrap the variables into the gradient propagation chain + move them to the GPU
        seqlen=int(np.min([args.bptt,data.size(1)-1-i]))
        x=torch.autograd.Variable(data[:,i:i+seqlen]).cuda()
        y=torch.autograd.Variable(data[:,i+1:i+seqlen+1]).cuda()

        # Truncated backpropagation
        states=model.detach(states)     # Otherwise the model would try to backprop all the way to the start of the data set

        # Forward pass
        logits,states=model.forward(x,states)
        loss=criterion(logits,y.view(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(),args.clip_norm)
        optimizer.step()

    return model


def eval(data,model,criterion):

    # Set model to evaluation mode (we're using dropout)
    model.eval()
    # Get initial hidden and memory states
    states=model.get_initial_states(data.size(0))

    # Loop sequence length (validation)
    total_loss=0
    num_loss=0
    for i in tqdm(range(0,data.size(1)-1,args.bptt),desc='> Eval',ncols=100,ascii=True):

        # Get the chunk and wrap the variables into the gradient propagation chain + move them to the GPU
        seqlen=int(np.min([args.bptt,data.size(1)-1-i]))
        x=torch.autograd.Variable(data[:,i:i+seqlen],volatile=True).cuda()
        y=torch.autograd.Variable(data[:,i+1:i+seqlen+1],volatile=True).cuda()

        # Truncated backpropagation
        states=model.detach(states)     # Otherwise the model would try to backprop all the way to the start of the data set

        # Forward pass
        logits,states=model.forward(x,states)
        loss=criterion(logits,y.view(-1))

        # Log stuff
        total_loss+=loss.data.cpu().numpy()
        num_loss+=np.prod(y.size())

    return float(total_loss)/float(num_loss)

########################################################################################################################
# Train/validation/test
########################################################################################################################

print 'Train...'

# Loop training epochs
lr=args.learning_rate
best_val_loss=np.inf
for e in tqdm(range(args.num_epochs),desc='Epoch',ncols=100,ascii=True):

    # Train
    model=train(data_train,model,criterion,optimizer)

    # Validation
    val_loss=eval(data_valid,model,criterion)

    # Anneal learning rate
    if val_loss<best_val_loss:
        best_val_loss=val_loss
    else:
        lr/=args.anneal_factor
        optimizer=torch.optim.SGD(model.parameters(),lr=lr)

    # Test
    test_loss=eval(data_test,model,criterion)

    # Report
    msg='Epoch %d: \tValid loss=%.4f \tTest loss=%.4f \tTest perplexity=%.1f'%(e+1,val_loss,test_loss,np.exp(test_loss))
    tqdm.write(msg)

########################################################################################################################
