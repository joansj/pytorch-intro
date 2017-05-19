import sys,argparse
import numpy as np
import cPickle as pickle
from tqdm import tqdm

# Parse arguments
parser=argparse.ArgumentParser(description='Main script using CIFAR-10')
parser.add_argument('--seed',default=333,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--filename_in',default='../dat/ptb.pkl',type=str,required=False,help='(default=%(default)s)')
parser.add_argument('--batch_size',default=20,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--num_epochs',default=100,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--patience',default=3,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--learning_rate',default=1e-3,type=float,required=False,help='(default=%(default)f)')
parser.add_argument('--bptt',default=35,type=int,required=False,help='(default=%(default)d)')
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
optimizer=torch.optim.Adam(model.parameters(),lr=args.learning_rate)

########################################################################################################################
# Train/test routines
########################################################################################################################

def train(data,model,criterion,optimizer):

    # Set model to training mode (we're using dropout)
    model.train()
    # Get initial hidden and memory states
    states=model.get_initial_states(args.batch_size)

    # Loop sequence length (train)
    for i in tqdm(range(0,data.size(1)-1,args.bptt),desc='> Train',ncols=100,ascii=True):

        # Get the chunk
        seqlen=int(np.min([args.bptt,data.size(1)-1-i]))

        # Wrap the variables into the gradient propagation chain and move them to the GPU
        x=torch.autograd.Variable(data[:,i:i+seqlen]).cuda()
        y=torch.autograd.Variable(data[:,i+1:i+seqlen+1]).cuda()

        # Truncated backpropagation
        states=model.detach(states)

        # Forward pass
        logits,states=model.forward(x,states)
        loss=criterion(logits,y.view(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def eval(data,model,criterion):

    # Set model to evaluation mode (we're using dropout)
    model.eval()
    # Get initial hidden and memory states
    states=model.get_initial_states(args.batch_size)

    # Loop sequence length (validation)
    total_loss=0
    num_loss=0
    for i in tqdm(range(0,data.size(1)-1,args.bptt),desc='> Eval',ncols=100,ascii=True):

        # Get the chunk
        seqlen=int(np.min([args.bptt,data.size(1)-1-i]))

        # Wrap the variables into the gradient propagation chain and move them to the GPU
        x=torch.autograd.Variable(data[:,i:i+seqlen],volatile=True).cuda()
        y=torch.autograd.Variable(data[:,i+1:i+seqlen+1],volatile=True).cuda()

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
loss_vals=[]
patience=args.patience
for e in tqdm(range(args.num_epochs),desc='Epoch',ncols=100,ascii=True):

    # Train
    model=train(data_train,model,criterion,optimizer)

    # Validation
    l=eval(data_valid,model,criterion)
    loss_vals.append(l)

    # Test
    loss_test=eval(data_test,model,criterion)

    # Report
    msg='Epoch %d: \tValid loss=%.4f \tTest loss=%.4f \tTest perplexity=%.1f'%(e+1,loss_vals[-1],loss_test,np.exp(loss_test))
    if len(loss_vals)>1:
        if loss_vals[-1]<=np.min(loss_vals[:-1]):
            patience=args.patience
            msg+=' \t(*)'
        else:
            patience-=1
    tqdm.write(msg)
    if patience==0: break

########################################################################################################################
