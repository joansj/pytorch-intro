import sys,argparse
import numpy as np
from tqdm import tqdm

# Parse arguments
parser=argparse.ArgumentParser(description='Main script using CIFAR-10')
parser.add_argument('--seed',default=17724,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--log_interval',default=200,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--data_folder',default='../dat/',type=str,required=False,help='(default=%(default)s')
parser.add_argument('--batch_size',default=128,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--num_epochs',default=300,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--learning_rate',default=1e-3,type=float,required=False,help='(default=%(default)f)')
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
import cifar_model

########################################################################################################################
# Load data
########################################################################################################################

print 'Load data...'

# Set some data set parameters
image_size=32
image_channels=3
num_classes=10

# Prepare data augmentation
train_transform=torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(image_size,padding=4),    # Random crop sub-parts of the image
    torchvision.transforms.RandomHorizontalFlip(),              # Horizontal flip with probability=0.5
    torchvision.transforms.ToTensor(),                          # Conversion from PILImage to Tensor (also normalizes between 0 and 1)
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),       # Standardize
])
test_transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Prepare data loaders
train_set=torchvision.datasets.CIFAR10(root=args.data_folder,train=True,transform=train_transform,download=True)    # Downloads data and points to it
train_loader=torch.utils.data.DataLoader(dataset=train_set,batch_size=args.batch_size,shuffle=True,num_workers=3)   # Provides an iterator over the elements of the data set
test_set=torchvision.datasets.CIFAR10(root=args.data_folder,train=False,transform=test_transform,download=True)
test_loader=torch.utils.data.DataLoader(dataset=test_set,batch_size=args.batch_size,shuffle=False,num_workers=3)

########################################################################################################################
# Inits
########################################################################################################################

print 'Init...'

# Instantiate and init the model, and move it to the GPU
#model=cifar_model.ConvNet((image_size,image_size,image_channels),num_classes).cuda()
model=cifar_model.ResNet((image_size,image_size,image_channels),num_classes).cuda()
#model=torch.nn.DataParallel(model,device_ids=[0,2])        # Just to have an idea how easy it is to parallelize on GPUs

# Define loss function
criterion=torch.nn.CrossEntropyLoss()

# Define optimizer
optimizer=torch.optim.Adam(model.parameters(),lr=args.learning_rate)

########################################################################################################################
# Train model
########################################################################################################################

print 'Train...'

# Set model to training mode (we're using batch normalization)
model.train()

# Loop training epochs
lossvals=[]
for e in tqdm(range(args.num_epochs),desc='Epoch',ncols=100,ascii=True):

    # Loop batches
    for images,labels in tqdm(train_loader,desc='> Batch',ncols=100,ascii=True):

        # Wrap the variables into the gradient propagation chain and move them to the GPU
        images=torch.autograd.Variable(images).cuda()
        labels=torch.autograd.Variable(labels).cuda()
        #print images[0]; sys.exit()

        # Forward pass
        outputs=model.forward(images)
        loss=criterion(outputs,labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log stuff
        lossvals.append(loss.data.cpu().numpy())
        if len(lossvals)%args.log_interval==0:
            msg='Epoch %d, iter %.2e: \tLoss=%.4f \tSmooth loss=%.4f'%(e+1,len(lossvals),lossvals[-1],np.mean(lossvals[-args.log_interval:]))
            tqdm.write(msg)

########################################################################################################################
# Test model
########################################################################################################################

print 'Test...'

# Change model to evaluation mode (we're using batch normalization)
model.eval()

# Loop images
hits=[]
for images,labels in tqdm(test_loader,desc='Evaluation',ncols=100,ascii=True):

    # Wrap the variables into the gradient propagation chain and move them to the GPU
    images=torch.autograd.Variable(images,volatile=True).cuda()
    labels=labels.cuda()

    # Forward pass
    outputs=model.forward(images)

    # Eval
    _,predicted=torch.max(outputs.data,1)
    correct=(labels==predicted).int().cpu().numpy()
    hits+=list(correct)

# Report
print '='*100,'\nTest accuracy = %.1f%%\n'%(100*np.mean(hits)),'='*100

########################################################################################################################
