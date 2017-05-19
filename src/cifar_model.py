import sys
import torch

########################################################################################################################
# Small convnet -- Just for example
########################################################################################################################

class ConvNet(torch.nn.Module):

    def __init__(self,image_size,num_classes):
        super(ConvNet,self).__init__()
        # Get image size (we only deal with square images)
        size,_,num_channels=image_size

        # Define a convolutional layer
        self.conv1=torch.nn.Conv2d(num_channels,10,kernel_size=3,stride=1,padding=1)
        # Define a rectified linear unit
        self.relu=torch.nn.ReLU()
        # Define a pooling layer
        self.pool=torch.nn.MaxPool2d(kernel_size=2,stride=2)
        # Define another convolutional layer
        self.conv2=torch.nn.Conv2d(10,20,kernel_size=3,stride=1,padding=1)
        # We do not need to define model relus nor pooling (no parameters to train, we can reuse the same ones)

        # Define final fully-connected layers
        self.fc1=torch.nn.Linear(20*8*8,120)
        self.fc2=torch.nn.Linear(120,num_classes)
        return

    def forward(self,x):
        # First stage: convolution -> relu -> pooling
        y=self.pool(self.relu(self.conv1(x)))
        # Second stage: convolution -> relu -> pooling
        y=self.pool(self.relu(self.conv2(y)))
        # Reshape to batch_size-by-whatever
        y=y.view(x.size(0),-1)
        # Last stage: fc -> relu -> fc
        y=self.fc2(self.relu(self.fc1(y)))
        # Return predictions
        return y

########################################################################################################################


########################################################################################################################
# He et al. "Deep residual learning for image recognition." Proceedings of the IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), pp. 770-778. 2016.
# https://arxiv.org/abs/1512.03385
########################################################################################################################

class ResNet(torch.nn.Module):

    def __init__(self,image_size,num_classes):
        super(ResNet,self).__init__()
        # Get image size (we only deal with square images)
        size,_,num_channels=image_size

        # Some configuration of our model (corresponding to ResNet50)
        channels_init=64
        compression=4
        structure=[(3,64,1),(4,128,2),(6,256,2),(3,512,2)]

        # Define initial module (non-residual)
        self.conv1=torch.nn.Conv2d(num_channels,channels_init,kernel_size=3,stride=1,padding=1)
        self.bn1=torch.nn.BatchNorm2d(channels_init)
        self.relu=torch.nn.ReLU()

        # Stack residual blocks following structure
        self.layers=[]
        channels_current=channels_init
        size_current=size
        # Loop structure elements
        for n,channels,stride in structure:
            # Loop internal structure layers
            for i in range(n):
                # Only do a stride!=1 at first piece of the super-block
                if i>0: stride=1
                # Instantiate and append block
                b=Block(channels_current,channels,compression,stride).cuda()
                self.layers.append(b)
                # Update current number of channels and image size
                channels_current=channels
                size_current=size_current//stride
        # Convert a list of layers into a nested, sequential operation (options for more elaborate relations than nested exist, see ModuleList)
        self.layers=torch.nn.Sequential(*self.layers)

        # Define an average pooling layer
        self.avgpool=torch.nn.AvgPool2d(size_current)

        # Define the final classification layer
        self.fc=torch.nn.Linear(channels_current,num_classes)

        return

    def forward(self,x):
        # Apply initial module
        y=self.relu(self.bn1(self.conv1(x)))
        # Apply all blocks
        y=self.layers(y)
        # Apply pooling
        y=self.avgpool(y)
        # Reshape
        y=y.view(x.size(0),-1)
        # Apply classification layer
        y=self.fc(y)
        # Return prediction
        return y

########################################################################################################################

class Block(torch.nn.Module):

    def __init__(self,channels_current,channels,compression,stride):
        super(Block,self).__init__()
        # Set the number of internal channels
        channels_internal=channels//compression

        # Define a relu
        self.relu=torch.nn.ReLU()
        # Define the three sequential convolutions + batch normalization
        self.conv1=torch.nn.Conv2d(channels_current,channels_internal,kernel_size=1)
        self.bn1=torch.nn.BatchNorm2d(channels_internal)
        self.conv2=torch.nn.Conv2d(channels_internal,channels_internal,kernel_size=3,stride=stride,padding=1)
        self.bn2=torch.nn.BatchNorm2d(channels_internal)
        self.conv3=torch.nn.Conv2d(channels_internal,channels,kernel_size=1)
        self.bn3=torch.nn.BatchNorm2d(channels)

        # Create the shortcut
        self.shortcut=torch.nn.Sequential()
        # If number of channels changed or we did some downsampling, the shortcut needs to take care of that
        if channels!=channels_current or stride!=1:
            # A list of layers: convolution + batch normalization
            self.shortcut=torch.nn.Sequential(
                torch.nn.Conv2d(channels_current,channels,kernel_size=1,stride=stride),
                torch.nn.BatchNorm2d(channels)
            )

        return

    def forward(self,x):
        # Apply the three sequential convolutions + batch normalization
        y=self.relu(self.bn1(self.conv1(x)))
        y=self.relu(self.bn2(self.conv2(y)))
        y=self.bn3(self.conv3(y))
        # Add the shortcut
        y+=self.shortcut(x)
        # Activation
        y=self.relu(y)
        # Return predictions
        return y

########################################################################################################################
