

import torch.nn as nn
import torch.nn.functional as F
import torch


def weights_init_normal(m): #takes an input m, which is the layer/module of the neural network.
    classname = m.__class__.__name__ #first obtains the classname of the layer 
    #If the classname contains "Conv" (indicating it's a convolutional layer),
    #  the function initializes the weight data of that layer using normal distribution with mean 0.0 and standard deviation 0.02. 
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02) #This is done using torch.nn.init.normal_
        if hasattr(m, "bias") and m.bias is not None: #If the layer has a bias term  the function sets the bias data to a constant value of 0.0 using torch.nn.init.constant_.
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


#############################################
#           RESNET SINGLE RESIDUAL BLOCK
###############################################
#A residual block consists of two 3x3 convolutional layers followed by an instance normalization and ReLU activation function.
#  The output of the second convolutional layer is then added to the original input of the residual block.

class ResidualBlock(nn.Module): #defines a single residual block.
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), #pad the input with reflection padding of size 1
            nn.Conv2d(in_features, in_features, 3), #in_features input channels and in_features output channels
            nn.InstanceNorm2d(in_features), #normalizes the activations channel-wise (instance-wise) independently.
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        # takes an input x, passes it through the block (sequence of layers), and adds the result to the original input x
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    # A tuple representing the shape of the input image (e.g., channels, height, width).,  number of residual blocks to use in the generator.
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet, self).__init__()

        channels = input_shape[0] #initial number of channels is set to the number of channels in the input shape

        # Initial convolution block consists of 
        #a reflection padding layer, a 7x7 convolutional layer, instance normalization, and a ReLU activation function.
        out_features = 64 #represents the number of output channels after the initial convolution block.
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        #Two downsampling blocks are created using a for loop with a stride of 2 in the convolutional layers,
        # effectively reducing the spatial dimensions of the feature maps while increasing the number of channels.
        #  The number of out_features is doubled after each downsampling block.
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        # The number of residual blocks is determined by the num_residual_blocks parameter passed to the constructor.
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Upsampling
        # Two upsampling blocks are created using a for loop with a scale factor of 2
        for _ in range(2):
            out_features //= 2 #number of out_features is halved after each upsampling block.
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        # the output layer consists of a reflection padding layer, a 7x7 convolutional layer, and a Tanh activation function.
        # The number of output channels in the final layer is set to be the same as the number of input channels (restoring the original number of channels).
        model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x): #defines the forward pass of the generator
        return self.model(x) #takes an input x and passes it through the generator model (self.model) defined in the constructor.
        #result of the forward pass is returned as the output of the generator.


##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        # the discriminator downsamples the input image by a factor of 2^4 
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4) #output shape of the discriminator, often referred to as the PatchGAN output
        
#function defines a single block of the discriminator, which consists of a convolutional layer with optional instance normalization and a leaky ReLU activation function.
        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize: #indicating whether instance normalization should be applied after the convolutional layer (default is True).
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        #discriminator's architecture is defined using a series of discriminator blocks.
        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)), #used to add padding to the right and bottom of the feature map before applying the last convolutional layer. done to ensure the feature map size matches the calculated output shape.
            nn.Conv2d(512, 1, 4, padding=1)# reduces the feature map to a single-channel output, representing the probability of the input image being real (1) or fake (0).

        )


    def forward(self, img): #takes an input img and passes it through the discriminator model (self.model) defined in the constructor.
        return self.model(img) #discriminator's output, representing the probability of the input image being real or fake.
