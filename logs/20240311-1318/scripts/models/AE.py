import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50
from torch.nn.init import kaiming_normal_


class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None, stride=1):
        # The class inherits from nn.Module, making it a component of a neural network.
        super(ResBlock, self).__init__()
        # If no output dimension is specified, it defaults to the input dimension.
        if outdim is None:
            outdim = indim
        # Determines if downsampling is necessary. Downsampling is skipped if input and output dimensions are the same and stride is 1.
        if indim == outdim and stride == 1:
            self.downsample = None
        else:
            # If downsampling is needed, it's done through a 1x1 convolution followed by batch normalization.
            self.downsample = nn.Sequential(
                nn.Conv2d(indim, outdim, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outdim)
            )

        # First convolutional layer of the block with kernel size 3x3, padding to keep size constant, stride that may vary, and no bias term.
        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride, bias=False)
        # Batch normalization after the first convolutional layer.
        self.bn1 = nn.BatchNorm2d(outdim)
        # Second convolutional layer of the block with kernel size 3x3, padding to keep size constant, and no bias term.
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1, bias=False)
        # Batch normalization after the second convolutional layer.
        self.bn2 = nn.BatchNorm2d(outdim)

    def forward(self, x):
        # Saves the input value to be added back to the output (for the residual connection).
        identity = x

        # Applies the first convolution, followed by batch normalization, and then ReLU activation function.
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        # Applies the second convolution followed by batch normalization.
        out = self.conv2(out)
        out = self.bn2(out)

        # If downsampling was defined, it is applied to the identity (input) before adding it to the output.
        if self.downsample is not None:
            identity = self.downsample(x)

        # Adds the identity to the output (residual connection) and applies ReLU activation function.
        out += identity
        out = F.relu(out)

        return out


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # Initialize ResNet-50 with the option to load pretrained ImageNet parameters
        resnet = resnet50(pretrained=False)
        # Redefine the initial convolution layer to accept 8 input channels instead of 3
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Initial convolution layer
        # Apply Kaiming initialization to the modified conv1 layer
        kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.bn1 = resnet.bn1      # Batch normalization following the initial convolution
        self.relu = resnet.relu    # ReLU activation after batch normalization
        self.maxpool = resnet.maxpool  # Max pooling to reduce spatial dimensions

        # Residual blocks from ResNet-50 for hierarchical feature extraction
        self.res2 = resnet.layer1  # Outputs features at 1/4 scale with 256 channels
        self.res3 = resnet.layer2  # Outputs features at 1/8 scale with 512 channels
        self.res4 = resnet.layer3  # Outputs features at 1/16 scale with 1024 channels

        # Register max and min values as buffers for normalization
        # These values should be defined prior to initializing this class
        #                                                       X    Y     U      V
        self.register_buffer('max_values', torch.FloatTensor([33.0, 13.0, 0.23, 0.09]).view(1, 4, 1, 1))
        self.register_buffer('min_values', torch.FloatTensor([25.0, -13.0, -0.29, -0.11]).view(1, 4, 1, 1))

    def forward(self, in_f):
        # Normalize the input feature based on the registered max and min values
        f = (in_f - self.min_values) / (self.max_values - self.min_values)

        # Process the normalized input through the initial ResNet-50 layers
        x = self.conv1(f)
        x = self.bn1(x)
        r1 = self.relu(x)  # Features after initial processing, at 1/2 scale
        x = self.maxpool(r1)  # Features after max pooling, at 1/4 scale

        # Further processing through ResNet-50 residual blocks
        r2 = self.res2(x)  # Features at 1/4 scale with 256 channels
        r3 = self.res3(r2)  # Features at 1/8 scale with 512 channels
        r4 = self.res4(r3)  # Features at 1/16 scale with 1024 channels

        # Return features from multiple levels of the network
        return r4, r3, r2, r1


class Refine(nn.Module):
    def __init__(self, inplanes, planes):
        super(Refine, self).__init__()
        # Convolution layer to transform the input feature map to the desired number of planes
        self.convFS = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, stride=1)
        # Residual blocks for feature refinement
        self.ResFS = ResBlock(planes, planes)
        self.ResMM = ResBlock(planes, planes)
        # Scaling factor for upsampling the previous prediction map
        self.scale_factor = 2

    def forward(self, f, pm):
        # Apply convolution and a residual block to the input feature map
        s = self.ResFS(self.convFS(f))
        # Upsample the previous prediction map and add it to the transformed feature map
        # Assuming s and pm are PyTorch tensors
        m = s + F.interpolate(pm, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        # Further refine the merged map with another residual block
        m = self.ResMM(m)

        return m


class Decoder(nn.Module):
    def __init__(self, device):
        super(Decoder, self).__init__()
        self.device = device
        # Define medium dimensions for global and local feature paths
        mdim_global = 256  # Dimensionality for global feature processing
        mdim_local = 32    # Dimensionality for local feature processing

        # Global processing path
        # Convolution layer to reduce feature depth from 1024 to mdim_global
        self.convFM = nn.Conv2d(1024, mdim_global, kernel_size=3, padding=1, stride=1)
        kaiming_normal_(self.convFM.weight, mode='fan_out', nonlinearity='relu')
        # Residual block for further processing of global features
        self.ResMM = ResBlock(mdim_global, mdim_global)
        # Refinement layers that progressively integrate and refine features from deeper layers
        self.RF3 = Refine(512, mdim_global)  # Refines features from r3 to global path
        self.RF2 = Refine(256, mdim_global)  # Refines features from r2 to global path
        # Final prediction layer for global features, outputting a single channel regression map
        self.pred_global = nn.Conv2d(mdim_global, 1, kernel_size=3, padding=1, stride=1)
        # Convolutional layer for processing concatenated features
        self.convGL = nn.Conv2d(2, 1, kernel_size=3, padding=1, stride=1)
        kaiming_normal_(self.convGL.weight, mode='fan_out', nonlinearity='relu')

        # Local processing path
        # Convolution layer to reduce feature depth from 256 (r1 features) to mdim_local
        self.local_convFM = nn.Conv2d(64, mdim_local, kernel_size=3, padding=1, stride=1)
        # Residual block for processing local features
        self.local_ResMM = ResBlock(mdim_local, mdim_local)
        # Final prediction layer for local features, also outputting a single channel regression map
        self.pred_local = nn.Conv2d(mdim_local, 1, kernel_size=3, padding=1, stride=1)

    def forward(self, r4, r3, r2, r1, feature_shape):
        bs, _, h, w = feature_shape  # Extract target height and width
        # Global path processing for regression prediction
        p_global = self.ResMM(self.convFM(r4))  # Process the r4 through global path
        p_global = self.RF3(r3, p_global)  # Integrate and refine with r3 features
        p_global = self.RF2(r2, p_global)  # Further refine with r2 features
        p_global = F.relu(p_global)  # Apply ReLU for non-linearity
        p_global = self.pred_global(p_global)  # Make global path regression prediction
        p_global = F.interpolate(p_global, scale_factor=2, mode='bilinear', align_corners=False)  # Upsample global prediction
        # (1,1,128,32)
        
        # Local path processing for detailed regression prediction
        p_local = self.local_ResMM(self.local_convFM(r1))  # Process r1 through local path
        p_local = F.relu(p_local)  # Apply ReLU for non-linearity
        p_local = self.pred_local(p_local)  # Make local path regression prediction

        # Upsample both global and local predictions to match input resolution
        p_global = F.interpolate(p_global, scale_factor=2, mode='bilinear', align_corners=False)  # Upsample global prediction
        p_local = F.interpolate(p_local, scale_factor=2, mode='bilinear', align_corners=False)  # Upsample local prediction
        
        # Concatenate global and local predictions along the channel dimension
        combined = torch.cat((p_global, p_local), dim=1)  # Ensure p_global and p_local are upsampled to the same size
        # Process the concatenated features to produce the final output
        p = self.convGL(combined)  # This layer reduces the channel dimensions and combines features

        return p


class AutoEncoder(nn.Module):
    def __init__(self, device):
        super(AutoEncoder, self).__init__()
        # Encoder for extracting multi-scale features from input
        self.encoder = Encoder()
        # Decoder for generating regression output from encoded features
        self.decoder = Decoder(device)

    def forward(self, x):
        # Pass input through encoder to get multi-scale features
        r4, r3, r2, r1 = self.encoder(x)
        # Use the size of the input to guide the upsampling in the decoder
        feature_shape = x.size()  # Capture input size for output matching
        # Pass encoded features and input size to decoder to generate regression output
        output = self.decoder(r4, r3, r2, r1, feature_shape)
        return output
