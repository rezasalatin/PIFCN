## Variational AutoEncoder with Partial Convolutions for Image Hole Filling

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torch.nn.init import kaiming_normal_


def match_and_add(tensor_a, tensor_b):
    """
    Upsample tensor_b to match the size of tensor_a and then add them together.
    This function assumes tensor_a is the target size.
    """
    height_a, width_a = tensor_a.size()[2], tensor_a.size()[3]
    tensor_b_upsampled = F.interpolate(tensor_b, size=(height_a, width_a), mode='bilinear', align_corners=False)
    return tensor_a + tensor_b_upsampled

class PartialConv2d(nn.Module):
    """
    A residual block that performs two convolutions followed by a residual connection.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(PartialConv2d, self).__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        # Initialize weights for input_conv
        kaiming_normal_(self.input_conv.weight, mode='fan_out', nonlinearity='leaky_relu')

        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        # Initialize the mask convolution with a constant weight of 1 and bias of 0
        torch.nn.init.constant_(self.mask_conv.weight, 1.0)
        self.mask_conv.bias = None  
        # Mask convolution is NOT trainable
        for param in self.mask_conv.parameters():
            param.requires_grad = False

        # Store padding as an attribute
        self.padding = padding

    def forward(self, input, mask):

        # Extend mask for padding dynamically
        if self.padding != 0:
            input = F.pad(input, (self.padding, self.padding, self.padding, self.padding), mode='constant', value=0)
            mask = F.pad(mask, (self.padding, self.padding, self.padding, self.padding), mode='constant', value=0)

        # Apply mask to the input
        masked_input = input * mask
        print("hello 1")
        # Apply convolution to masked input and mask
        output = self.input_conv(masked_input)
        mask_output = self.mask_conv(mask)

        # Normalize output using mask
        with torch.no_grad():
            tmp = torch.ones_like(mask_output)  # Create a kernel with all elements equal to 1
            tmp_sum = self.mask_conv(tmp)  # Compute the sum of the kernel

            print("hello")
            # Print the size of tmp_sum
            print("Size of tmp_sum:", tmp_sum.size())
            
            # If you want to print only the first few elements, you can slice the tensor
            print("First few values of tmp_sum:", tmp_sum[:, :, :2, :2])

            mask_ratio = tmp_sum / (mask_output + 1e-8)
            mask_ratio[mask_output == 0] = 0.0

        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(output)
        else:
            output_bias = torch.zeros_like(output)
        # Apply mask ratio and add back the bias
        output = mask_ratio * (output - output_bias) + output_bias

        return output, mask_output

class PartialConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(PartialConvBlock, self).__init__()
        # Partial convolution layer
        self.pconv = PartialConv2d(in_channels, out_channels, kernel_size, stride, padding)
        # Batch normalization
        self.bn = nn.BatchNorm2d(out_channels)
        # PReLU activation
        self.prelu = nn.PReLU()

    def forward(self, x, mask):
        # Apply partial convolution
        x, mask = self.pconv(x, mask)
        # Normalize the outputs
        x = self.bn(x)
        # Apply the activation function
        x = self.prelu(x)
        return x, mask


class PartialResBlock(nn.Module):
    """
    A residual block that performs two partial convolutions followed by a residual connection.
    """
    def __init__(self, indim, outdim=None, stride=1, kernel_size=3, padding=1):
        super(PartialResBlock, self).__init__()
        outdim = outdim or indim
        self.conv1 = PartialConv2d(indim, outdim, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(outdim)
        self.conv2 = PartialConv2d(outdim, outdim, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm2d(outdim)
        self.prelu = nn.PReLU()

        if indim != outdim or stride != 1:
            self.downsample = nn.Sequential(
                PartialConv2d(indim, outdim, 1, stride, 0),
                nn.BatchNorm2d(outdim)
            )
        else:
            self.downsample = None

    def forward(self, x, mask):
        # Preserve the original data for the identity path
        identity = x
        identity_mask = mask 

        x, mask = self.conv1(x, mask)
        x = self.prelu(self.bn1(x))
        x, mask = self.conv2(x, mask)
        x = self.bn2(x)
        
        # Apply downsampling to identity if necessary
        if self.downsample is not None:
            identity, identity_mask = self.downsample(identity, identity_mask)
            identity, identity_mask = self.downsample_bn(identity, identity_mask)
            # Downsample the mask using max pooling to maintain coverage area
            identity_mask = F.max_pool2d(identity_mask, kernel_size=self.downsample.stride, stride=self.downsample.stride, padding=0)

        # Add the processed identity to the output of the residual blocks
        x += identity
        mask += identity_mask  # Combine masks logically

        return self.prelu(x), mask
    
class PartialRefine(nn.Module):
    def __init__(self, inplanes, planes):
        super(PartialRefine, self).__init__()
        self.convFS = PartialConv2d(inplanes, planes, kernel_size=3, stride=1, padding=1)
        self.ResFS = PartialResBlock(planes, planes)
        self.ResMM = PartialResBlock(planes, planes)

    def forward(self, f, pm, mask):
        f, mask = self.convFS(f, mask)
        s, mask = self.ResFS(f, mask)
        m = match_and_add(s, F.interpolate(pm, scale_factor=2, mode='bilinear', align_corners=False))
        m, mask = self.ResMM(m, mask)
        return m, mask
    
class Encoder(nn.Module):
    def __init__(self, input_channels=4):
        super(Encoder, self).__init__()
        self.conv1 = PartialConv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        kaiming_normal_(self.conv1.input_conv.weight, mode='fan_out', nonlinearity='leaky_relu')

        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.bn1 = resnet.bn1
        self.prelu = nn.PReLU()
        self.maxpool = resnet.maxpool

        # Residual blocks replaced with custom partial convolutions
        self.res2 = PartialResBlock(64, 256, stride=1)  # Custom implementation for layer1
        self.res3 = PartialResBlock(256, 512, stride=2)  # Custom implementation for layer2
        self.res4 = PartialResBlock(512, 1024, stride=2)  # Custom implementation for layer3
        
        # Input normalization
        self.register_buffer('max_values', torch.tensor([33.0, 13.0, 0.23, 0.09]).view(1, 4, 1, 1))
        self.register_buffer('min_values', torch.tensor([25.0, -13.0, -0.29, -0.11]).view(1, 4, 1, 1))
        
        # Initialize linear layers for latent space mapping
        self.fc_mu = nn.Linear(1024 * 4 * 5, 64)
        self.fc_logvar = nn.Linear(1024 * 4 * 5, 64)

    def forward(self, in_f, mask):
        f = (in_f - self.min_values) / (self.max_values - self.min_values)        
        x, mask = self.conv1(f, mask)
        x = self.prelu(self.bn1(x))
        x = self.maxpool(x)  # Maxpool may need adaptation for partial convolutions or mask handling
        mask = self.maxpool(mask.float())
        mask = (mask > 0).float()  # Convert back to binary mask

        r2, mask = self.res2(x, mask)
        r3, mask = self.res3(r2, mask)
        r4, mask = self.res4(r3, mask)
        
        mu = self.fc_mu(r4.flatten(start_dim=1))
        logvar = self.fc_logvar(r4.flatten(start_dim=1))
        
        return r4, r3, r2, x, mu, logvar, mask

class Decoder(nn.Module):
    def __init__(self, output_channels=1):
        super(Decoder, self).__init__()
        mdim_global, mdim_local = 256, 32

        self.convFM = PartialConv2d(1024, mdim_global, kernel_size=3, stride=1, padding=1)
        kaiming_normal_(self.convFM.input_conv.weight, mode='fan_out', nonlinearity='leaky_relu')

        self.ResMM = PartialResBlock(mdim_global, mdim_global)
        self.RF3 = PartialResBlock(512, mdim_global)
        self.RF2 = PartialResBlock(256, mdim_global)
        self.pred_global = PartialResBlock(mdim_global, output_channels, kernel_size=3, stride=1, padding=1)
        
        self.local_convFM = PartialResBlock(64, mdim_local, kernel_size=3, stride=1, padding=1)
        self.local_ResMM = PartialResBlock(mdim_local, mdim_local)
        self.pred_local = PartialResBlock(mdim_local, output_channels, kernel_size=3, stride=1, padding=1)

        self.convGL = PartialConv2d(2 * output_channels, output_channels, kernel_size=3, stride=1, padding=1)
        kaiming_normal_(self.convGL.input_conv.weight, mode='fan_out', nonlinearity='leaky_relu')

        self.prelu = nn.PReLU()  # PReLU initialization for shared usage

        # for latent representation
        self.expand_z4 = nn.Linear(64, 1024*4*5)        

    def forward(self, r4, r3, r2, r1, z4, feature_shape, mask):
        bs, _, h, w = feature_shape
        
        # Process global features
        z4 = self.expand_z4(z4).view(-1, 1024, 4, 5)
        r4z4 = match_and_add(r4, z4)
        global_features, mask = self.convFM(r4z4, mask)
        global_features, mask = self.ResMM(global_features, mask)
        
        # Process features from lower layers and refine with global features
        global_features, mask = self.RF3(r3, global_features, mask)
        global_features, mask = self.RF2(r2, global_features, mask)
        global_features = self.prelu(global_features)
        global_features = self.pred_global(global_features, mask)
        p_global = F.interpolate(global_features, scale_factor=(4, 4), mode='bilinear', align_corners=False)
        
        # Local features handling
        local_features, mask = self.local_convFM(r1, mask)
        local_features, mask = self.local_ResMM(local_features, mask)
        local_features = self.prelu(local_features)
        local_features = self.pred_local(local_features, mask)
        p_local = F.interpolate(local_features, scale_factor=(4, 4), mode='bilinear', align_corners=False)
        
        # Combine global and local features
        combined_features = torch.cat((p_global, p_local), dim=1)
        output, mask = self.convGL(combined_features, mask)
        
        return output

class VAE(nn.Module):
    """
    Autoencoder that encodes input features and decodes them to generate regression output.
    """
    def __init__(self, input_channels=4, output_channels=1):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_channels=input_channels)
        self.decoder = Decoder(output_channels=output_channels)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, mask):
        # Replace NaNs in x with zeros (or any other appropriate fill value)
        x = x.clone()  # Avoid modifying the original input directly
        x[mask == 0] = 0

        # Encode the input along with the mask
        r4, r3, r2, r1, mu, logvar, mask = self.encoder(x, mask)
        z = self.reparameterize(mu, logvar)

        # Decode the encoded features along with the updated mask
        output = self.decoder(r4, r3, r2, r1, z, x.size(), mask)
        return output, mu, logvar