## Variational AutoEncoder with ResNet101

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet101, ResNet101_Weights
from torch.nn.init import kaiming_normal_


def match_and_add(tensor_a, tensor_b):
    """
    Upsample tensor_b to match the size of tensor_a and then add them together.
    This function assumes tensor_a is the target size.
    """
    height_a, width_a = tensor_a.size()[2], tensor_a.size()[3]
    tensor_b_upsampled = F.interpolate(tensor_b, size=(height_a, width_a), mode='bilinear', align_corners=False)
    return tensor_a + tensor_b_upsampled

class ResBlock(nn.Module):
    """
    A residual block that performs two convolutions followed by a residual connection.
    """
    def __init__(self, indim, outdim=None, stride=1):
        super(ResBlock, self).__init__()
        outdim = outdim or indim
        self.downsample = self._make_downsample(indim, outdim, stride)
        self.conv1 = nn.Conv2d(indim, outdim, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(outdim)
        self.conv2 = nn.Conv2d(outdim, outdim, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(outdim)
        self.prelu = nn.PReLU()  # PReLU initialization

    def _make_downsample(self, indim, outdim, stride):
        if indim != outdim or stride != 1:
            return nn.Sequential(
                nn.Conv2d(indim, outdim, 1, stride, bias=False),
                nn.BatchNorm2d(outdim)
            )

    def forward(self, x):
        identity = x
        out = self.prelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.prelu(out)
    
class Refine(nn.Module):
    def __init__(self, inplanes, planes):
        super(Refine, self).__init__()
        self.convFS = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1)
        self.ResFS = ResBlock(planes, planes)
        self.ResMM = ResBlock(planes, planes)

    def forward(self, f, pm):
        s = self.ResFS(self.convFS(f))
        m = match_and_add(s, F.interpolate(pm, scale_factor=2, mode='bilinear', align_corners=False))
        return self.ResMM(m)

class Encoder(nn.Module):
    def __init__(self, input_channels=4):
        super(Encoder, self).__init__()
        resnet = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=(3, 2), bias=False)
        kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')  # Adjusted for PReLU
        self.bn1 = resnet.bn1
        self.prelu = nn.PReLU()  # PReLU initialization
        self.maxpool = resnet.maxpool
        self.res2 = resnet.layer1
        self.res3 = resnet.layer2
        self.res4 = resnet.layer3
        self.res5 = resnet.layer4  # Additional layer in ResNet-101

        #self.register_buffer('max_values', torch.tensor([240.0, 201.6, 0.528, 1.870, 6.25]).view(1, 5, 1, 1))
        #self.register_buffer('min_values', torch.tensor([0.0, 0.0, -0.718, -1.589, -6.25]).view(1, 5, 1, 1))

        self.register_buffer('max_values', torch.tensor([240.0, 201.6, 0.528, 1.870]).view(1, 4, 1, 1))
        self.register_buffer('min_values', torch.tensor([0.0, 0.0, -0.718, -1.589]).view(1, 4, 1, 1))
        
        # Initialize linear layers correctly based on the actual feature map sizes
        self.fc_mu = nn.ModuleList([
            nn.Linear(2048*2*3, 64)  # Adjust dimensions for r5
        ])
        self.fc_logvar = nn.ModuleList([
            nn.Linear(2048*2*3, 64)
        ])

    def forward(self, in_f):
        f = (in_f - self.min_values) / (self.max_values - self.min_values)        
        x = self.maxpool(self.prelu(self.bn1(self.conv1(f))))  # Apply PReLU
        r2 = self.res2(x)
        r3 = self.res3(r2)
        r4 = self.res4(r3)
        r5 = self.res5(r4)  # Additional processing step
        
        mu = [self.fc_mu[0](r5.flatten(start_dim=1))]
        logvar = [self.fc_logvar[0](r5.flatten(start_dim=1))]
        
        return r5, r4, r3, r2, x, mu, logvar

class Decoder(nn.Module):
    def __init__(self, output_channels=1):
        super(Decoder, self).__init__()
        mdim_global, mdim_local = 256, 32
        self.convFM = nn.Conv2d(2048, mdim_global, kernel_size=3, stride=1, padding=1)
        kaiming_normal_(self.convFM.weight, mode='fan_out', nonlinearity='relu')  # Adjusted for PReLU
        self.ResMM = ResBlock(mdim_global, mdim_global)
        self.RF4 = Refine(1024, mdim_global)
        self.RF3 = Refine(512, mdim_global)
        self.RF2 = Refine(256, mdim_global)
        self.pred_global = nn.Conv2d(mdim_global, 1, kernel_size=3, stride=1, padding=1)
        self.local_convFM = nn.Conv2d(64, mdim_local, kernel_size=3, stride=1, padding=1)
        self.local_ResMM = ResBlock(mdim_local, mdim_local)
        self.pred_local = nn.Conv2d(mdim_local, 1, kernel_size=3, stride=1, padding=1)

        self.convGL = nn.Conv2d(2, output_channels, kernel_size=3, stride=1, padding=1)
        kaiming_normal_(self.convGL.weight, mode='fan_out', nonlinearity='relu')  # Adjusted for PReLU
        self.prelu = nn.PReLU()  # PReLU initialization for shared usage
        # for latent representation
        self.expand_z5 = nn.Linear(64, 2048*2*3)        

    def forward(self, r5, r4, r3, r2, r1, z5, feature_shape):
        bs, _, h, w = feature_shape
        
        z5 = self.expand_z5(z5).view(-1, 2048, 2, 3)
        r5z5 = match_and_add(r5, z5)
        global_features = self.convFM(r5z5)
        global_features = self.ResMM(global_features)
        global_features = self.RF4(r4, global_features)
        global_features = self.RF3(r3, global_features)
        global_features = self.RF2(r2, global_features)
        global_features = self.prelu(global_features)
        global_features = self.pred_global(global_features)
        p_global = F.interpolate(global_features, scale_factor=(4, 4), mode='bilinear', align_corners=False)
        
        local_features = self.local_convFM(r1)
        local_features = self.local_ResMM(local_features)
        local_features = self.prelu(local_features)
        local_features = self.pred_local(local_features)
        p_local = F.interpolate(local_features, scale_factor=(4, 4), mode='bilinear', align_corners=False)
        combined_features = torch.cat((p_global, p_local), dim=1)
        output = self.convGL(combined_features)
        
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
        std = [torch.exp(0.5 * lv) for lv in logvar]
        eps = [torch.randn_like(s) for s in std]
        z = [m + e * s for m, s, e in zip(mu, std, eps)]
        return z

    def forward(self, x):
        r5, r4, r3, r2, r1, mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(r5, r4, r3, r2, r1, *z, x.size()), mu, logvar  # Pass latent variables and features to the decoder
