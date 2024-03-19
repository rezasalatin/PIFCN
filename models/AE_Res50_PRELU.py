import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50
from torch.nn.init import kaiming_normal_

class ResBlock(nn.Module):
    """
    A residual block that performs two convolutions followed by a residual connection,
    using PReLU instead of Leaky ReLU.
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

class Encoder(nn.Module):
    """
    Encoder module using PReLU instead of Leaky ReLU for the initial convolution.
    """
    def __init__(self):
        super(Encoder, self).__init__()
        resnet = resnet50(pretrained=False)
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')  # Adjusted for PReLU
        self.bn1 = resnet.bn1
        self.prelu = nn.PReLU()  # PReLU initialization
        self.maxpool = resnet.maxpool
        self.res2 = resnet.layer1
        self.res3 = resnet.layer2
        self.res4 = resnet.layer3
        self.register_buffer('max_values', torch.tensor([33.0, 13.0, 0.23, 0.09]).view(1, 4, 1, 1))
        self.register_buffer('min_values', torch.tensor([25.0, -13.0, -0.29, -0.11]).view(1, 4, 1, 1))

    def forward(self, in_f):
        f = (in_f - self.min_values) / (self.max_values - self.min_values)
        x = self.maxpool(self.prelu(self.bn1(self.conv1(f))))  # Apply PReLU
        return self.res4(self.res3(self.res2(x))), self.res3(self.res2(x)), self.res2(x), x

class Refine(nn.Module):
    """
    Refinement module that upsamples and refines features using convolution and residual blocks.
    
    Parameters:
    - inplanes (int): Number of input channels.
    - planes (int): Number of output channels.
    """
    def __init__(self, inplanes, planes):
        super(Refine, self).__init__()
        self.convFS = nn.Conv2d(inplanes, planes, 3, 1, 1)
        self.ResFS = ResBlock(planes, planes)
        self.ResMM = ResBlock(planes, planes)

    def forward(self, f, pm):
        s = self.ResFS(self.convFS(f))
        m = s + F.interpolate(pm, scale_factor=2, mode='bilinear', align_corners=False)
        return self.ResMM(m)

class Decoder(nn.Module):
    """
    Decoder module using PReLU instead of Leaky ReLU.
    """
    def __init__(self):
        super(Decoder, self).__init__()
        mdim_global, mdim_local = 256, 32
        self.convFM = nn.Conv2d(1024, mdim_global, 3, 1, 1)
        kaiming_normal_(self.convFM.weight, mode='fan_out', nonlinearity='relu')  # Adjusted for PReLU
        self.ResMM = ResBlock(mdim_global, mdim_global)
        self.RF3 = Refine(512, mdim_global)
        self.RF2 = Refine(256, mdim_global)
        self.pred_global = nn.Conv2d(mdim_global, 1, 3, 1, 1)
        self.convGL = nn.Conv2d(2, 1, 3, 1, 1)
        kaiming_normal_(self.convGL.weight, mode='fan_out', nonlinearity='relu')  # Adjusted for PReLU
        self.local_convFM = nn.Conv2d(64, mdim_local, 3, 1, 1)
        self.local_ResMM = ResBlock(mdim_local, mdim_local)
        self.pred_local = nn.Conv2d(mdim_local, 1, 3, 1, 1)
        self.prelu = nn.PReLU()  # PReLU initialization for shared usage

    def forward(self, r4, r3, r2, r1, feature_shape):
        bs, _, h, w = feature_shape
        p_global = F.interpolate(self.pred_global(self.prelu(self.RF2(r2, self.RF3(r3, self.ResMM(self.convFM(r4)))))), scale_factor=4, mode='bilinear', align_corners=False)
        p_local = F.interpolate(self.pred_local(self.prelu(self.local_ResMM(self.local_convFM(r1)))), scale_factor=4, mode='bilinear', align_corners=False)
        return self.convGL(torch.cat((p_global, p_local), dim=1))

class AutoEncoder(nn.Module):
    """
    Autoencoder that encodes input features and decodes them to generate regression output.
    """
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        return self.decoder(*self.encoder(x), x.size())
