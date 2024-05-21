import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet101, ResNet101_Weights
from torch.nn.init import kaiming_normal_

def match_and_add(tensor_a, tensor_b):
    # Upsample and add
    height_a, width_a = tensor_a.size()[2], tensor_a.size()[3]
    tensor_b_upsampled = F.interpolate(tensor_b, size=(height_a, width_a), mode='bicubic', align_corners=False)
    return tensor_a + tensor_b_upsampled

class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None, stride=1):
        super(ResBlock, self).__init__()
        outdim = outdim or indim
        self.downsample = self._make_downsample(indim, outdim, stride)
        self.conv1 = nn.Conv2d(indim, outdim, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(outdim)
        self.conv2 = nn.Conv2d(outdim, outdim, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(outdim)
        self.prelu = nn.PReLU()

    def _make_downsample(self, indim, outdim, stride):
        if indim != outdim or stride != 1:
            return nn.Sequential(
                nn.Conv2d(indim, outdim, 1, stride, bias=False),
                nn.BatchNorm2d(outdim)
            )
        return None

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
        m = match_and_add(s, F.interpolate(pm, scale_factor=2, mode='bicubic', align_corners=False))
        return self.ResMM(m)

class Encoder(nn.Module):
    def __init__(self, input_channels):
        super(Encoder, self).__init__()
        resnet = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.bn1 = resnet.bn1
        self.prelu = nn.PReLU()
        #self.maxpool = resnet.maxpool
        self.maxpool = nn.AdaptiveMaxPool2d((32, 32))
        self.res2 = resnet.layer1
        self.res3 = resnet.layer2
        self.res4 = resnet.layer3
        self.res5 = resnet.layer4

        max_vals = torch.tensor([240.0, 201.6, 0.528, 1.870, 6.25])
        min_vals = torch.tensor([0.0, 0.0, -0.718, -1.589, -6.25])
        self.register_buffer('max_values', max_vals[:input_channels].view(1, input_channels, 1, 1))
        self.register_buffer('min_values', min_vals[:input_channels].view(1, input_channels, 1, 1))

        self.fc_mu = None
        self.fc_logvar = None

    def forward(self, in_f):
        original_height, original_width = in_f.size(2), in_f.size(3)
        f = (in_f - self.min_values) / (self.max_values - self.min_values)
        x = self.maxpool(self.prelu(self.bn1(self.conv1(f))))
        r2 = self.res2(x)
        r3 = self.res3(r2)
        r4 = self.res4(r3)
        r5 = self.res5(r4)

        flat_features = r5.flatten(start_dim=1)

        if self.fc_mu is None or self.fc_logvar is None:
            self.fc_mu = nn.Linear(2048 * 4 * 4, 64).to(r5.device)
            self.fc_logvar = nn.Linear(2048 * 4 * 4, 64).to(r5.device)

        mu = self.fc_mu(flat_features)
        logvar = self.fc_logvar(flat_features)

        return r5, r4, r3, r2, x, mu, logvar, original_height, original_width


class Decoder(nn.Module):
    def __init__(self, output_channels):
        super(Decoder, self).__init__()
        mdim_global, mdim_local = 256, 32
        self.convFM = nn.Conv2d(2048, mdim_global, kernel_size=3, stride=1, padding=1)
        kaiming_normal_(self.convFM.weight, mode='fan_out', nonlinearity='relu')
        self.ResMM = ResBlock(mdim_global, mdim_global)
        self.RF4 = Refine(1024, mdim_global)
        self.RF3 = Refine(512, mdim_global)
        self.RF2 = Refine(256, mdim_global)
        self.pred_global = nn.Conv2d(mdim_global, 1, kernel_size=3, stride=1, padding=1)

        self.local_convFM = nn.Conv2d(64, mdim_local, kernel_size=3, stride=1, padding=1)
        self.local_ResMM = ResBlock(mdim_local, mdim_local)
        self.pred_local = nn.Conv2d(mdim_local, 1, kernel_size=3, stride=1, padding=1)

        self.convGL = nn.Conv2d(2, output_channels, kernel_size=3, stride=1, padding=1)
        kaiming_normal_(self.convGL.weight, mode='fan_out', nonlinearity='relu')
        self.prelu = nn.PReLU()
        self.expand_z5 = None

    def initialize(self, device):
        self.expand_z5 = nn.Linear(64, 2048 * 4 * 4).to(device)

    def forward(self, r5, r4, r3, r2, r1, z5, original_height, original_width):
        z5 = self.expand_z5(z5).view(-1, 2048, 4, 4)
        r5z5 = match_and_add(r5, z5)
        global_features = self.convFM(r5z5)
        global_features = self.ResMM(global_features)
        global_features = self.RF4(r4, global_features)
        global_features = self.RF3(r3, global_features)
        global_features = self.RF2(r2, global_features)
        global_features = self.prelu(global_features)
        global_features = self.pred_global(global_features)
        
        p_global = F.interpolate(global_features, size=(original_height, original_width), mode='bicubic', align_corners=False)

        local_features = self.local_convFM(r1)
        local_features = self.local_ResMM(local_features)
        local_features = self.prelu(local_features)
        local_features = self.pred_local(local_features)
        p_local = F.interpolate(local_features, size=(original_height, original_width), mode='bicubic', align_corners=False)

        combined_features = torch.cat((p_global, p_local), dim=1)
        output = self.convGL(combined_features)

        return output

class VAE(nn.Module):
    def __init__(self, input_channels=4, output_channels=1):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_channels=input_channels)
        self.decoder_initialized = False
        self.output_channels = output_channels

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def initialize_decoder(self, x):
        self.decoder = Decoder(output_channels=self.output_channels).to(x.device)
        self.decoder.initialize(x.device)
        self.decoder_initialized = True

    def forward(self, x):
        r5, r4, r3, r2, r1, mu, logvar, original_height, original_width = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        if not self.decoder_initialized:
            self.initialize_decoder(x)
        return self.decoder(r5, r4, r3, r2, r1, z, original_height, original_width), mu, logvar
