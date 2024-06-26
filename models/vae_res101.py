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
    
# Channel attention mechanism
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_planes, in_planes//ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes//ratio, in_planes, 1, bias=False)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

# Spatial attention mechanism
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        x = torch.cat([max_out, avg_out], dim=1)
        x = self.conv1(x)
        return torch.sigmoid(x)

# Combined channel and spatial attention mechanism
class CombinedAttention(nn.Module):
    def __init__(self, in_planes, ratio):
        super(CombinedAttention, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention()
        self.downsample = nn.AdaptiveAvgPool2d((56, 56))

    def forward(self, x):
        x = self.downsample(x) * self.channel_attention(x)
        x = self.downsample(x) * self.spatial_attention(x)
        # = self.downsample(x)
        return x

class Encoder(nn.Module):
    def __init__(self, input_channels, device):
        super(Encoder, self).__init__()
        self.device = device
        resnet = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1).to(device)
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=1, padding=3, bias=False).to(device)
        kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.bn1 = resnet.bn1.to(device)
        self.prelu = nn.PReLU().to(device)
        #self.downsample = nn.AdaptiveAvgPool2d((56, 56))
        self.downsample = nn.AdaptiveMaxPool2d((56, 56))
        self.res2 = resnet.layer1.to(device)
        self.res3 = resnet.layer2.to(device)
        self.res4 = resnet.layer3.to(device)
        self.res5 = resnet.layer4.to(device)

        max_vals = torch.tensor([240.0, 201.6, 0.528, 1.870, 6.25]).to(device)
        min_vals = torch.tensor([0.0, 0.0, -0.718, -1.589, -6.25]).to(device)
        self.register_buffer('max_values', max_vals[:input_channels].view(1, input_channels, 1, 1))
        self.register_buffer('min_values', min_vals[:input_channels].view(1, input_channels, 1, 1))

        self.fc_mu = nn.Linear(2048 * 7 * 7, 64).to(device)
        self.fc_logvar = nn.Linear(2048 * 7 * 7, 64).to(device)

        self.conv_skip = nn.Conv2d(input_channels, 64, kernel_size=1, stride=1, bias=False).to(device)
        self.bn_skip = nn.BatchNorm2d(64).to(device)
    
    def forward(self, in_f):
        in_f = in_f.to(self.device)
        original_height, original_width = in_f.size(2), in_f.size(3)
        f = (in_f - self.min_values) / (self.max_values - self.min_values)
        x = self.prelu(self.bn1(self.conv1(f)))
        x = self.downsample(x)
        skip = self.bn_skip(self.conv_skip(f))
        skip = self.downsample(skip)
        x = x + skip

        r2 = self.res2(x)
        r3 = self.res3(r2)
        r4 = self.res4(r3)
        r5 = self.res5(r4)

        flat_features = r5.flatten(start_dim=1)
        mu = self.fc_mu(flat_features)
        logvar = self.fc_logvar(flat_features)

        return r5, r4, r3, r2, x, mu, logvar, original_height, original_width

class Decoder(nn.Module):
    def __init__(self, output_channels, device):
        super(Decoder, self).__init__()
        self.device = device
        mdim_global, mdim_local = 256, 32
        self.convFM = nn.Conv2d(2048, mdim_global, kernel_size=3, stride=1, padding=1).to(device)
        kaiming_normal_(self.convFM.weight, mode='fan_out', nonlinearity='relu')
        self.ResMM = ResBlock(mdim_global, mdim_global).to(device)
        self.RF4 = Refine(1024, mdim_global).to(device)
        self.RF3 = Refine(512, mdim_global).to(device)
        self.RF2 = Refine(256, mdim_global).to(device)
        self.pred_global = nn.Conv2d(mdim_global, 1, kernel_size=3, stride=1, padding=1).to(device)

        self.local_convFM = nn.Conv2d(64, mdim_local, kernel_size=3, stride=1, padding=1).to(device)
        self.local_ResMM = ResBlock(mdim_local, mdim_local).to(device)
        self.pred_local = nn.Conv2d(mdim_local, 1, kernel_size=3, stride=1, padding=1).to(device)

        self.convGL = nn.Conv2d(2, output_channels, kernel_size=3, stride=1, padding=1).to(device)
        kaiming_normal_(self.convGL.weight, mode='fan_out', nonlinearity='relu')
        self.prelu = nn.PReLU().to(device)
        self.expand_z5 = nn.Linear(64, 2048 * 7 * 7).to(device)

    def forward(self, r5, r4, r3, r2, r1, z5, original_height, original_width):
        z5 = self.expand_z5(z5).view(-1, 2048, 7, 7).to(self.device)
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
    def __init__(self, input_channels=4, output_channels=1, device='cuda'):
        super(VAE, self).__init__()
        self.device = device
        self.encoder = Encoder(input_channels=input_channels, device=device)
        self.decoder_initialized = False
        self.output_channels = output_channels
        self.decoder = None

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(self.device)
        z = mu + eps * std
        return z

    def initialize_decoder(self, x):
        self.decoder = Decoder(output_channels=self.output_channels, device=self.device).to(self.device)
        self.decoder_initialized = True

    def forward(self, x):
        r5, r4, r3, r2, r1, mu, logvar, original_height, original_width = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        if not self.decoder_initialized:
            self.initialize_decoder(x)
        return self.decoder(r5, r4, r3, r2, r1, z, original_height, original_width), mu, logvar