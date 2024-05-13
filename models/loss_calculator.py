import torch
from torchvision.models import resnet50
import torch.nn.functional as F

class ResNet50Features(torch.nn.Module):
    def __init__(self):
        super(ResNet50Features, self).__init__()
        # Load a pretrained ResNet50 model
        resnet = resnet50(pretrained=True)
        # Extract layers from ResNet50 for feature extraction
        self.features = torch.nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        # Disable training for these parameters
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        # Collect intermediate layers
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {4, 5, 6, 7}:  # Collect outputs after each layer block
                results.append(x)
        return results
    
class LossCalculator():
    def __init__(self, device):
        self.feature_extractor = ResNet50Features().to(device)

    def perceptual_loss(generated, target, feature_extractor):
        gen_features = feature_extractor(generated)
        target_features = feature_extractor(target)
        perceptual_l = 0
        for gen_f, target_f in zip(gen_features, target_features):
            perceptual_l += F.l1_loss(gen_f, target_f)
        return perceptual_l

    def gram_matrix(input):
        a, b, c, d = input.size()
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)

    def style_loss(generated, target, feature_extractor):
        gen_features = feature_extractor(generated)
        target_features = feature_extractor(target)
        style_l = 0
        for gen_f, target_f in zip(gen_features, target_features):
            gm_gen = gram_matrix(gen_f)
            gm_target = gram_matrix(target_f)
            style_l += F.l1_loss(gm_gen, gm_target)
        return style_l