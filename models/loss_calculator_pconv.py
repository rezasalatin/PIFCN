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
            resnet.layer1,  # Similar to VGG's pool1
            resnet.layer2,  # Similar to VGG's pool2
            resnet.layer3,  # Similar to VGG's pool3
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

    def dilate_mask(mask, dilation=1):
        """Dilate the mask by one pixel in each direction using a 3x3 kernel."""
        if mask.dtype != torch.float32:
            mask = mask.float()
        kernel = torch.ones((1, 1, 3, 3), device=mask.device)
        dilated_mask = F.conv2d(mask.unsqueeze(1), kernel, padding=dilation) > 0
        return dilated_mask.squeeze(1)

    def hole_and_valid_losses(self, generated, target, mask):
        """Calculate per-pixel L1 losses for hole and valid areas."""
        hole_loss = F.l1_loss((1 - mask) * generated, (1 - mask) * target, reduction='sum')
        valid_loss = F.l1_loss(mask * generated, mask * target, reduction='sum')
        return hole_loss, valid_loss

    def perceptual_loss(self, generated, target, mask):
        comp = self.create_comp_image(generated, target, mask)
        gen_features = self.feature_extractor(generated)
        target_features = self.feature_extractor(target)
        comp_features = self.feature_extractor(comp)
        perceptual_l = 0
        for gen_f, target_f, comp_f in zip(gen_features, target_features, comp_features):
            perceptual_l += F.l1_loss(gen_f, target_f) + F.l1_loss(comp_f, target_f)
        return perceptual_l

    def gram_matrix(self, input):
        a, b, c, d = input.size()
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)

    def style_loss(self, output, composite, target):
        output_features = self.feature_extractor(output)
        composite_features = self.feature_extractor(composite)
        target_features = self.feature_extractor(target)

        style_loss_output = 0
        style_loss_composite = 0
        for out_f, comp_f, tar_f in zip(output_features, composite_features, target_features):
            G_out = self.gram_matrix(out_f)
            G_comp = self.gram_matrix(comp_f)
            G_tar = self.gram_matrix(tar_f)

            style_loss_output += F.l1_loss(G_out, G_tar)
            style_loss_composite += F.l1_loss(G_comp, G_tar)

        return style_loss_output, style_loss_composite
    
    def total_variation_loss(self, composite, mask):
        """Calculate TV loss specifically within the dilated hole region."""
        # Dilate the mask to define the region P
        dilated_mask = self.dilate_mask(mask)
        
        # Calculate TV loss only within this dilated region
        tv_loss = 0
        vertical_diff = (composite[:, :, 1:, :] - composite[:, :, :-1, :]) * dilated_mask[:, :, 1:, :]
        horizontal_diff = (composite[:, :, :, 1:] - composite[:, :, :, :-1]) * dilated_mask[:, :, :, 1:]
        
        tv_loss += torch.sum(torch.abs(vertical_diff))
        tv_loss += torch.sum(torch.abs(horizontal_diff))
        
        return tv_loss / composite.numel()

    
    def create_comp_image(self, generated, target, mask):
        """Creates the composite image for perceptual loss calculation."""
        comp = generated * mask + target * (1 - mask)
        return comp

    def combined_loss(self, target, generated, mask):
        composite = self.create_comp_image(generated, target, mask)
        hole_l, valid_l = self.hole_and_valid_losses(generated, target, mask)
        perc_l = self.perceptual_loss(generated, target, mask)
        style_l_out, style_l_comp = self.style_loss(generated, composite, target)
        tv_l = self.total_variation_loss(generated, mask)
        # Adjust weights as necessary based on their importance to the task
        combined_l = valid_l + 6 * hole_l + 0.05 * perc_l + 120 * (style_l_out+style_l_comp) + 0.1 * tv_l
        return combined_l