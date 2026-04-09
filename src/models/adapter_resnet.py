import torch
import torch.nn as nn

class ResidualAdapter(nn.Module):
    def __init__(self, in_channels, reduction=32):
        super().__init__()
        bottleneck = in_channels // reduction
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        return self.adapter(x)

class AdapterBlock(nn.Module):
    def __init__(self, block, channels):
        super().__init__()
        self.block = block
        self.bn = nn.BatchNorm2d(channels)
        self.adapter = ResidualAdapter(channels)

    def forward(self, x):
        block_out = self.block(x)
        normed = self.bn(block_out)
        return block_out + self.adapter(normed)
    
class ResNetWithAdapters(nn.Module):
    def __init__(self, base, domain_list):
        super().__init__()

        # Shared stem
        self.stem = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool
        )
        self.base_layers = nn.ModuleDict({
            'layer1': base.layer1,
            'layer2': base.layer2,
            'layer3': base.layer3,
            'layer4': base.layer4
        })
        
        # Shared layers (no adapters)
        self.layer1 = base.layer1
        self.layer2 = base.layer2

        # Domain-specific adapters only for last 2 layers
        self.adapters = nn.ModuleDict({
            domain: nn.ModuleDict({
                'layer3': self._wrap_with_adapters(base.layer3, 1024),
                'layer4': self._wrap_with_adapters(base.layer4, 2048)
            })
            for domain in domain_list
        })

        self.avgpool = base.avgpool

    def _wrap_with_adapters(self, layer, channels):
        return nn.Sequential(
            *[AdapterBlock(block, channels) for block in layer]
        )

    def forward(self, x, domain):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.adapters[domain]['layer3'](x)
        x = self.adapters[domain]['layer4'](x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # Normalize embeddings for better distance calculations in few-shot
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        return x
