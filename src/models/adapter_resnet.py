import torch
import torch.nn as nn
import copy

class ResidualAdapter(nn.Module):
    def __init__(self, channels, reduction=32):
        super().__init__()
        bottleneck = max(channels // reduction, 1)

        self.adapter = nn.Sequential(
            nn.Conv2d(channels, bottleneck, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels)
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
        out = self.block(x)
        return out + self.adapter(self.bn(out))


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

        # Shared early layers
        self.layer1 = base.layer1   # 256
        self.layer2 = base.layer2   # 512

        # Domain-specific layers (deep layers only)
        self.adapters = nn.ModuleDict({
            domain: nn.ModuleDict({
                'layer3': self._make_adapter_layer(base.layer3, 1024),
                'layer4': self._make_adapter_layer(base.layer4, 2048)
            })
            for domain in domain_list
        })

        self.avgpool = base.avgpool

    def _make_adapter_layer(self, layer, channels):
        # Deep copy so each domain has independent weights
        return nn.Sequential(*[
            AdapterBlock(copy.deepcopy(block), channels)
            for block in layer
        ])

    def forward(self, x, domain):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.adapters[domain]['layer3'](x)
        x = self.adapters[domain]['layer4'](x)

        # IMPORTANT: keep spatial features (for segmentation / CD later)
        return x