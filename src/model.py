"""CSRNet (Li i in., CVPR 2018) - model mapy gęstości.

Architektura:
- frontend: pierwsze 10 warstw konwolucyjnych VGG16 (do conv4_3) z wagami ImageNet;
  trzy max-pooling zmniejszają obraz 8 razy,
- backend: 6 konwolucji 3x3 z dylatacją 2 (większe pole widzenia bez dalszego zmniejszania),
- wyjście: konwolucja 1x1 -> 1 kanał (mapa gęstości), bez funkcji aktywacji.

Wejście ``(B, 3, H, W)`` (H, W podzielne przez 8) -> wyjście ``(B, 1, H/8, W/8)``.
Przewidywana liczba obiektów = suma mapy wyjściowej (po podzieleniu przez ``density_scale``
użyte w treningu).
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import VGG16_Weights, vgg16

BACKEND_CHANNELS = (512, 512, 512, 256, 128, 64)


class CSRNet(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        # features[:23] = conv1_1 ... conv4_3 + ReLU (10 konwolucji, 3 max-pool)
        self.frontend = vgg16(weights=weights).features[:23]

        layers: list[nn.Module] = []
        in_ch = 512
        for out_ch in BACKEND_CHANNELS:
            layers += [nn.Conv2d(in_ch, out_ch, 3, padding=2, dilation=2), nn.ReLU(inplace=True)]
            in_ch = out_ch
        self.backend = nn.Sequential(*layers)
        self.output = nn.Conv2d(in_ch, 1, 1)

        # backend i wyjście: inicjalizacja jak w oryginalnym CSRNet (N(0, 0.01), bias 0)
        for m in [*self.backend.modules(), self.output]:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.backend(self.frontend(x)))
