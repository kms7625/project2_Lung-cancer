# [목적] 2D ResNet-18 구조를 3D로 확장한 모델이다.
# Conv2D → Conv3D, BatchNorm2D → BatchNorm3D, MaxPool2D → MaxPool3D로 변경.
# 입력: (Batch, 1, 64, 64, 64) — 결절 ROI 큐브
# 출력: (Batch, 2) — 양성/악성 이진 분류

import torch
import torch.nn as nn

class BasicBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm3d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(out_ch)

        # stride가 다르거나 채널 수가 다를 때 shortcut 연결을 맞춰줌
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class ResNet3D(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # 입력 채널 1 (grayscale CT)
        self.conv1   = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1     = nn.BatchNorm3d(64)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64,  64,  2, stride=1)
        self.layer2 = self._make_layer(64,  128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc      = nn.Linear(512, num_classes)

    def _make_layer(self, in_ch, out_ch, num_blocks, stride):
        layers = [BasicBlock3D(in_ch, out_ch, stride)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock3D(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


if __name__ == '__main__':
    # 모델 구조 확인
    model = ResNet3D(num_classes=2)
    x = torch.randn(2, 1, 64, 64, 64)  # 배치 2개
    out = model(x)
    print(f'입력 shape: {x.shape}')
    print(f'출력 shape: {out.shape}')
    n_params = sum(p.numel() for p in model.parameters())
    print(f'파라미터 수: {n_params:,}')