"""Conta os parametros e as operacoes (GFLOPs) dos tres modelos, com 5 classes, no tamanho de
entrada que cada um usa para uma imagem do BDD100K (1280 x 720).

- YOLO26m:          640 x 384 (mantem a proporcao; lado menor completado ate multiplo de 32)
- RT-DETR-L:        640 x 640 (redimensionamento direto para quadrado)
- Faster R-CNN v2:  640 x 360 (mantem a proporcao)

GFLOPs = 2 x multiplicacoes-acumulacoes contadas pelo thop (mesma convencao da Ultralytics).
Nao precisa dos pesos treinados: o numero de parametros e de operacoes depende so da arquitetura.

Uso (com o .venv ativo, dentro de C:\\tcc):  python contagem_operacoes.py
"""
import copy

import thop
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from ultralytics.nn.tasks import DetectionModel, RTDETRDetectionModel


def gflops(modelo, altura, largura):
    m = copy.deepcopy(modelo).eval()
    with torch.no_grad():
        macs = thop.profile(m, inputs=(torch.zeros(1, 3, altura, largura),), verbose=False)[0]
    return 2 * macs / 1e9


class SoFRCNN(torch.nn.Module):
    """Embrulho para o thop: o Faster R-CNN recebe uma lista de imagens."""
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f([x[0]])


yolo = DetectionModel("yolo26m.yaml", nc=5, verbose=False)
rt = RTDETRDetectionModel("rtdetr-l.yaml", nc=5, verbose=False)
fr = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, num_classes=6,
                                min_size=360, max_size=640).eval()

linhas = [("YOLO26m", yolo, 384, 640), ("RT-DETR-L", rt, 640, 640), ("Faster R-CNN v2", SoFRCNN(fr), 360, 640)]
print(f"{'modelo':>16}{'parametros (M)':>16}{'entrada':>12}{'GFLOPs':>9}")
for nome, m, h, w in linhas:
    n = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"{nome:>16}{n:>16.1f}{f'{w}x{h}':>12}{gflops(m, h, w):>9.1f}")
