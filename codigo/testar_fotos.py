"""Roda os tres modelos treinados em fotos proprias e salva as imagens com as caixas desenhadas.

Serve como PILOTO do ensaio de campo: ver se os modelos detectam pessoas, carros e placas em
fotos tiradas com o seu celular, e a partir de que tamanho (em pixels) os objetos deixam de ser
detectados. NAO e uma metrica (nao ha anotacao); e uma verificacao qualitativa.

Uso (venv ativado, na pasta C:\\tcc, com treinar_frcnn.py na mesma pasta):
    python testar_fotos.py --pasta C:\\tcc\\fotos_piloto
    python testar_fotos.py --pasta C:\\tcc\\fotos_piloto --conf 0.25
    python testar_fotos.py --pasta fotos --conf-rtdetr 0.45 --conf-yolo 0.30 --conf-frcnn 0.50

Sem GPU funciona (mais devagar: alguns segundos por foto).
Fotos .jpg, .jpeg, .png, .bmp ou .webp. Fotos .HEIC (iPhone) precisam ser convertidas para .jpg.

Saida em C:\\tcc\\teste_fotos\\ (ou --saida):
    <foto>_comparacao.jpg   as tres versoes da foto, uma embaixo da outra
    <foto>_<modelo>.jpg     a foto com as caixas de cada modelo
    deteccoes.csv           uma linha por deteccao: classe, confianca, caixa e tamanho em pixels
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import RTDETR, YOLO

import treinar_frcnn as F

EXTENSOES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CORES = [(0, 200, 255), (255, 120, 0), (0, 160, 0), (60, 60, 230), (200, 0, 200),
         (0, 220, 220), (128, 128, 0)]  # BGR, uma por classe


def modelo_ultralytics(classe, caminho, conf, imgsz, dev):
    m = classe(str(caminho))
    nomes = [m.names[k] for k in sorted(m.names)]

    def prever(im):
        r = m.predict(im, imgsz=imgsz, conf=conf, max_det=300, device=dev, verbose=False)[0]
        b = r.boxes
        return (b.xyxy.float().cpu().numpy(), b.conf.float().cpu().numpy(),
                b.cls.cpu().numpy().astype(int))
    return prever, nomes


def modelo_frcnn(caminho, conf, dev, imgsz=None):
    ck = torch.load(caminho, map_location="cpu", weights_only=False)
    m, _ = F.criar_modelo(ck["arq"], ck["imgsz"], ck["nomes"], pesos=False)  # pesos vem do arquivo
    m.load_state_dict(ck["modelo"])
    m = m.to(dev).eval()
    m.roi_heads.score_thresh = conf
    lado = imgsz or ck["imgsz"]  # lado maior da imagem na entrada da rede (treino: 640)
    m.transform.min_size, m.transform.max_size = (lado,), lado

    @torch.no_grad()
    def prever(im):  # mesmo pre-processamento usado na medicao de latencia
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        s = lado / max(h, w)
        rgb = cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(rgb).permute(2, 0, 1).to(dev).float().div_(255)
        o = m([t])[0]
        return ((o["boxes"].float() / s).cpu().numpy(), o["scores"].float().cpu().numpy(),
                (o["labels"].cpu() - 1).numpy().astype(int))
    return prever, list(ck["nomes"])


def desenhar(im, caixas, confs, classes, nomes, titulo):
    out = im.copy()
    h, w = out.shape[:2]
    esp = max(2, round(max(h, w) / 600))       # espessura proporcional ao tamanho da foto
    esc = max(0.5, max(h, w) / 1600)            # tamanho do texto
    for (x1, y1, x2, y2), c, k in zip(caixas.astype(int), confs, classes):
        cor = CORES[k % len(CORES)]
        cv2.rectangle(out, (x1, y1), (x2, y2), cor, esp)
        txt = f"{nomes[k]} {c:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, esc, esp)
        y0 = max(y1, th + 6)
        cv2.rectangle(out, (x1, y0 - th - 6), (x1 + tw + 4, y0), cor, -1)
        cv2.putText(out, txt, (x1 + 2, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, esc, (255, 255, 255), esp)
    faixa = np.full((round(60 * esc) + 20, w, 3), 30, np.uint8)
    cv2.putText(faixa, titulo, (10, faixa.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, esc * 1.4,
                (255, 255, 255), esp)
    return np.vstack([faixa, out])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pasta", required=True, help="pasta com as fotos")
    ap.add_argument("--saida", default="teste_fotos")
    ap.add_argument("--rtdetr", default=r"treinos\rtdetr_l_640\weights\best.pt")
    ap.add_argument("--yolo", default=r"treinos\yolo26m_640\weights\best.pt")
    ap.add_argument("--frcnn", default=r"treinos\frcnn_r50v2_640_passos\best.pt")
    ap.add_argument("--conf", type=float, default=0.25, help="limiar de confianca para os tres")
    ap.add_argument("--conf-rtdetr", type=float, default=None)
    ap.add_argument("--conf-yolo", type=float, default=None)
    ap.add_argument("--conf-frcnn", type=float, default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None, help="vazio = GPU se houver, senao CPU")
    args = ap.parse_args()

    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    fotos = sorted(p for p in Path(args.pasta).iterdir() if p.suffix.lower() in EXTENSOES)
    if not fotos:
        raise SystemExit(f"Nenhuma foto encontrada em {Path(args.pasta).resolve()}")
    for nome, c in (("RT-DETR", args.rtdetr), ("YOLO", args.yolo), ("Faster R-CNN", args.frcnn)):
        if not Path(c).exists():
            raise SystemExit(f"ERRO: modelo nao encontrado para {nome}: {Path(c).resolve()}")
    saida = Path(args.saida)
    saida.mkdir(exist_ok=True)

    confs = {"rtdetr_l": args.conf_rtdetr or args.conf, "yolo26m": args.conf_yolo or args.conf,
             "frcnn_r50v2": args.conf_frcnn or args.conf}
    print(f"Dispositivo: {dev} | {len(fotos)} fotos | limiares: {confs}")
    modelos = [
        ("RT-DETR-L", "rtdetr_l", *modelo_ultralytics(RTDETR, args.rtdetr, confs["rtdetr_l"], args.imgsz, dev)),
        ("YOLO26m", "yolo26m", *modelo_ultralytics(YOLO, args.yolo, confs["yolo26m"], args.imgsz, dev)),
        ("Faster R-CNN v2", "frcnn_r50v2", *modelo_frcnn(args.frcnn, confs["frcnn_r50v2"], dev)),
    ]

    with open(saida / "deteccoes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["foto", "largura_foto", "altura_foto", "modelo", "limiar", "classe", "confianca",
                    "x1", "y1", "x2", "y2", "largura_px", "altura_px", "altura_px_equiv_1280"])
        for p in fotos:
            im = cv2.imread(str(p))  # aplica a orientacao EXIF do celular
            if im is None:
                print(f"AVISO: nao consegui abrir {p.name} (se for HEIC, converta para JPG)")
                continue
            H, W = im.shape[:2]
            paineis, resumo = [], []
            for rotulo, slug, prever, nomes in modelos:
                caixas, cs, ks = prever(im)
                cont = {}
                for (x1, y1, x2, y2), c, k in zip(caixas, cs, ks):
                    cont[nomes[k]] = cont.get(nomes[k], 0) + 1
                    w.writerow([p.name, W, H, rotulo, confs[slug], nomes[k], round(float(c), 4),
                                round(float(x1), 1), round(float(y1), 1), round(float(x2), 1),
                                round(float(y2), 1), round(float(x2 - x1), 1), round(float(y2 - y1), 1),
                                round(float(y2 - y1) * 1280 / W, 1)])
                txt = ", ".join(f"{n} {q}" for n, q in sorted(cont.items())) or "nada detectado"
                img = desenhar(im, caixas, cs, ks, nomes, f"{rotulo} (conf >= {confs[slug]:.2f}): {txt}")
                cv2.imwrite(str(saida / f"{p.stem}_{slug}.jpg"), img)
                paineis.append(cv2.resize(img, (1280, round(img.shape[0] * 1280 / img.shape[1]))))
                resumo.append(f"   {rotulo:>16}: {txt}")
            cv2.imwrite(str(saida / f"{p.stem}_comparacao.jpg"), np.vstack(paineis))
            print(f"{p.name} ({W}x{H})\n" + "\n".join(resumo), flush=True)
    print(f"\nImagens e deteccoes.csv salvos em {saida.resolve()}")


if __name__ == "__main__":
    main()
