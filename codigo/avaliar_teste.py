"""Avaliacao final dos tres modelos, com o mesmo avaliador para todos.

Uso (venv ativado, na pasta C:\\tcc, com treinar_frcnn.py na mesma pasta):
  1) Conferencia na VALIDACAO (rapida; deve reproduzir as notas do treino):
       python avaliar_teste.py --split val --sem-latencia --bootstrap 0
  2) Avaliacao final no TESTE (uma unica vez):
       python avaliar_teste.py

O que mede, para cada modelo:
  - mAP50-95, mAP50, precisao e recall, geral e por classe, com o MESMO codigo de
    metricas usado no treino (DetMetrics da Ultralytics 8.4.173), conf 0,001, ate 300 deteccoes;
  - metricas COCO oficiais (pycocotools): AP, AP50, AP75 e AP por tamanho de objeto
    (pequeno < 32x32 px, medio ate 96x96 px, grande acima), em pixels da imagem original;
  - intervalos de confianca de 95% por bootstrap pareado (mesmos sorteios de imagens para
    os tres modelos), inclusive para as diferencas entre pares de modelos;
  - matriz de confusao (conf >= 0,25, IoU >= 0,45);
  - latencia por imagem de ponta a ponta (pre-processamento + inferencia + pos-processamento),
    lote 1, em FP32 e FP16, com aquecimento, e pico de memoria da GPU na inferencia.

Saida em C:\\tcc\\avaliacao_<split>\\ : resumo.txt, metricas.json, por_classe.csv, coco.csv,
bootstrap.csv, diferencas.csv, latencia.csv, confusao_<modelo>.csv/.png e as previsoes salvas
(previsoes_<modelo>.pkl; se existirem, sao reaproveitadas -- apague para refazer).
"""
import argparse
import contextlib
import csv
import io
import json
import pickle
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from ultralytics import RTDETR, YOLO
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils.metrics import DetMetrics, ap_per_class, box_iou

import treinar_frcnn as F

VAZIO = (np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64))


# ------------------------------------------------------------------ dados
def carregar_gt(raiz, split):
    gt = {}
    for p in sorted((raiz / "images" / split).glob("*.jpg")):
        with Image.open(p) as im:
            W, H = im.size
        caixas, classes = F.ler_rotulos(raiz / "labels" / split / (p.stem + ".txt"), W, H)
        gt[p.name] = (caixas, classes, W, H)
    return gt


# ------------------------------------------------------------------ previsoes
def prever_ultra(tipo, caminho, raiz, split, dev, imgsz, lote):
    """Previsoes do RT-DETR e do YOLO capturadas DENTRO do validador oficial da Ultralytics
    (o mesmo que deu as notas do treino): mesmo redimensionamento (lotes retangulares),
    mesmo NMS (iou 0,7, multi_label=True: uma caixa pode valer para mais de uma classe),
    mesmo conf 0,001 e max_det. As caixas sao levadas para os pixels da imagem original.
    Motivo: o predict() padrao usa multi_label=False e outro preenchimento da imagem,
    o que dava 0,397 em vez de 0,412 no YOLO26m (RT-DETR nao usa NMS e ja batia)."""
    m = (RTDETR if tipo == "rtdetr" else YOLO)(str(caminho))
    out = {}

    class Captura(m._smart_load("validator")):
        def update_metrics(self, preds, batch):
            for si, pred in enumerate(preds):
                pb = self._prepare_batch(si, batch)
                pn = self.scale_preds({k: v.clone() for k, v in pred.items()}, pb)
                out[Path(pb["im_file"]).name] = (pn["bboxes"].float().cpu().numpy(),
                                                 pn["conf"].float().cpu().numpy(),
                                                 pn["cls"].cpu().numpy().astype(np.int64))
            super().update_metrics(preds, batch)

    r = m.val(validator=Captura, data=str(raiz / "data.yaml"), split=split, imgsz=imgsz,
              batch=lote, conf=0.001, iou=0.7, max_det=300, plots=False,
              device=str(dev.index or 0) if dev.type == "cuda" else "cpu",
              verbose=False, project="runs_avaliacao", name=f"{tipo}_{split}", exist_ok=True)
    print(f"   validador oficial da Ultralytics: mAP50-95 = {r.box.map:.4f}  "
          f"(o resumo abaixo deve dar o mesmo valor)", flush=True)
    return out


def carregar_frcnn(caminho, dev):
    ck = torch.load(caminho, map_location="cpu", weights_only=False)
    m, _ = F.criar_modelo(ck["arq"], ck["imgsz"], ck["nomes"])
    m.load_state_dict(ck["modelo"])
    return m.to(dev).eval(), ck


@torch.no_grad()
def prever_frcnn(caminho, raiz, split, dev, lote, workers):
    m, ck = carregar_frcnn(caminho, dev)
    dl = F.carregador(F.BDDYolo(raiz, split, ck["imgsz"], treino=False), lote, workers, False)
    out = {}
    for imgs, _, metas in dl:
        imgs, _ = F.para_gpu(imgs, None, dev)
        for o, meta in zip(m(imgs), metas):  # FP32, igual aos modelos da Ultralytics
            out[meta["nome"]] = ((o["boxes"].float().cpu() / meta["escala"]).numpy().astype(np.float32),
                                 o["scores"].float().cpu().numpy().astype(np.float32),
                                 (o["labels"].cpu() - 1).numpy().astype(np.int64))
    return out


# ------------------------------------------------------------------ metricas
def estatisticas_por_imagem(preds, gt):
    """Para cada imagem: matriz de acertos (10 limiares de IoU), confiancas, classes."""
    lista = []
    for nome, (gb, gc, _, _) in gt.items():
        pb, ps, pc = preds.get(nome, VAZIO)
        if len(gc) == 0 or len(pc) == 0:
            tp = np.zeros((len(pc), 10), dtype=bool)
        else:
            iou = box_iou(torch.from_numpy(gb), torch.from_numpy(pb))
            tp = BaseValidator.match_predictions(F.IOUV, torch.from_numpy(pc), torch.from_numpy(gc), iou).numpy()
        lista.append((tp, ps, pc, gc, nome))
    return lista


def metricas_ultra(lista, nomes):
    met = DetMetrics(names=dict(enumerate(nomes)))
    for tp, ps, pc, gc, nome in lista:
        met.update_stats({"tp": tp, "conf": ps, "pred_cls": pc, "target_cls": gc,
                          "target_img": np.unique(gc), "im_name": nome})
    met.process()
    return met


def map_rapido(lista, idx):
    tp = np.concatenate([lista[i][0] for i in idx])
    ps = np.concatenate([lista[i][1] for i in idx])
    pc = np.concatenate([lista[i][2] for i in idx])
    gc = np.concatenate([lista[i][3] for i in idx])
    ap = ap_per_class(tp, ps, pc, gc)[5]
    return float(ap.mean()), float(ap[:, 0].mean())


def coco_gt(gt, nomes):
    from pycocotools.coco import COCO
    imagens, anots, ids = [], [], {}
    for i, (nome, (gb, gc, W, H)) in enumerate(gt.items(), start=1):
        ids[nome] = i
        imagens.append({"id": i, "file_name": nome, "width": W, "height": H})
        for b, c in zip(gb, gc):
            w, h = float(b[2] - b[0]), float(b[3] - b[1])
            anots.append({"id": len(anots) + 1, "image_id": i, "category_id": int(c) + 1,
                          "bbox": [float(b[0]), float(b[1]), w, h], "area": w * h, "iscrowd": 0})
    C = COCO()
    C.dataset = {"images": imagens, "annotations": anots,
                 "categories": [{"id": k + 1, "name": n} for k, n in enumerate(nomes)]}
    with contextlib.redirect_stdout(io.StringIO()):
        C.createIndex()
    return C, ids


def metricas_coco(C, ids, preds):
    from pycocotools.cocoeval import COCOeval
    res = []
    for nome, (pb, ps, pc) in preds.items():
        for b, s, c in zip(pb, ps, pc):
            res.append({"image_id": ids[nome], "category_id": int(c) + 1, "score": float(s),
                        "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]})
    if not res:
        return [0.0] * 12, "sem deteccoes"
    texto = io.StringIO()
    with contextlib.redirect_stdout(texto):
        D = C.loadRes(res)
        E = COCOeval(C, D, "bbox")
        E.evaluate()
        E.accumulate()
        E.summarize()
    linhas = [l for l in texto.getvalue().splitlines() if l.strip().startswith("Average")]
    return [float(x) for x in E.stats], "\n".join(linhas)


def matriz_confusao(preds, gt, nc, conf=0.25, iou_min=0.45):
    """Linhas = classe prevista (ultima = fundo); colunas = classe real (ultima = fundo)."""
    M = np.zeros((nc + 1, nc + 1), dtype=int)
    for nome, (gb, gc, _, _) in gt.items():
        pb, ps, pc = preds.get(nome, VAZIO)
        k = ps >= conf
        pb, pc = pb[k], pc[k]
        usados_g, usados_p = set(), set()
        if len(gc) and len(pc):
            I = box_iou(torch.from_numpy(gb), torch.from_numpy(pb)).numpy()
            pares = np.argwhere(I >= iou_min)
            for g, p in pares[np.argsort(-I[pares[:, 0], pares[:, 1]])]:
                if g not in usados_g and p not in usados_p:
                    usados_g.add(g), usados_p.add(p)
                    M[pc[p], gc[g]] += 1
        for g in range(len(gc)):
            if g not in usados_g:
                M[nc, gc[g]] += 1  # objeto nao detectado
        for p in range(len(pc)):
            if p not in usados_p:
                M[pc[p], nc] += 1  # falso positivo (fundo)
    return M


def salvar_confusao(M, nomes, titulo, base):
    rot = list(nomes) + ["fundo"]
    with open(f"{base}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["previsto \\ real"] + rot)
        for i, r in enumerate(M):
            w.writerow([rot[i]] + list(r))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    N = M / np.maximum(M.sum(0, keepdims=True), 1)  # normalizada por classe real
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(N, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(rot)), rot, rotation=45, ha="right")
    ax.set_yticks(range(len(rot)), rot)
    ax.set_xlabel("Classe real")
    ax.set_ylabel("Classe prevista")
    ax.set_title(titulo)
    for i in range(len(rot)):
        for j in range(len(rot)):
            if i == len(rot) - 1 and j == len(rot) - 1:
                continue
            ax.text(j, i, f"{N[i, j]:.2f}", ha="center", va="center",
                    color="white" if N[i, j] > 0.5 else "black", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{base}.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ latencia
def latencia(tipo, caminho, imgs_bgr, imgsz, dev, meia, aquec=20):
    usa_gpu = dev.type == "cuda"
    if usa_gpu:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    sinc = torch.cuda.synchronize if usa_gpu else (lambda: None)
    if tipo in ("rtdetr", "yolo"):
        m = (RTDETR if tipo == "rtdetr" else YOLO)(str(caminho))

        def f(im):
            extra = {"half": True} if meia else {}
            return m.predict(im, imgsz=imgsz, conf=0.25, max_det=300, device=dev,
                             verbose=False, **extra)[0].boxes.xyxy.cpu()
    else:
        m, ck = carregar_frcnn(caminho, dev)
        m.roi_heads.score_thresh = 0.25

        @torch.no_grad()
        def f(im):
            rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            s = ck["imgsz"] / max(h, w)
            rgb = cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_LINEAR)
            t = torch.from_numpy(rgb).permute(2, 0, 1).to(dev).float().div_(255)
            with torch.autocast(device_type=dev.type, enabled=meia and usa_gpu):
                o = m([t])[0]
            return (o["boxes"].float() / s).cpu()
    for im in imgs_bgr[:aquec]:
        f(im)
    tempos = []
    for im in imgs_bgr:
        sinc()
        t0 = time.perf_counter()
        f(im)
        sinc()
        tempos.append((time.perf_counter() - t0) * 1000)
    t = np.array(tempos)
    res = {"media_ms": float(t.mean()), "mediana_ms": float(np.median(t)),
           "p95_ms": float(np.percentile(t, 95)), "fps": float(1000 / t.mean()),
           "pico_alocado_MiB": round(torch.cuda.max_memory_allocated() / 2**20) if usa_gpu else None,
           "pico_reservado_MiB": round(torch.cuda.max_memory_reserved() / 2**20) if usa_gpu else None}
    del m
    if usa_gpu:
        torch.cuda.empty_cache()
    return res


# ------------------------------------------------------------------ principal
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"C:\bdd_tcc\data.yaml")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--rtdetr", default=r"treinos\rtdetr_l_640\weights\best.pt")
    ap.add_argument("--yolo", default=r"treinos\yolo26m_640\weights\best.pt")
    ap.add_argument("--frcnn", default=r"treinos\frcnn_r50v2_640_passos\best.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--lote", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--bootstrap", type=int, default=200, help="numero de sorteios (0 = sem)")
    ap.add_argument("--sem-latencia", action="store_true")
    ap.add_argument("--n-latencia", type=int, default=200)
    ap.add_argument("--device", default=None)
    ap.add_argument("--saida", default=None)
    args = ap.parse_args()

    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    raiz = Path(args.data).parent
    nomes = list(yaml.safe_load(Path(args.data).read_text())["names"].values())
    saida = Path(args.saida or f"avaliacao_{args.split}")
    saida.mkdir(exist_ok=True)
    modelos = {"RT-DETR-L": ("rtdetr", Path(args.rtdetr)), "YOLO26m": ("yolo", Path(args.yolo)),
               "Faster R-CNN v2": ("frcnn", Path(args.frcnn))}
    slug = {"RT-DETR-L": "rtdetr_l", "YOLO26m": "yolo26m", "Faster R-CNN v2": "frcnn_r50v2"}
    for nome, (_, c) in modelos.items():
        if not c.exists():
            raise SystemExit(f"ERRO: modelo nao encontrado para {nome}: {c.resolve()}")

    print(f"Conjunto: {args.split} | dispositivo: {dev}")
    gt = carregar_gt(raiz, args.split)
    n_obj = sum(len(v[1]) for v in gt.values())
    areas = np.concatenate([(v[0][:, 2] - v[0][:, 0]) * (v[0][:, 3] - v[0][:, 1]) for v in gt.values()])
    tam = {"pequenos (<32x32)": int((areas < 32**2).sum()),
           "medios": int(((areas >= 32**2) & (areas < 96**2)).sum()),
           "grandes (>96x96)": int((areas >= 96**2).sum())}
    print(f"{len(gt)} imagens, {n_obj} objetos | por tamanho: {tam}")

    # ---- previsoes (reaproveitadas se ja existirem)
    preds = {}
    for nome, (tipo, cam) in modelos.items():
        arq = saida / f"previsoes_{slug[nome]}.pkl"
        if arq.exists():
            print(f"[{nome}] usando previsoes salvas em {arq} (apague o arquivo para refazer)")
            preds[nome] = pickle.loads(arq.read_bytes())
            continue
        print(f"[{nome}] gerando previsoes...", flush=True)
        t0 = time.time()
        preds[nome] = (prever_frcnn(cam, raiz, args.split, dev, args.lote, args.workers) if tipo == "frcnn"
                       else prever_ultra(tipo, cam, raiz, args.split, dev, args.imgsz, args.lote))
        arq.write_bytes(pickle.dumps(preds[nome]))
        print(f"   {len(preds[nome])} imagens em {time.time() - t0:.0f} s")

    # ---- metricas
    resultado, linhas_classe, linhas_coco, listas = {}, [], [], {}
    C, ids = coco_gt(gt, nomes)
    texto = [f"AVALIACAO no conjunto '{args.split}': {len(gt)} imagens, {n_obj} objetos",
             f"Objetos por tamanho (pixels da imagem original): {tam}", ""]
    for nome in modelos:
        lista = estatisticas_por_imagem(preds[nome], gt)
        listas[nome] = lista
        met = metricas_ultra(lista, nomes)
        b = met.box
        r = {"mAP50-95": b.map, "mAP50": b.map50, "P": b.mp, "R": b.mr, "por_classe": {}}
        texto.append(f"=== {nome} ===")
        texto.append(f"{'Classe':>14}{'Imagens':>9}{'Objetos':>9}{'P':>8}{'R':>8}{'mAP50':>8}{'mAP50-95':>10}")
        texto.append(f"{'all':>14}{len(gt):>9}{n_obj:>9}{b.mp:>8.3f}{b.mr:>8.3f}{b.map50:>8.3f}{b.map:>10.3f}")
        for i, c in enumerate(b.ap_class_index):
            p, rr, a50, a = b.class_result(i)
            r["por_classe"][nomes[c]] = {"P": p, "R": rr, "mAP50": a50, "mAP50-95": a}
            texto.append(f"{nomes[c]:>14}{int(met.nt_per_image[c]):>9}{int(met.nt_per_class[c]):>9}"
                         f"{p:>8.3f}{rr:>8.3f}{a50:>8.3f}{a:>10.3f}")
            linhas_classe.append([nome, nomes[c], int(met.nt_per_class[c]), p, rr, a50, a])
        stats, resumo_coco = metricas_coco(C, ids, preds[nome])
        r["coco"] = dict(zip(["AP", "AP50", "AP75", "AP_pequenos", "AP_medios", "AP_grandes",
                              "AR1", "AR10", "AR100", "AR_pequenos", "AR_medios", "AR_grandes"], stats))
        linhas_coco.append([nome] + stats)
        texto += ["", "  Metricas COCO oficiais (pycocotools):"] + ["  " + l for l in resumo_coco.splitlines()]
        M = matriz_confusao(preds[nome], gt, len(nomes))
        salvar_confusao(M, nomes, f"{nome} ({args.split})", saida / f"confusao_{slug[nome]}")
        r["matriz_confusao"] = M.tolist()
        resultado[nome] = r
        texto.append("")
        print(f"[{nome}] mAP50-95 {b.map:.4f} | mAP50 {b.map50:.4f} | COCO AP {stats[0]:.4f}", flush=True)

    with open(saida / "por_classe.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["modelo", "classe", "objetos", "P", "R", "mAP50", "mAP50-95"])
        w.writerows(linhas_classe)
    with open(saida / "coco.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["modelo", "AP", "AP50", "AP75", "AP_pequenos", "AP_medios", "AP_grandes",
                    "AR1", "AR10", "AR100", "AR_pequenos", "AR_medios", "AR_grandes"])
        w.writerows(linhas_coco)

    # ---- bootstrap pareado
    if args.bootstrap > 0:
        print(f"Bootstrap com {args.bootstrap} sorteios...", flush=True)
        rng = np.random.default_rng(0)
        n = len(gt)
        sorteios = [rng.integers(0, n, n) for _ in range(args.bootstrap)]
        amostras = {nome: np.array([map_rapido(listas[nome], idx) for idx in sorteios]) for nome in modelos}
        ic = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
        texto.append(f"=== Intervalos de confianca de 95% (bootstrap pareado, {args.bootstrap} sorteios) ===")
        with open(saida / "bootstrap.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["modelo", "metrica", "valor", "ic95_inf", "ic95_sup"])
            for nome in modelos:
                for k, rot in ((0, "mAP50-95"), (1, "mAP50")):
                    a, b2 = ic(amostras[nome][:, k])
                    v = resultado[nome][rot]
                    resultado[nome][f"ic95_{rot}"] = [a, b2]
                    w.writerow([nome, rot, v, a, b2])
                    texto.append(f"{nome:>16} {rot:>9}: {v:.4f}  [{a:.4f}; {b2:.4f}]")
        nomes_m = list(modelos)
        with open(saida / "diferencas.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["modelo_A", "modelo_B", "metrica", "diferenca_A_menos_B", "ic95_inf", "ic95_sup",
                        "IC_exclui_zero"])
            texto.append("")
            texto.append("Diferencas entre modelos (A - B):")
            for i in range(3):
                for j in range(i + 1, 3):
                    A, B = nomes_m[i], nomes_m[j]
                    for k, rot in ((0, "mAP50-95"), (1, "mAP50")):
                        d = amostras[A][:, k] - amostras[B][:, k]
                        a, b2 = ic(d)
                        dv = resultado[A][rot] - resultado[B][rot]
                        exclui = a > 0 or b2 < 0
                        w.writerow([A, B, rot, dv, a, b2, exclui])
                        texto.append(f"  {A} - {B} ({rot}): {dv:+.4f}  [{a:+.4f}; {b2:+.4f}]"
                                     f"{'  -> diferenca robusta' if exclui else '  -> IC inclui zero'}")
        texto.append("")

    # ---- latencia
    if not args.sem_latencia:
        print("Medindo latencia...", flush=True)
        todas = sorted((raiz / "images" / args.split).glob("*.jpg"))
        escolhidas = random.Random(0).sample(todas, min(args.n_latencia, len(todas)))
        imgs = [cv2.imread(str(p)) for p in escolhidas]
        precisoes = [("FP32", False)] + ([("FP16", True)] if dev.type == "cuda" else [])
        texto.append(f"=== Latencia por imagem (lote 1, {len(imgs)} imagens, conf 0,25, ponta a ponta) ===")
        texto.append(f"{'modelo':>16}{'precisao':>9}{'media ms':>10}{'mediana':>9}{'p95':>8}{'FPS':>7}{'pico MiB':>10}")
        with open(saida / "latencia.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["modelo", "precisao", "media_ms", "mediana_ms", "p95_ms", "fps",
                        "pico_alocado_MiB", "pico_reservado_MiB"])
            for nome, (tipo, cam) in modelos.items():
                for rot, meia in precisoes:
                    L = latencia(tipo, cam, imgs, args.imgsz, dev, meia)
                    resultado[nome].setdefault("latencia", {})[rot] = L
                    w.writerow([nome, rot, L["media_ms"], L["mediana_ms"], L["p95_ms"], L["fps"],
                                L["pico_alocado_MiB"], L["pico_reservado_MiB"]])
                    texto.append(f"{nome:>16}{rot:>9}{L['media_ms']:>10.1f}{L['mediana_ms']:>9.1f}"
                                 f"{L['p95_ms']:>8.1f}{L['fps']:>7.1f}{str(L['pico_alocado_MiB']):>10}")
                    print(f"   {nome} {rot}: {L['media_ms']:.1f} ms ({L['fps']:.1f} FPS)", flush=True)
        texto.append("")

    (saida / "metricas.json").write_text(json.dumps(
        {"split": args.split, "imagens": len(gt), "objetos": n_obj, "objetos_por_tamanho": tam,
         "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
         "torch": torch.__version__, "modelos": resultado}, indent=2), encoding="utf-8")
    (saida / "resumo.txt").write_text("\n".join(texto) + "\n", encoding="utf-8")
    print("\n" + "\n".join(texto))
    print(f"Resultados salvos em {saida.resolve()}")


if __name__ == "__main__":
    main()
