"""Faster R-CNN (torchvision) no dataset do TCC: sonda de hardware e treino com retomada.

Uso (venv ativado, na pasta C:\\tcc):
    python treinar_frcnn.py --sonda              # mede memoria e tempo (r50 e r50v2)
    python treinar_frcnn.py --arq r50v2 --batch 4   # treino completo

Para interromper: Ctrl+C (uma vez). Para continuar: rode exatamente o mesmo comando.

Protocolo igual ao do RT-DETR e do YOLO:
  - mesmas imagens e anotacoes (formato YOLO em C:\\bdd_tcc), 640 px no lado maior;
  - pesos pre-treinados no COCO, com as classes car, truck, bus e person
    reaproveitadas da cabeca do COCO (traffic sign aprendida do zero);
  - 100 epocas no maximo, parada antecipada com paciencia 20;
  - nota de cada epoca = mAP50-95 na validacao, calculado com o MESMO codigo de
    metricas da Ultralytics 8.4.173 (DetMetrics), com conf 0,001 e ate 300 deteccoes;
  - agenda da taxa de aprendizado: "linear" (mesma queda linear ate 1% da Ultralytics)
    ou "passos" (agenda de referencia do torchvision: 26 epocas, taxa dividida por 10
    nas epocas 16 e 22, sem parada antecipada; vale a melhor epoca na validacao).
Receita propria do Faster R-CNN (referencia do torchvision): SGD, momentum 0,9,
weight decay 1e-4, taxa 0,02 x lote/16, aquecimento linear de 1000 iteracoes,
aumento de dados apenas por espelhamento horizontal.

Saida em C:\\tcc\\treinos\\frcnn_<arq>_640\\ :
    best.pt, last.pt, last_copia.pt, results.csv, registro_epocas.csv,
    final_val.json (metricas por classe do melhor modelo) e CONCLUIDO.txt
"""
import argparse
import csv
import ctypes
import json
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import pil_to_tensor
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils.metrics import DetMetrics, box_iou

ARQS = {
    "r50": (fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights.COCO_V1),
    "r50v2": (fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1),
}
IOUV = SimpleNamespace(iouv=torch.linspace(0.5, 0.95, 10))  # mesmos 10 limiares da Ultralytics


class MemoriaNoLimite(RuntimeError):
    pass


def agora():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def anexar(caminho, cabecalho, linha):
    novo = not caminho.exists()
    with open(caminho, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if novo:
            w.writerow(cabecalho)
        w.writerow(linha)


# ------------------------------------------------------------------ dados
def ler_rotulos(txt, W, H):
    """Le um .txt no formato YOLO e devolve caixas xyxy em pixels da imagem original.
    Linhas repetidas sao descartadas, como faz a Ultralytics."""
    caixas, classes, vistas = [], [], set()
    if txt.exists():
        for linha in txt.read_text().splitlines():
            linha = linha.strip()
            if not linha or linha in vistas:
                continue
            vistas.add(linha)
            c, cx, cy, w, h = linha.split()
            cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
            if w > 0 and h > 0:
                caixas.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
                classes.append(int(c))
    return (np.array(caixas, dtype=np.float32).reshape(-1, 4), np.array(classes, dtype=np.int64))


class BDDYolo(Dataset):
    def __init__(self, raiz, split, imgsz, treino, indices=None):
        self.imgs = sorted((Path(raiz) / "images" / split).glob("*.jpg"))
        if indices is not None:
            self.imgs = [self.imgs[i] for i in indices]
        self.rotulos = Path(raiz) / "labels" / split
        self.imgsz, self.treino = imgsz, treino

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        p = self.imgs[i]
        img = Image.open(p).convert("RGB")
        W, H = img.size
        caixas, classes = ler_rotulos(self.rotulos / (p.stem + ".txt"), W, H)
        s = self.imgsz / max(W, H)
        if s != 1:
            img = img.resize((round(W * s), round(H * s)), Image.BILINEAR)
        t = pil_to_tensor(img)  # uint8; vira float na GPU
        b = torch.from_numpy(caixas * s)
        if self.treino and torch.rand(1).item() < 0.5:  # espelhamento horizontal
            t = t.flip(-1)
            b = torch.stack([t.shape[-1] - b[:, 2], b[:, 1], t.shape[-1] - b[:, 0], b[:, 3]], 1)
        alvo = {"boxes": b.float().reshape(-1, 4), "labels": torch.from_numpy(classes + 1)}
        meta = {"caixas": caixas, "classes": classes, "escala": s, "nome": p.name}
        return t, alvo, meta


def juntar(lote):
    return tuple(zip(*lote))


def carregador(ds, batch, workers, embaralhar, semente=0):
    g = torch.Generator()
    g.manual_seed(semente)
    return DataLoader(ds, batch_size=batch, shuffle=embaralhar, num_workers=workers,
                      collate_fn=juntar, pin_memory=True, generator=g,
                      persistent_workers=workers > 0)


# ------------------------------------------------------------------ modelo
def criar_modelo(arq, imgsz, nomes, pesos=True):
    fn, W = ARQS[arq]
    m = fn(weights=W if pesos else None, weights_backbone=None, min_size=imgsz, max_size=imgsz,
           box_score_thresh=0.001, box_detections_per_img=300)
    antigo = m.roi_heads.box_predictor
    novo = FastRCNNPredictor(antigo.cls_score.in_features, len(nomes) + 1)
    cats = W.meta["categories"]
    reaproveitadas = []
    with torch.no_grad():
        for k, nome in enumerate(["__background__"] + list(nomes)):
            if pesos and nome in cats:
                j = cats.index(nome)
                novo.cls_score.weight[k] = antigo.cls_score.weight[j]
                novo.cls_score.bias[k] = antigo.cls_score.bias[j]
                novo.bbox_pred.weight[4 * k:4 * k + 4] = antigo.bbox_pred.weight[4 * j:4 * j + 4]
                novo.bbox_pred.bias[4 * k:4 * k + 4] = antigo.bbox_pred.bias[4 * j:4 * j + 4]
                if k > 0:
                    reaproveitadas.append(nome)
    m.roi_heads.box_predictor = novo
    return m, reaproveitadas


def para_gpu(imgs, alvos, dev):
    imgs = [im.to(dev, non_blocking=True).float().div_(255) for im in imgs]
    alvos = [{k: v.to(dev, non_blocking=True) for k, v in a.items()} for a in alvos] if alvos else None
    return imgs, alvos


@torch.no_grad()
def avaliar(model, loader, dev, nomes, amp):
    """mAP com o mesmo codigo da Ultralytics, em coordenadas da imagem original."""
    model.eval()
    met = DetMetrics(names=dict(enumerate(nomes)))
    for imgs, _, metas in loader:
        imgs, _ = para_gpu(imgs, None, dev)
        with torch.autocast(device_type=dev.type, enabled=amp):
            saidas = model(imgs)
        for o, meta in zip(saidas, metas):
            pb = o["boxes"].float().cpu() / meta["escala"]
            pc = o["labels"].cpu() - 1
            conf = o["scores"].float().cpu()
            gb, gc = torch.from_numpy(meta["caixas"]), torch.from_numpy(meta["classes"])
            if len(gc) == 0 or len(pc) == 0:
                tp = np.zeros((len(pc), 10), dtype=bool)
            else:
                tp = BaseValidator.match_predictions(IOUV, pc, gc, box_iou(gb, pb)).numpy()
            met.update_stats({"tp": tp, "conf": conf.numpy(), "pred_cls": pc.numpy(),
                              "target_cls": gc.numpy(), "target_img": np.unique(gc.numpy()),
                              "im_name": meta["nome"]})
    met.process()
    model.train()
    return met


def tabela(met, nomes):
    b = met.box
    linhas = [f"{'Classe':>14}{'Imagens':>9}{'Objetos':>9}{'P':>8}{'R':>8}{'mAP50':>8}{'mAP50-95':>10}",
              f"{'all':>14}{'':>9}{int(met.nt_per_class.sum()):>9}{b.mp:>8.3f}{b.mr:>8.3f}{b.map50:>8.3f}{b.map:>10.3f}"]
    por_classe = {}
    for i, c in enumerate(b.ap_class_index):
        p, r, ap50, ap = b.class_result(i)
        por_classe[nomes[c]] = {"P": p, "R": r, "mAP50": ap50, "mAP50-95": ap,
                                "imagens": int(met.nt_per_image[c]), "objetos": int(met.nt_per_class[c])}
        linhas.append(f"{nomes[c]:>14}{int(met.nt_per_image[c]):>9}{int(met.nt_per_class[c]):>9}"
                      f"{p:>8.3f}{r:>8.3f}{ap50:>8.3f}{ap:>10.3f}")
    return "\n".join(linhas), por_classe


def salvar_atomico(obj, caminho):
    tmp = caminho.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, caminho)


def tempo_das_epocas(registro):
    dur = {}
    with open(registro, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            dur[int(r["epoca"])] = float(r["duracao_epoca_s"])
    return sum(dur.values())


# ------------------------------------------------------------------ treino
def treinar(args):
    raiz = Path(args.data).parent
    nomes = list(yaml.safe_load(Path(args.data).read_text())["names"].values())
    sufixo = "_passos" if args.agenda == "passos" else ""
    run = Path(args.projeto).resolve() / f"frcnn_{args.arq}_{args.imgsz}{sufixo}"
    run.mkdir(parents=True, exist_ok=True)
    last, copia, best, feito = run / "last.pt", run / "last_copia.pt", run / "best.pt", run / "CONCLUIDO.txt"
    if feito.exists():
        print(f"Este treino ja terminou:\n{feito.read_text(encoding='utf-8')}")
        return

    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    usa_gpu = dev.type == "cuda"
    amp = usa_gpu
    vram = torch.cuda.get_device_properties(0).total_memory / 2**20 if usa_gpu else 0
    torch.manual_seed(0), random.seed(0), np.random.seed(0)

    estado = None
    for ck in (last, copia):
        if ck.exists():
            try:
                estado = torch.load(ck, map_location="cpu", weights_only=False)
                print(f"\n>>> Retomando do checkpoint {ck} (epocas concluidas: {estado['epoca']})\n")
                break
            except Exception as e:
                print(f"AVISO: nao foi possivel ler {ck} ({str(e)[:80]}). Tentando a copia.")
    passos = args.agenda == "passos"
    cfg = dict(estado["cfg"]) if estado else {
        "arq": args.arq, "imgsz": args.imgsz, "batch": args.batch or 4,
        "epochs": args.epochs if not passos or args.epochs != 100 else 26,
        "patience": args.patience, "workers": args.workers, "agenda": args.agenda,
        "passos": args.passos}
    if cfg.get("agenda") == "passos":
        cfg["patience"] = cfg["epochs"]  # receita de referencia: sem parada antecipada
    if estado and args.batch:
        cfg["batch"] = args.batch
    lr0 = 0.02 * cfg["batch"] / 16

    # sempre com os pesos do COCO (do cache local), para a arquitetura ser identica na retomada
    model, reaproveitadas = criar_modelo(cfg["arq"], cfg["imgsz"], nomes)
    model.to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr0, momentum=0.9, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    inicio, melhor, melhor_ep = 0, 0.0, 0
    if estado:
        model.load_state_dict(estado["modelo"])
        opt.load_state_dict(estado["otim"])
        scaler.load_state_dict(estado["scaler"])
        inicio, melhor, melhor_ep = estado["epoca"], estado["melhor"], estado["melhor_epoca"]
    else:
        print(f"\n>>> Treino novo em {run}\n    Classes aproveitadas do COCO: {reaproveitadas}")
    print(f"    Configuracao: {cfg} | lr0 {lr0:.5f} | AMP {amp} | dispositivo {dev}")

    ds_tr = BDDYolo(raiz, "train", cfg["imgsz"], treino=True)
    ds_va = BDDYolo(raiz, "val", cfg["imgsz"], treino=False)
    dl_tr = carregador(ds_tr, cfg["batch"], cfg["workers"], True, semente=inicio)
    dl_va = carregador(ds_va, cfg["batch"], cfg["workers"], False)
    nb, E = len(dl_tr), cfg["epochs"]
    aquecimento = min(1000, nb - 1)
    if cfg.get("agenda") == "passos":  # taxa / 10 a cada passo (referencia do torchvision)
        lf = lambda x: 0.1 ** sum(x >= p for p in cfg["passos"])
    else:  # queda linear igual a da Ultralytics
        lf = lambda x: max(1 - x / E, 0) * (1 - 0.01) + 0.01

    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    motivo, detalhe = "erro", ""
    try:
        for ep in range(inicio, E):
            t_ep = time.time()
            if usa_gpu:
                torch.cuda.reset_peak_memory_stats()
            soma, n_ok, ruins = None, 0, 0
            passo = max(1, nb // 10)
            for i, (imgs, alvos, _) in enumerate(dl_tr):
                it = ep * nb + i
                fator = (0.001 + (1 - 0.001) * it / aquecimento) if it < aquecimento else 1.0
                for g in opt.param_groups:
                    g["lr"] = lr0 * lf(ep) * fator
                imgs, alvos = para_gpu(imgs, alvos, dev)
                with torch.autocast(device_type=dev.type, enabled=amp):
                    perdas = model(imgs, alvos)
                    perda = sum(perdas.values())
                if not torch.isfinite(perda):
                    ruins += 1
                    if ruins > 50:
                        raise RuntimeError("perda invalida (NaN/inf) em mais de 50 lotes")
                    continue
                opt.zero_grad(set_to_none=True)
                scaler.scale(perda).backward()
                scaler.step(opt)
                scaler.update()
                v = torch.stack([x.detach().float() for x in perdas.values()])
                soma = v if soma is None else soma + v
                n_ok += 1
                if usa_gpu and i % 25 == 0:
                    pico = torch.cuda.max_memory_reserved() / 2**20
                    if pico > 0.95 * vram:
                        raise MemoriaNoLimite(f"pico de {pico:.0f} MiB em uma placa de {vram:.0f} MiB")
                if (i + 1) % passo == 0 or i + 1 == nb:
                    gm = torch.cuda.max_memory_reserved() / 2**30 if usa_gpu else 0
                    print(f"   epoca {ep + 1}/{E}  lote {i + 1}/{nb}  GPU {gm:.2f}G  "
                          f"perda {float(soma.sum()) / n_ok:.3f}  {time.time() - t_ep:.0f} s", flush=True)
            met = avaliar(model, dl_va, dev, nomes, amp)
            nota = float(met.box.map)
            if nota > melhor or melhor == 0:
                melhor, melhor_ep = nota, ep + 1
                salvar_atomico({"modelo": model.state_dict(), "arq": cfg["arq"], "imgsz": cfg["imgsz"],
                                "nomes": nomes, "epoca": ep + 1, "nota": nota}, best)
            salvar_atomico({"epoca": ep + 1, "modelo": model.state_dict(), "otim": opt.state_dict(),
                            "scaler": scaler.state_dict(), "melhor": melhor, "melhor_epoca": melhor_ep,
                            "cfg": cfg}, last)
            shutil.copy2(last, run / "last_copia.tmp")
            os.replace(run / "last_copia.tmp", copia)
            dur = time.time() - t_ep
            medias = (soma / max(n_ok, 1)).tolist() if soma is not None else [0.0] * len(perdas)
            anexar(run / "results.csv",
                   ["epoch"] + [f"train/{k}" for k in perdas] + ["metrics/precision(B)", "metrics/recall(B)",
                    "metrics/mAP50(B)", "metrics/mAP50-95(B)", "lr", "lotes_descartados"],
                   [ep + 1] + [round(x, 5) for x in medias] + [round(met.box.mp, 5), round(met.box.mr, 5),
                    round(met.box.map50, 5), round(nota, 5), lr0 * lf(ep), ruins])
            anexar(run / "registro_epocas.csv",
                   ["data_hora", "epoca", "duracao_epoca_s", "pico_gpu_reservado_MiB",
                    "pico_gpu_alocado_MiB", "nota_mAP50-95", "melhor_nota_ate_aqui"],
                   [agora(), ep + 1, round(dur, 1),
                    round(torch.cuda.max_memory_reserved() / 2**20) if usa_gpu else "",
                    round(torch.cuda.max_memory_allocated() / 2**20) if usa_gpu else "",
                    round(nota, 5), round(melhor, 5)])
            print(f">> Epoca {ep + 1}/{E} | {dur:.0f} s | P {met.box.mp:.3f} R {met.box.mr:.3f} "
                  f"mAP50 {met.box.map50:.3f} mAP50-95 {nota:.3f} | melhor {melhor:.3f} (epoca {melhor_ep})",
                  flush=True)
            if ep + 1 - melhor_ep >= cfg["patience"]:
                print(f"\nParada antecipada: {cfg['patience']} epocas sem melhora. Melhor epoca: {melhor_ep}.")
                break
        motivo = "concluido"
    except KeyboardInterrupt:
        motivo = "interrompido"
    except MemoriaNoLimite as e:
        motivo, detalhe = "memoria", str(e)
    finally:
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

    if motivo == "concluido":
        print("\nValidando o melhor modelo (best.pt)...")
        model.load_state_dict(torch.load(best, map_location="cpu", weights_only=False)["modelo"])
        met = avaliar(model, dl_va, dev, nomes, amp)
        txt, por_classe = tabela(met, nomes)
        print(txt)
        (run / "final_val.json").write_text(json.dumps(
            {"melhor_epoca": melhor_ep, "mAP50-95": met.box.map, "mAP50": met.box.map50,
             "P": met.box.mp, "R": met.box.mr, "por_classe": por_classe}, indent=2), encoding="utf-8")
        total = tempo_das_epocas(run / "registro_epocas.csv")
        feito.write_text(f"Terminado em {agora()}\nMelhor epoca: {melhor_ep}\n"
                         f"Tempo das epocas (treino + validacao, todas as sessoes): {total / 3600:.2f} h\n"
                         f"Melhor modelo: {best}\n", encoding="utf-8")
        print(f"\n>>> TREINO CONCLUIDO.\n{feito.read_text(encoding='utf-8')}")
    elif motivo == "interrompido":
        print("\n>>> Treino interrompido. A epoca em andamento sera refeita.\n"
              ">>> Para continuar, rode o mesmo comando de novo.")
    elif motivo == "memoria":
        print(f"\n>>> TREINO PARADO: memoria da GPU no limite ({detalhe}).\n"
              ">>> Rode de novo com um lote menor (--batch) e envie esta mensagem.")


# ------------------------------------------------------------------ sonda
def sonda_um(args, arq, batch, saida):
    raiz = Path(args.data).parent
    nomes = list(yaml.safe_load(Path(args.data).read_text())["names"].values())
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    usa_gpu, res = dev.type == "cuda", {"arq": arq, "lote": batch}
    try:
        torch.manual_seed(0)
        model, _ = criar_modelo(arq, args.imgsz, nomes)
        model.to(dev).train()
        res["parametros_M"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 1)
        opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                              lr=0.02 * batch / 16 * 0.1, momentum=0.9, weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=usa_gpu)
        n_tr = len(list((raiz / "images" / "train").glob("*.jpg")))
        idx = random.Random(0).sample(range(n_tr), min(n_tr, batch * 120))
        dl = carregador(BDDYolo(raiz, "train", args.imgsz, True, idx), batch, args.workers, True)
        if usa_gpu:
            torch.cuda.reset_peak_memory_stats()
        t0, n_img, medindo = None, 0, False
        aquec = min(15, len(dl) // 4)
        for i, (imgs, alvos, _) in enumerate(dl):
            if i == aquec:  # descarta o aquecimento
                if usa_gpu:
                    torch.cuda.synchronize()
                t0, n_img, medindo = time.time(), 0, True
            imgs, alvos = para_gpu(imgs, alvos, dev)
            with torch.autocast(device_type=dev.type, enabled=usa_gpu):
                perda = sum(model(imgs, alvos).values())
            opt.zero_grad(set_to_none=True)
            scaler.scale(perda).backward()
            scaler.step(opt)
            scaler.update()
            if medindo:
                n_img += len(imgs)
                if time.time() - t0 > 90:
                    break
        if usa_gpu:
            torch.cuda.synchronize()
        s_tr = (time.time() - t0) / max(n_img, 1)
        n_va = len(list((raiz / "images" / "val").glob("*.jpg")))
        dl_va = carregador(BDDYolo(raiz, "val", args.imgsz, False, list(range(min(n_va, 200)))),
                           batch, args.workers, False)
        t1 = time.time()
        avaliar(model, dl_va, dev, nomes, usa_gpu)
        s_va = (time.time() - t1) / min(n_va, 200)
        reservado = torch.cuda.max_memory_reserved() / 2**20 if usa_gpu else 0
        total = torch.cuda.get_device_properties(0).total_memory / 2**20 if usa_gpu else 1
        res.update(status="ok", pico_reservado_MiB=round(reservado), vram_MiB=round(total),
                   no_limite=bool(reservado > 0.92 * total), s_img_treino=round(s_tr, 4),
                   s_img_val=round(s_va, 4),
                   min_por_epoca=round((s_tr * 10605 + s_va * 1872) / 60, 1))
    except Exception as e:
        res.update(status="falta_memoria" if "out of memory" in str(e).lower() else "erro",
                   erro=str(e)[:300])
    Path(saida).write_text(json.dumps(res))


def sonda(args):
    resultados = []
    for arq in args.arqs:
        for b in args.lotes:
            print(f"\n===== Faster R-CNN {arq} | lote {b} =====", flush=True)
            tmp = Path("_sonda_frcnn.json")
            tmp.unlink(missing_ok=True)
            subprocess.run([sys.executable, __file__, "--data", args.data, "--imgsz", str(args.imgsz),
                            "--workers", str(args.workers), "--_sonda", arq, str(b)])
            r = json.loads(tmp.read_text()) if tmp.exists() else {"arq": arq, "lote": b, "status": "processo_caiu"}
            tmp.unlink(missing_ok=True)
            resultados.append(r)
            print(r, flush=True)
            Path("sonda_frcnn.json").write_text(json.dumps(resultados, indent=2))
            if r.get("status") == "ok" and not r["no_limite"]:
                break
    print(f"\n{'arq':<7}{'lote':>5}{'status':>15}{'params M':>10}{'pico MiB':>10}{'limite?':>9}"
          f"{'s/img trein':>12}{'s/img val':>10}{'min/epoca':>11}")
    for r in resultados:
        print(f"{r['arq']:<7}{r['lote']:>5}{r['status']:>15}{str(r.get('parametros_M', '-')):>10}"
              f"{str(r.get('pico_reservado_MiB', '-')):>10}{str(r.get('no_limite', '-')):>9}"
              f"{str(r.get('s_img_treino', '-')):>12}{str(r.get('s_img_val', '-')):>10}"
              f"{str(r.get('min_por_epoca', '-')):>11}")
        if r["status"] != "ok":
            print("      ", r.get("erro", ""))
    print("\nmin/epoca = estimativa para 10.605 imagens de treino + 1.872 de validacao. Salvo em sonda_frcnn.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"C:\bdd_tcc\data.yaml")
    ap.add_argument("--arq", choices=list(ARQS), default="r50v2")
    ap.add_argument("--projeto", default="treinos")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch", type=int, default=None, help="padrao: 4")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--agenda", choices=["linear", "passos"], default="linear")
    ap.add_argument("--passos", type=int, nargs="+", default=[16, 22],
                    help="(agenda passos) epocas em que a taxa e dividida por 10")
    ap.add_argument("--sonda", action="store_true", help="mede memoria e tempo em vez de treinar")
    ap.add_argument("--arqs", nargs="+", default=["r50", "r50v2"], help="(sonda) arquiteturas")
    ap.add_argument("--lotes", type=int, nargs="+", default=[8, 4, 2], help="(sonda) lotes a tentar")
    ap.add_argument("--_sonda", nargs=2, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args._sonda:
        sonda_um(args, args._sonda[0], int(args._sonda[1]), "_sonda_frcnn.json")
    elif args.sonda:
        sonda(args)
    else:
        treinar(args)


if __name__ == "__main__":
    main()
