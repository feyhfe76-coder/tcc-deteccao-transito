"""Gera o dataset do TCC a partir do BDD100K (copia de 2018, um .json por imagem).

Uso (venv ativado, Pillow instalado):
    python preparar_dataset.py C:\\bdd100k C:\\bdd_tcc

O que faz:
  1. Filtra imagens com clima 'clear' e hora 'daytime'.
  2. Mantem so as classes: car, truck, bus, person, traffic sign (as demais,
     incluindo rider, motor, bike, traffic light e train, sao descartadas).
  3. Divide:  teste     = validacao oficial filtrada
              validacao = ~15% do treino oficial filtrado (por grupos de prefixo do nome)
              treino    = o restante
  4. Copia as imagens e grava as anotacoes em dois formatos:
       - YOLO  (.txt por imagem)          -> YOLO e RT-DETR (Ultralytics)
       - COCO  (instances_<grupo>.json)   -> Faster R-CNN e avaliacao comum
  5. Grava train.txt / val.txt / test.txt, data.yaml, resumo.txt e a pasta
     amostras\\ com imagens de conferencia (caixas lidas dos .txt ja convertidos).

O dataset original nao e alterado.
"""
import argparse
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw

CLASSES = ["car", "truck", "bus", "person", "traffic sign"]
CORES = [(0, 200, 255), (255, 140, 0), (200, 80, 255), (255, 60, 60), (255, 220, 0)]
MIN_LADO = 1.0  # caixas com largura ou altura <= 1 px sao descartadas

ap = argparse.ArgumentParser()
ap.add_argument("origem", help="pasta do BDD100K original, ex.: C:\\bdd100k")
ap.add_argument("destino", help="pasta nova a ser criada, ex.: C:\\bdd_tcc")
ap.add_argument("--val-frac", type=float, default=0.15)
ap.add_argument("--semente", type=int, default=0)
args = ap.parse_args()

origem, destino = Path(args.origem), Path(args.destino)
if destino.exists() and any(destino.iterdir()):
    sys.exit(f"ERRO: {destino} ja existe e nao esta vazia. Apague-a ou escolha outro nome.")


def selecionar(grupo_oficial):
    """Le as anotacoes de um grupo oficial e devolve as imagens 'clear' + 'daytime'."""
    pasta = origem / "labels/100k" / grupo_oficial
    arquivos = sorted(pasta.glob("*.json"))
    if not arquivos:
        sys.exit(f"ERRO: nenhum .json em {pasta}")
    itens = []
    print(f"Lendo {len(arquivos)} anotacoes de '{grupo_oficial}'...", flush=True)
    for i, arq in enumerate(arquivos):
        d = json.loads(arq.read_text(encoding="utf-8"))
        a = d.get("attributes") or {}
        if a.get("weather") == "clear" and a.get("timeofday") == "daytime":
            objs = [o for f in d.get("frames", []) for o in f.get("objects", []) if "box2d" in o]
            itens.append({"stem": arq.stem, "oficial": grupo_oficial, "objs": objs})
        if (i + 1) % 10000 == 0:
            print(f"   ... {i + 1}", flush=True)
    print(f"   selecionadas: {len(itens)}")
    return itens


teste = selecionar("val")
candidatos = selecionar("train")

# ---------- divisao treino/validacao por grupos de prefixo ----------
grupos = defaultdict(list)
for it in candidatos:
    grupos[it["stem"].split("-")[0]].append(it)
chaves = sorted(grupos)
random.Random(args.semente).shuffle(chaves)
alvo, val, treino = args.val_frac * len(candidatos), [], []
for k in chaves:
    (val if len(val) < alvo else treino).extend(grupos[k])
prefixos_teste = {it["stem"].split("-")[0] for it in teste}
sobrepostos = sum(it["stem"].split("-")[0] in prefixos_teste for it in treino + val)

divisao = {"train": sorted(treino, key=lambda x: x["stem"]),
           "val": sorted(val, key=lambda x: x["stem"]),
           "test": sorted(teste, key=lambda x: x["stem"])}

# ---------- gravacao ----------
resumo = {}
for grupo, itens in divisao.items():
    dir_img, dir_lbl = destino / "images" / grupo, destino / "labels" / grupo
    dir_img.mkdir(parents=True)
    dir_lbl.mkdir(parents=True)
    coco = {"images": [], "annotations": [],
            "categories": [{"id": i + 1, "name": c} for i, c in enumerate(CLASSES)]}
    cont, fundo, descartadas_classe, descartadas_tam, lista = Counter(), 0, 0, 0, []
    print(f"Gravando '{grupo}' ({len(itens)} imagens)...", flush=True)
    for n, it in enumerate(itens, start=1):
        src = origem / "images/100k" / it["oficial"] / (it["stem"] + ".jpg")
        dst = dir_img / src.name
        shutil.copy2(src, dst)
        with Image.open(dst) as im:
            W, H = im.size
        coco["images"].append({"id": n, "file_name": src.name, "width": W, "height": H})
        linhas = []
        for o in it["objs"]:
            if o.get("category") not in CLASSES:
                descartadas_classe += 1
                continue
            c = CLASSES.index(o["category"])
            b = o["box2d"]
            x1, y1 = max(0.0, min(b["x1"], b["x2"])), max(0.0, min(b["y1"], b["y2"]))
            x2, y2 = min(float(W), max(b["x1"], b["x2"])), min(float(H), max(b["y1"], b["y2"]))
            w, h = x2 - x1, y2 - y1
            if w <= MIN_LADO or h <= MIN_LADO:
                descartadas_tam += 1
                continue
            linhas.append(f"{c} {(x1 + w / 2) / W:.6f} {(y1 + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
            coco["annotations"].append({
                "id": len(coco["annotations"]) + 1, "image_id": n, "category_id": c + 1,
                "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                "area": round(w * h, 2), "iscrowd": 0})
            cont[CLASSES[c]] += 1
        (dir_lbl / (it["stem"] + ".txt")).write_text("\n".join(linhas) + ("\n" if linhas else ""))
        fundo += not linhas
        lista.append(str(dst.resolve()))
        if n % 2000 == 0:
            print(f"   ... {n}", flush=True)
    (destino / f"{grupo}.txt").write_text("\n".join(lista) + "\n", encoding="utf-8")
    (destino / f"instances_{grupo}.json").write_text(json.dumps(coco))
    resumo[grupo] = {"imagens": len(itens), "imagens_sem_objeto": fundo,
                     "objetos": {c: cont[c] for c in CLASSES},
                     "caixas_de_outras_classes_descartadas": descartadas_classe,
                     "caixas_degeneradas_descartadas": descartadas_tam}

# ---------- data.yaml (Ultralytics) ----------
yaml = [f"path: {destino.resolve().as_posix()}", "train: images/train", "val: images/val",
        "test: images/test", "names:"] + [f"  {i}: {c}" for i, c in enumerate(CLASSES)]
(destino / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")

# ---------- amostras de conferencia (lidas dos .txt convertidos) ----------
dir_am = destino / "amostras"
dir_am.mkdir()
rng = random.Random(args.semente)
for it in rng.sample(divisao["train"], min(12, len(divisao["train"]))):
    img = Image.open(destino / "images/train" / (it["stem"] + ".jpg")).convert("RGB")
    W, H = img.size
    des = ImageDraw.Draw(img)
    for linha in (destino / "labels/train" / (it["stem"] + ".txt")).read_text().splitlines():
        c, cx, cy, w, h = linha.split()
        c, cx, cy, w, h = int(c), float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
        des.rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], outline=CORES[c], width=2)
        des.text((cx - w / 2 + 2, max(0, cy - h / 2 - 11)), CLASSES[c], fill=CORES[c])
    img.save(dir_am / (it["stem"] + ".jpg"), quality=92)

# ---------- resumo ----------
total = sum(r["imagens"] for r in resumo.values())
txt = [f"Dataset do TCC gerado em {datetime.now():%d/%m/%Y %H:%M}",
       f"Origem: {origem}  |  Destino: {destino}",
       "Filtro: weather == 'clear' e timeofday == 'daytime'",
       f"Classes: {CLASSES} (ids YOLO a partir de 0; ids COCO a partir de 1, na mesma ordem)",
       f"Divisao: teste = validacao oficial filtrada; validacao = {args.val_frac:.0%} do treino "
       f"oficial filtrado, por grupos de prefixo do nome; semente = {args.semente}", ""]
for g, r in resumo.items():
    txt.append(f"{g}: {r['imagens']} imagens ({r['imagens'] / total:.1%}) | sem objeto: "
               f"{r['imagens_sem_objeto']} | objetos: {r['objetos']} | outras classes descartadas: "
               f"{r['caixas_de_outras_classes_descartadas']} | degeneradas: "
               f"{r['caixas_degeneradas_descartadas']}")
txt += ["", f"Total: {total} imagens",
        f"Imagens de treino/validacao cujo prefixo tambem aparece no teste: {sobrepostos}"]
(destino / "resumo.txt").write_text("\n".join(txt) + "\n", encoding="utf-8")
(destino / "resumo.json").write_text(json.dumps(resumo, indent=2))
print("\n" + "\n".join(txt))
print(f"\nPronto. Confira a pasta {dir_am} e envie o conteudo de resumo.txt.")
