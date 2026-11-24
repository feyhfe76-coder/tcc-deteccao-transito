"""Etapa E: monta conjuntos de avaliacao por condicao (noite, chuva...) a partir da VALIDACAO
OFICIAL do BDD100K, com as mesmas regras do conjunto do TCC (5 classes, mesmo descarte de caixas).

Nenhuma dessas imagens foi usada no treino nem na escolha de epoca (o treino veio do treino
oficial; o teste do TCC e a parte 'clear' + 'daytime' desta mesma validacao oficial).

Uso (venv ativado, na pasta C:\\tcc):
    python preparar_condicoes.py C:\\bdd100k C:\\bdd_condicoes

O que faz:
  1. Le os 10.000 .json da validacao oficial e mostra a tabela clima x periodo do dia.
  2. CONFERENCIA: recalcula o grupo 'clear' + 'daytime' e compara com o teste do TCC
     (deve dar 1.764 imagens e 29.823 objetos). Esse grupo nao e copiado.
  3. Para cada grupo com pelo menos --min-imagens imagens, cria C:\\bdd_condicoes\\<grupo>\\ com
     images\\test, labels\\test (formato YOLO) e data.yaml, prontos para o avaliar_teste.py:
        python avaliar_teste.py --data C:\\bdd_condicoes\\<grupo>\\data.yaml --sem-latencia --saida avaliacao_<grupo>
  4. Grava C:\\bdd_condicoes\\resumo.txt com as contagens.

O dataset original nao e alterado.
"""
import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

CLASSES = ["car", "truck", "bus", "person", "traffic sign"]
MIN_LADO = 1.0  # igual ao preparar_dataset.py

# grupo -> (descricao, regra sobre (clima, periodo))
GRUPOS = {
    "nublado_dia": ("dia, nublado ou parcialmente nublado",
                    lambda w, t: t == "daytime" and w in ("overcast", "partly cloudy")),
    "chuva_dia": ("dia, chuva", lambda w, t: t == "daytime" and w == "rainy"),
    "neve_dia": ("dia, neve", lambda w, t: t == "daytime" and w == "snowy"),
    "neblina": ("neblina, qualquer periodo", lambda w, t: w == "foggy"),
    "noite_limpo": ("noite, tempo limpo", lambda w, t: t == "night" and w == "clear"),
    "noite_chuva": ("noite, chuva", lambda w, t: t == "night" and w == "rainy"),
    "amanhecer_anoitecer": ("amanhecer/anoitecer, qualquer clima", lambda w, t: t == "dawn/dusk"),
}
REFERENCIA = lambda w, t: w == "clear" and t == "daytime"  # o teste do TCC


def caixas_validas(objs, W, H):
    """Mesmas regras do preparar_dataset.py: so as 5 classes, recorte na imagem, descarta <= 1 px."""
    linhas, cont, peq_med_gra = [], Counter(), Counter()
    for o in objs:
        if o.get("category") not in CLASSES:
            continue
        c = CLASSES.index(o["category"])
        b = o["box2d"]
        x1, y1 = max(0.0, min(b["x1"], b["x2"])), max(0.0, min(b["y1"], b["y2"]))
        x2, y2 = min(float(W), max(b["x1"], b["x2"])), min(float(H), max(b["y1"], b["y2"]))
        w, h = x2 - x1, y2 - y1
        if w <= MIN_LADO or h <= MIN_LADO:
            continue
        linhas.append(f"{c} {(x1 + w / 2) / W:.6f} {(y1 + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
        cont[CLASSES[c]] += 1
        a = w * h
        peq_med_gra["pequenos" if a < 32 ** 2 else "medios" if a < 96 ** 2 else "grandes"] += 1
    return linhas, cont, peq_med_gra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("origem", help="pasta do BDD100K original, ex.: C:\\bdd100k")
    ap.add_argument("destino", help="pasta nova, ex.: C:\\bdd_condicoes")
    ap.add_argument("--min-imagens", type=int, default=100,
                    help="grupos com menos imagens sao listados mas nao criados")
    args = ap.parse_args()
    origem, destino = Path(args.origem), Path(args.destino)
    if destino.exists() and any(destino.iterdir()):
        sys.exit(f"ERRO: {destino} ja existe e nao esta vazia. Apague-a ou escolha outro nome.")

    pasta = origem / "labels/100k/val"
    arquivos = sorted(pasta.glob("*.json"))
    if not arquivos:
        sys.exit(f"ERRO: nenhum .json em {pasta}")
    print(f"Lendo {len(arquivos)} anotacoes da validacao oficial...", flush=True)
    itens, tabela = [], Counter()
    for arq in arquivos:
        d = json.loads(arq.read_text(encoding="utf-8"))
        a = d.get("attributes") or {}
        w, t = a.get("weather", "?"), a.get("timeofday", "?")
        tabela[(w, t)] += 1
        objs = [o for f in d.get("frames", []) for o in f.get("objects", []) if "box2d" in o]
        itens.append({"stem": arq.stem, "clima": w, "periodo": t, "objs": objs})

    climas = sorted({w for w, _ in tabela})
    periodos = sorted({t for _, t in tabela})
    txt = ["Validacao oficial do BDD100K: imagens por clima (linhas) x periodo do dia (colunas)",
           f"{'':>16}" + "".join(f"{p:>12}" for p in periodos) + f"{'total':>9}"]
    for w in climas:
        txt.append(f"{w:>16}" + "".join(f"{tabela[(w, p)]:>12}" for p in periodos)
                   + f"{sum(tabela[(w, p)] for p in periodos):>9}")
    txt.append("")

    # ---------- conferencia com o teste do TCC (sem copiar) ----------
    ref = [it for it in itens if REFERENCIA(it["clima"], it["periodo"])]
    n_obj = 0
    for it in ref:
        with Image.open(origem / "images/100k/val" / (it["stem"] + ".jpg")) as im:
            W, H = im.size
        n_obj += len(caixas_validas(it["objs"], W, H)[0])
    ok = "CONFERE" if (len(ref), n_obj) == (1764, 29823) else "NAO CONFERE - me avise antes de seguir"
    txt += [f"Conferencia 'clear' + 'daytime': {len(ref)} imagens, {n_obj} objetos "
            f"(teste do TCC: 1764 imagens, 29823 objetos) -> {ok}", ""]
    print("\n".join(txt), flush=True)

    # ---------- grupos ----------
    txt.append(f"{'grupo':>22}{'imagens':>9}{'objetos':>9}{'pequenos':>10}{'medios':>8}{'grandes':>9}"
               f"  {'objetos por classe'}")
    comandos = []
    for g, (desc, regra) in GRUPOS.items():
        sel = [it for it in itens if regra(it["clima"], it["periodo"])]
        if len(sel) < args.min_imagens:
            txt.append(f"{g:>22}{len(sel):>9}   (menos de {args.min_imagens} imagens: nao criado) - {desc}")
            continue
        dir_img, dir_lbl = destino / g / "images" / "test", destino / g / "labels" / "test"
        dir_img.mkdir(parents=True)
        dir_lbl.mkdir(parents=True)
        cont, tam = Counter(), Counter()
        print(f"Gravando '{g}' ({len(sel)} imagens)...", flush=True)
        for it in sel:
            src = origem / "images/100k/val" / (it["stem"] + ".jpg")
            shutil.copy2(src, dir_img / src.name)
            with Image.open(src) as im:
                W, H = im.size
            linhas, c, s = caixas_validas(it["objs"], W, H)
            cont.update(c)
            tam.update(s)
            (dir_lbl / (it["stem"] + ".txt")).write_text("\n".join(linhas) + ("\n" if linhas else ""))
        yaml = [f"path: {(destino / g).resolve().as_posix()}", "train: images/test", "val: images/test",
                "test: images/test", "names:"] + [f"  {i}: {c}" for i, c in enumerate(CLASSES)]
        (destino / g / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
        txt.append(f"{g:>22}{len(sel):>9}{sum(cont.values()):>9}{tam['pequenos']:>10}{tam['medios']:>8}"
                   f"{tam['grandes']:>9}  {dict(cont)} - {desc}")
        comandos.append(f"python avaliar_teste.py --data {(destino / g / 'data.yaml').resolve()} "
                        f"--sem-latencia --saida avaliacao_{g}")
    txt += ["", "Comandos de avaliacao (um por grupo):"] + comandos
    destino.mkdir(exist_ok=True)
    (destino / "resumo.txt").write_text("\n".join(txt) + "\n", encoding="utf-8")
    print("\n" + "\n".join(txt))
    print(f"\nPronto. Envie o conteudo de {destino / 'resumo.txt'}")


if __name__ == "__main__":
    main()
