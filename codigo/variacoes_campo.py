"""Roda o ensaio de campo em várias combinações de formato de entrada e monta uma tabela comparativa.

Combinações (recorte x imgsz):
    nenhum   640   -> protocolo principal (o vídeo vertical inteiro, como nos resultados já obtidos)
    paisagem 640   -> faixa 16:9 de largura total, na mesma posição em todos os vídeos (formato das imagens
                      do BDD100K); na primeira rodada abre uma janela para você posicionar a faixa com o mouse
    nenhum   960 / 1280, paisagem 960 / 1280 -> entrada maior (alvo maior na rede; mais lento)
Os limiares de confiança continuam os da validação (definidos em 640); as variações são ANÁLISE
COMPLEMENTAR, não substituem o resultado principal.

As caixas anotadas são reaproveitadas de campo_resultados\\anotacoes.json (nada é desenhado de novo).

Uso (venv ativado, na pasta C:\\tcc, com analisar_campo.py, testar_fotos.py e treinar_frcnn.py):
    python variacoes_campo.py --videos C:\\tcc\\campo\\videos
    python variacoes_campo.py --videos C:\\tcc\\campo\\videos --combinacoes paisagem:640 nenhum:960

Saída: campo_variacoes\\<recorte>_<imgsz>\\ (saídas completas de cada rodada) e
       campo_variacoes\\comparativo.csv / comparativo.txt (taxa por distância, modelo e combinação).
Rodadas já concluídas (com resumo.txt) são puladas; apague a pasta da rodada para refazer.
"""
import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

PADRAO = ["nenhum:640", "paisagem:640", "nenhum:960", "paisagem:960", "nenhum:1280", "paisagem:1280"]
MODELOS = ["RT-DETR-L", "YOLO26m", "Faster R-CNN v2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--anotacoes", default=str(Path("campo_resultados") / "anotacoes.json"))
    ap.add_argument("--saida", default="campo_variacoes")
    ap.add_argument("--ignorar-inicio", default="5")
    ap.add_argument("--passo", default="3")
    ap.add_argument("--combinacoes", nargs="+", default=PADRAO, help="recorte:imgsz, ex.: paisagem:640")
    args = ap.parse_args()
    if not Path(args.anotacoes).exists():
        raise SystemExit(f"Anotações não encontradas: {Path(args.anotacoes).resolve()}")
    base = Path(args.saida)
    base.mkdir(exist_ok=True)

    linhas, tempos = [], {}
    for comb in args.combinacoes:
        recorte, imgsz = comb.split(":")
        pasta = base / f"{recorte}_{imgsz}"
        if not (pasta / "resumo.txt").exists():
            print(f"\n===== {recorte}, imgsz {imgsz} =====", flush=True)
            t0 = time.time()
            subprocess.run([sys.executable, "analisar_campo.py", "--videos", args.videos, "--saida", str(pasta),
                            "--anotacoes", args.anotacoes, "--recorte", recorte, "--imgsz", imgsz,
                            "--ignorar-inicio", args.ignorar_inicio, "--passo", args.passo], check=True)
            tempos[comb] = time.time() - t0
        with open(pasta / "taxa_por_distancia.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                linhas.append({"recorte": recorte, "imgsz": int(imgsz), "distancia_m": float(r["distancia_m"]),
                               "alvo_na_rede_px": r["altura_alvo_na_rede_px"],
                               "alvo_na_rede_rtdetr_px": r.get("altura_alvo_na_rede_rtdetr_px", ""),
                               **{m: float(r[f"taxa_{m}"]) for m in MODELOS}})

    with open(base / "comparativo.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)
    txt = ["COMPARATIVO DE FORMATOS DE ENTRADA (taxa de detecção; limiares da validação; "
           f"{args.ignorar_inicio} s iniciais descartados)", ""]
    for m in MODELOS:
        txt.append(m)
        dists = sorted({l["distancia_m"] for l in linhas})
        txt.append("   combinação       " + "".join(f"{d:>7g} m" for d in dists))
        for comb in args.combinacoes:
            rc, im = comb.split(":")
            v = {l["distancia_m"]: l[m] for l in linhas if l["recorte"] == rc and l["imgsz"] == int(im)}
            txt.append(f"   {rc:>8} {im:>5}    " + "".join(f"{v[d]:>8.0%} " for d in dists))
        txt.append("")
    txt.append("Altura do alvo na rede (px), YOLO26m e Faster R-CNN v2 | RT-DETR-L (que estica a imagem para um quadrado):")
    for comb in args.combinacoes:
        rc, im = comb.split(":")
        v = [l for l in linhas if l["recorte"] == rc and l["imgsz"] == int(im)]
        txt.append(f"   {rc:>8} {im:>5}    " + "  ".join(f"{l['distancia_m']:g} m: {l['alvo_na_rede_px']} | "
                                                          f"{l['alvo_na_rede_rtdetr_px']}" for l in v))
    if tempos:
        txt += ["", "Tempo de cada rodada nesta execução (inclui leitura dos vídeos; não é a latência do modelo):"]
        txt += [f"   {c}: {t / 60:.1f} min" for c, t in tempos.items()]
    (base / "comparativo.txt").write_text("\n".join(txt) + "\n", encoding="utf-8")
    print("\n" + "\n".join(txt))


if __name__ == "__main__":
    main()
