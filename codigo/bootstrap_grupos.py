"""Bootstrap pareado refeito com mais sorteios e por GRUPO de video (prefixo do nome da imagem).

Por que: o bootstrap original sorteava imagens individuais (200 sorteios). Imagens com o mesmo
prefixo no nome podem vir da mesma sessao de gravacao e ser parecidas entre si; sortea-las
como se fossem independentes pode deixar os intervalos de confianca estreitos demais.
Aqui o sorteio e feito por grupo de prefixo (o mesmo criterio usado para separar treino e
validacao) e com 2.000 sorteios. Para comparacao, tambem refaz o sorteio por imagem com 2.000.

Uso (venv ativado, na pasta C:\\tcc, com avaliar_teste.py e treinar_frcnn.py na mesma pasta):
    python bootstrap_grupos.py --bdd C:\\bdd100k
    python bootstrap_grupos.py --teste avaliacao_noite_limpo --data C:\\bdd_condicoes\\noite_limpo\\data.yaml --saida bootstrap_noite_limpo

--bdd (opcional, so para o teste principal): confere se os NOMES das imagens do teste sao
exatamente as imagens 'clear' + 'daytime' da validacao oficial do BDD100K.

Saida em C:\\tcc\\bootstrap_grupos\\ : resumo.txt, intervalos.csv, diferencas.csv
Leva de alguns minutos a cerca de meia hora (usa varios nucleos do processador; nao usa a GPU).
"""
import argparse
import csv
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

MODELOS = {"RT-DETR-L": "rtdetr_l", "YOLO26m": "yolo26m", "Faster R-CNN v2": "frcnn_r50v2"}
_LISTAS = None


def _inicia(listas):
    global _LISTAS
    _LISTAS = listas


def _avalia(idx):
    from avaliar_teste import map_rapido
    return [map_rapido(_LISTAS[m], idx) for m in MODELOS]


def conferir_nomes(bdd, nomes_teste, texto):
    sel = set()
    for arq in (Path(bdd) / "labels/100k/val").glob("*.json"):
        a = json.loads(arq.read_text(encoding="utf-8")).get("attributes") or {}
        if a.get("weather") == "clear" and a.get("timeofday") == "daytime":
            sel.add(arq.stem)
    teste = {Path(n).stem for n in nomes_teste}
    ok = sel == teste
    texto += [f"Conferencia de nomes: {len(teste)} imagens no teste, {len(sel)} 'clear'+'daytime' na "
              f"validacao oficial, {len(sel & teste)} em comum -> "
              + ("IDENTICOS" if ok else f"DIFERENTES (so no teste: {len(teste - sel)}, "
                                        f"so no BDD: {len(sel - teste)})"), ""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"C:\bdd_tcc\data.yaml")
    ap.add_argument("--teste", default="avaliacao_test", help="pasta com as previsoes (.pkl)")
    ap.add_argument("--saida", default="bootstrap_grupos")
    ap.add_argument("--sorteios", type=int, default=2000)
    ap.add_argument("--bdd", default=None, help="pasta do BDD100K original, para conferir os nomes")
    ap.add_argument("--nucleos", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    from avaliar_teste import carregar_gt, estatisticas_por_imagem
    raiz = Path(args.data).parent
    saida = Path(args.saida)
    saida.mkdir(exist_ok=True)
    print("Lendo rotulos e previsoes...", flush=True)
    gt = carregar_gt(raiz, "test")
    listas = {}
    for m, slug in MODELOS.items():
        arq = Path(args.teste) / f"previsoes_{slug}.pkl"
        if not arq.exists():
            raise SystemExit(f"ERRO: nao encontrei {arq.resolve()}")
        listas[m] = estatisticas_por_imagem(pickle.loads(arq.read_bytes()), gt)

    nomes = list(gt)  # mesma ordem usada nas listas
    n = len(nomes)
    grupos = {}
    for i, nome in enumerate(nomes):
        grupos.setdefault(Path(nome).stem.split("-")[0], []).append(i)
    tam = np.array([len(v) for v in grupos.values()])
    texto = [f"BOOTSTRAP PAREADO | previsoes: {args.teste} | {n} imagens | {args.sorteios} sorteios", "",
             f"Grupos de prefixo (mesmo criterio da divisao treino/validacao): {len(grupos)} grupos | "
             f"imagens por grupo: media {tam.mean():.2f}, maximo {tam.max()} | "
             f"grupos com mais de 1 imagem: {(tam > 1).sum()} ({tam[tam > 1].sum()} imagens)", ""]
    if args.bdd:
        print("Conferindo nomes com o BDD100K original...", flush=True)
        conferir_nomes(args.bdd, nomes, texto)

    rng = np.random.default_rng(0)
    idx_grupos = list(grupos.values())
    sorteios = {
        "por imagem": [rng.integers(0, n, n) for _ in range(args.sorteios)],
        "por grupo": [np.concatenate([idx_grupos[g] for g in rng.integers(0, len(idx_grupos), len(idx_grupos))])
                      for _ in range(args.sorteios)],
    }
    amostras = {}
    with ProcessPoolExecutor(args.nucleos, initializer=_inicia, initargs=(listas,)) as ex:
        for modo, lista in sorteios.items():
            print(f"Sorteando {modo} ({args.sorteios} sorteios, {args.nucleos} nucleos)...", flush=True)
            res = []
            for k, r in enumerate(ex.map(_avalia, lista, chunksize=20), start=1):
                res.append(r)
                if k % 200 == 0:
                    print(f"   {k}/{args.sorteios}", flush=True)
            amostras[modo] = np.array(res)  # [sorteio, modelo, (mAP50-95, mAP50)]

    nm = list(MODELOS)
    ic = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    linhas_ic, linhas_d = [], []
    texto += ["=== Intervalos de confianca de 95% (mAP50-95) ===",
              f"{'modelo':>16}{'por imagem':>22}{'por grupo':>22}{'largura grupo/imagem':>24}"]
    for j, m in enumerate(nm):
        a = {modo: ic(amostras[modo][:, j, 0]) for modo in amostras}
        larg = (a["por grupo"][1] - a["por grupo"][0]) / (a["por imagem"][1] - a["por imagem"][0])
        texto.append(f"{m:>16}   [{a['por imagem'][0]:.4f}; {a['por imagem'][1]:.4f}]"
                     f"   [{a['por grupo'][0]:.4f}; {a['por grupo'][1]:.4f}]{larg:>18.2f}x")
        for modo in amostras:
            for k, rot in ((0, "mAP50-95"), (1, "mAP50")):
                lo, hi = ic(amostras[modo][:, j, k])
                linhas_ic.append([modo, m, rot, lo, hi])
    texto += ["", "=== Diferencas entre modelos (A - B), mAP50-95 ===",
              f"{'A - B':>32}{'modo':>12}{'IC95':>22}{'P(A>B)':>9}  conclusao"]
    for i in range(3):
        for j in range(i + 1, 3):
            for modo in amostras:
                for k, rot in ((0, "mAP50-95"), (1, "mAP50")):
                    d = amostras[modo][:, i, k] - amostras[modo][:, j, k]
                    lo, hi = ic(d)
                    p = float((d > 0).mean())
                    concl = "robusta" if lo > 0 or hi < 0 else "IC inclui zero"
                    linhas_d.append([nm[i], nm[j], modo, rot, lo, hi, p, concl])
                    if k == 0:
                        texto.append(f"{nm[i] + ' - ' + nm[j]:>32}{modo:>12}   [{lo:+.4f}; {hi:+.4f}]"
                                     f"{p:>9.3f}  {concl}")
    texto += ["", f"P(A>B) = fracao dos {args.sorteios} sorteios em que A superou B "
              f"(1,000 = em todos; resolucao de {1 / args.sorteios:.4f}).",
              "Largura grupo/imagem > 1 indica que o sorteio por grupo ampliou a incerteza."]

    with open(saida / "intervalos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["modo", "modelo", "metrica", "ic95_inf", "ic95_sup"])
        w.writerows(linhas_ic)
    with open(saida / "diferencas.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["modelo_A", "modelo_B", "modo", "metrica", "ic95_inf", "ic95_sup", "P_A_maior_B", "conclusao"])
        w.writerows(linhas_d)
    (saida / "resumo.txt").write_text("\n".join(texto) + "\n", encoding="utf-8")
    print("\n" + "\n".join(texto))
    print(f"\nSalvo em {saida.resolve()}")


if __name__ == "__main__":
    main()
