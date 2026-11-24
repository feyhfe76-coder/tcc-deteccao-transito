"""Analises complementares A, B e C, feitas sobre as previsoes ja salvas (sem rodar os modelos).

  A) Ponto de operacao de cada modelo: limiar de confianca de maximo F1 medio entre classes,
     escolhido na VALIDACAO (mesmo criterio da Ultralytics) e aplicado no TESTE.
     Gera a matriz de confusao, P, R, F1 e falsos positivos por imagem no limiar de cada
     modelo e, para comparacao, no limiar unico 0,25 (com o mesmo IoU).
  B) Latencia -> distancia percorrida pelo veiculo durante o processamento (d = v * t),
     para 60 e 100 km/h, com a mediana e o p95, e comparacao com o intervalo entre quadros
     de uma camera de 30 FPS (33,3 ms).
  C) Bootstrap pareado refeito com a mesma semente e os mesmos sorteios do avaliar_teste.py:
     confere se os intervalos batem com diferencas.csv e calcula P(A > B), a fracao dos
     sorteios em que o modelo A superou o B. Salva os sorteios e um histograma das diferencas.

Uso (venv ativado, na pasta C:\\tcc, com avaliar_teste.py e treinar_frcnn.py na mesma pasta):
    python analise_abc.py
    python analise_abc.py --iou-matriz 0.45                       (outro IoU na matriz)
    python analise_abc.py --teste avaliacao_test_s1 --saida analise_abc_s1   (semente 1)

Entradas: avaliacao_val\\previsoes_*.pkl, avaliacao_test\\previsoes_*.pkl, diferencas.csv e
latencia.csv do teste, e os rotulos em C:\\bdd_tcc (via --data).
Saida em C:\\tcc\\analise_abc\\ : resumo.txt, limiares.csv, operacao_teste.csv,
operacao_por_classe.csv, confusao_<modelo>_limiar.csv/.png, latencia_distancia.csv,
bootstrap_prob.csv, bootstrap_sorteios.csv, bootstrap_diferencas.png.
"""
import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
import yaml
from ultralytics.utils.metrics import ap_per_class, smooth

from avaliar_teste import (carregar_gt, estatisticas_por_imagem, map_rapido, matriz_confusao,
                           salvar_confusao)

MODELOS = {"RT-DETR-L": "rtdetr_l", "YOLO26m": "yolo26m", "Faster R-CNN v2": "frcnn_r50v2"}


def carregar_preds(pasta, slug):
    arq = Path(pasta) / f"previsoes_{slug}.pkl"
    if not arq.exists():
        raise SystemExit(f"ERRO: nao encontrei {arq.resolve()}\n"
                         f"Rode antes o avaliar_teste.py para gerar as previsoes dessa pasta.")
    return pickle.loads(arq.read_bytes())


def limiar_max_f1(lista):
    """Limiar de confianca de maximo F1 medio entre classes, IoU 0,50 (criterio da Ultralytics)."""
    tp = np.concatenate([x[0] for x in lista])
    conf = np.concatenate([x[1] for x in lista])
    pc = np.concatenate([x[2] for x in lista])
    gc = np.concatenate([x[3] for x in lista])
    _, _, p, r, f1, _, _, _, _, f1_curve, x, _ = ap_per_class(tp, conf, pc, gc)
    i = int(smooth(f1_curve.mean(0), 0.1).argmax())
    return float(x[i]), float(p.mean()), float(r.mean()), float(f1.mean())


def operacao(M, nc, n_img):
    """Contagens a partir da matriz de confusao (linhas = previsto, colunas = real, ultima = fundo)."""
    vp = int(np.trace(M[:nc, :nc]))
    previstas = int(M[:nc, :].sum())
    reais = int(M[:, :nc].sum())
    fp_fundo = int(M[:nc, nc].sum())          # caixa onde nao ha objeto anotado
    troca = int(M[:nc, :nc].sum()) - vp       # caixa no lugar certo, classe errada
    fn = int(M[nc, :nc].sum())                # objeto anotado sem caixa
    P = vp / previstas if previstas else 0.0
    R = vp / reais if reais else 0.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0
    return {"VP": vp, "FP_fundo": fp_fundo, "troca_classe": troca, "FN": fn,
            "previstas": previstas, "objetos": reais, "P": P, "R": R, "F1": F1,
            "FP_por_imagem": (previstas - vp) / n_img}


def secao_a(args, nomes, gt_val, gt_te, saida, texto):
    nc, n_img = len(nomes), len(gt_te)
    texto += [f"=== A) Ponto de operacao (limiar escolhido na validacao; matriz com IoU >= "
              f"{args.iou_matriz:.2f}) ===",
              f"{'modelo':>16}{'limiar':>8}{'F1 val':>8} | {'conf':>6}{'P':>7}{'R':>7}{'F1':>7}"
              f"{'FP/img':>8}{'FP fundo':>10}{'troca':>7}{'FN':>7}"]
    lim_rows, op_rows, cls_rows = [], [], []
    for nome, slug in MODELOS.items():
        lim, pv, rv, fv = limiar_max_f1(estatisticas_por_imagem(carregar_preds(args.val, slug), gt_val))
        lim_rows.append([nome, round(lim, 4), round(pv, 4), round(rv, 4), round(fv, 4)])
        preds_te = carregar_preds(args.teste, slug)
        for rot, conf in (("limiar do modelo", lim), ("0,25 (unico)", 0.25)):
            M = matriz_confusao(preds_te, gt_te, nc, conf=conf, iou_min=args.iou_matriz)
            o = operacao(M, nc, n_img)
            op_rows.append([nome, rot, round(conf, 4)] + [round(v, 4) if isinstance(v, float) else v
                                                          for v in o.values()])
            texto.append(f"{nome:>16}{lim:>8.3f}{fv:>8.3f} | {conf:>6.3f}{o['P']:>7.3f}{o['R']:>7.3f}"
                         f"{o['F1']:>7.3f}{o['FP_por_imagem']:>8.2f}{o['FP_fundo']:>10}"
                         f"{o['troca_classe']:>7}{o['FN']:>7}   ({rot})")
            if rot == "limiar do modelo":
                salvar_confusao(M, nomes, f"{nome} (teste, conf >= {conf:.2f})",
                                saida / f"confusao_{slug}_limiar")
                for k, cn in enumerate(nomes):
                    vp, prev, real = M[k, k], M[k, :].sum(), M[:, k].sum()
                    cls_rows.append([nome, cn, round(conf, 4), int(real), int(vp),
                                     round(vp / prev, 4) if prev else 0.0,
                                     round(vp / real, 4) if real else 0.0])
    with open(saida / "limiares.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["modelo", "limiar_max_F1_val", "P_val", "R_val", "F1_val"])
        w.writerows(lim_rows)
    with open(saida / "operacao_teste.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["modelo", "limiar_usado", "conf", "VP", "FP_fundo", "troca_classe", "FN",
                    "previstas", "objetos", "P", "R", "F1", "FP_por_imagem"])
        w.writerows(op_rows)
    with open(saida / "operacao_por_classe.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["modelo", "classe", "conf", "objetos", "VP", "P", "R"])
        w.writerows(cls_rows)
    texto += ["  P, R e F1 contam como acerto so a caixa com a classe certa; 'troca' = caixa sobre um",
              "  objeto anotado mas com classe errada; 'FP fundo' = caixa onde nao ha objeto anotado.", ""]


def secao_b(args, saida, texto):
    arq = Path(args.teste) / "latencia.csv"
    if not arq.exists():
        texto += [f"=== B) Latencia -> distancia: {arq} nao existe (avaliacao feita sem latencia) ===", ""]
        return
    quadro = 1000 / 30
    texto += ["=== B) Distancia percorrida durante o processamento (d = v x t) ===",
              f"{'modelo':>16}{'prec.':>6}{'mediana':>9}{'p95':>8} |"
              + "".join(f"{f'{v} km/h med':>14}{'p95':>7}" for v in args.velocidades)
              + f"  > {quadro:.1f} ms?"]
    linhas = []
    with open(arq, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            med, p95 = float(r["mediana_ms"]), float(r["p95_ms"])
            dist = []
            for v in args.velocidades:
                ms = v / 3.6
                dist += [ms * med / 1000, ms * p95 / 1000]
            excede = "sim" if med > quadro else "nao"
            linhas.append([r["modelo"], r["precisao"], round(med, 2), round(p95, 2)]
                          + [round(d, 3) for d in dist] + [excede])
            texto.append(f"{r['modelo']:>16}{r['precisao']:>6}{med:>9.1f}{p95:>8.1f} |"
                         + "".join(f"{dist[2 * i]:>12.2f} m{dist[2 * i + 1]:>6.2f}"
                                   for i in range(len(args.velocidades)))
                         + f"  {excede}")
    with open(saida / "latencia_distancia.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        cab = ["modelo", "precisao", "mediana_ms", "p95_ms"]
        for v in args.velocidades:
            cab += [f"dist_{v}kmh_mediana_m", f"dist_{v}kmh_p95_m"]
        w.writerow(cab + [f"mediana_acima_de_{quadro:.1f}ms"])
        w.writerows(linhas)
    texto += [f"  Ultima coluna: a mediana passa do intervalo entre quadros de uma camera de 30 FPS",
              "  (o modelo nao processaria todos os quadros nesse hardware). So inferencia: nao inclui",
              "  captura, transmissao, decisao e atuacao.", ""]


def secao_c(args, gt_te, saida, texto):
    listas = {nome: estatisticas_por_imagem(carregar_preds(args.teste, slug), gt_te)
              for nome, slug in MODELOS.items()}
    rng = np.random.default_rng(0)  # mesma semente e mesma ordem de sorteio do avaliar_teste.py
    n = len(gt_te)
    sorteios = [rng.integers(0, n, n) for _ in range(args.bootstrap)]
    amostras = {nome: np.array([map_rapido(listas[nome], idx) for idx in sorteios]) for nome in MODELOS}

    oficial = {}
    arq = Path(args.teste) / "diferencas.csv"
    if arq.exists():
        with open(arq, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                oficial[(r["modelo_A"], r["modelo_B"], r["metrica"])] = (float(r["ic95_inf"]),
                                                                         float(r["ic95_sup"]))
    texto += [f"=== C) Bootstrap pareado ({args.bootstrap} sorteios, semente 0) ===",
              f"{'A - B':>34}{'metrica':>10}{'IC95 inf':>10}{'IC95 sup':>10}{'P(A>B)':>9}  confere?"]
    linhas, nomes_m = [], list(MODELOS)
    for i in range(3):
        for j in range(i + 1, 3):
            A, B = nomes_m[i], nomes_m[j]
            for k, rot in ((0, "mAP50-95"), (1, "mAP50")):
                d = amostras[A][:, k] - amostras[B][:, k]
                a, b = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
                prob = float((d > 0).mean())
                if (A, B, rot) in oficial:
                    ok = "sim" if np.allclose((a, b), oficial[(A, B, rot)], atol=1e-6) else "NAO"
                else:
                    ok = "sem arquivo"
                linhas.append([A, B, rot, a, b, prob, int((d > 0).sum()), args.bootstrap, ok])
                texto.append(f"{A + ' - ' + B:>34}{rot:>10}{a:>+10.4f}{b:>+10.4f}{prob:>9.3f}  {ok}")
    with open(saida / "bootstrap_prob.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["modelo_A", "modelo_B", "metrica", "ic95_inf", "ic95_sup", "P_A_maior_B",
                    "sorteios_A_maior_B", "sorteios", "confere_com_diferencas_csv"])
        w.writerows(linhas)
    with open(saida / "bootstrap_sorteios.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sorteio"] + [f"{m} {r}" for m in MODELOS for r in ("mAP50-95", "mAP50")])
        for s in range(args.bootstrap):
            w.writerow([s] + [round(float(amostras[m][s, k]), 6) for m in MODELOS for k in (0, 1)])
    texto += [f"  P(A>B) = fracao dos {args.bootstrap} sorteios em que A superou B. Com {args.bootstrap}"
              f" sorteios, 1,000 significa apenas 'acima de ~{1 - 1 / args.bootstrap:.3f}'.",
              "  'confere?' compara os intervalos com os do diferencas.csv (devem ser iguais).", ""]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, j in ((0, 1), (1, 2), (0, 2)):
        A, B = nomes_m[i], nomes_m[j]
        ax.hist(amostras[A][:, 0] - amostras[B][:, 0], bins=30, alpha=0.6, label=f"{A} - {B}")
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Diferenca de mAP50-95 entre modelos (teste, reamostragens bootstrap)")
    ax.set_ylabel("Numero de sorteios")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(saida / "bootstrap_diferencas.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"C:\bdd_tcc\data.yaml")
    ap.add_argument("--val", default="avaliacao_val", help="pasta com as previsoes da validacao")
    ap.add_argument("--teste", default="avaliacao_test", help="pasta com as previsoes do teste")
    ap.add_argument("--saida", default="analise_abc")
    ap.add_argument("--iou-matriz", type=float, default=0.50)
    ap.add_argument("--bootstrap", type=int, default=200, help="use o mesmo valor do avaliar_teste.py")
    ap.add_argument("--velocidades", type=float, nargs="+", default=[60, 100], help="km/h")
    ap.add_argument("--so", default="ABC", help="quais analises rodar, por exemplo --so A ou --so BC")
    args = ap.parse_args()

    raiz = Path(args.data).parent
    nomes = list(yaml.safe_load(Path(args.data).read_text())["names"].values())
    saida = Path(args.saida)
    saida.mkdir(exist_ok=True)
    print("Lendo rotulos do teste...", flush=True)
    gt_te = carregar_gt(raiz, "test")
    texto = [f"ANALISES A/B/C | validacao: {args.val} | teste: {args.teste} ({len(gt_te)} imagens)", ""]
    so = args.so.upper()
    if "A" in so:
        print("Lendo rotulos da validacao...", flush=True)
        gt_val = carregar_gt(raiz, "val")
        print("A) ponto de operacao...", flush=True)
        secao_a(args, nomes, gt_val, gt_te, saida, texto)
    if "B" in so:
        print("B) latencia -> distancia...", flush=True)
        secao_b(args, saida, texto)
    if "C" in so:
        print(f"C) bootstrap com {args.bootstrap} sorteios (alguns minutos)...", flush=True)
        secao_c(args, gt_te, saida, texto)
    (saida / "resumo.txt").write_text("\n".join(texto) + "\n", encoding="utf-8")
    print("\n" + "\n".join(texto))
    print(f"Resultados salvos em {saida.resolve()}")


if __name__ == "__main__":
    main()
