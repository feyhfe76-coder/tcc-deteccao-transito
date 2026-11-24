"""Junta os resultados da etapa E (uma pasta avaliacao_<grupo> por condicao) numa tabela unica,
com a queda de cada modelo em relacao ao teste principal (dia, tempo limpo).

Uso (na pasta C:\\tcc, depois de rodar o avaliar_teste.py em cada grupo):
    python tabela_condicoes.py
Saida: condicoes.csv e condicoes.txt na pasta C:\\tcc\\avaliacao_condicoes\\
"""
import csv
import json
from pathlib import Path

GRUPOS = [("referencia (dia, limpo)", "avaliacao_test"),
          ("dia, nublado", "avaliacao_nublado_dia"),
          ("dia, chuva", "avaliacao_chuva_dia"),
          ("dia, neve", "avaliacao_neve_dia"),
          ("amanhecer/anoitecer", "avaliacao_amanhecer_anoitecer"),
          ("noite, limpo", "avaliacao_noite_limpo"),
          ("noite, chuva", "avaliacao_noite_chuva")]
MODELOS = ["RT-DETR-L", "YOLO26m", "Faster R-CNN v2"]

saida = Path("avaliacao_condicoes")
saida.mkdir(exist_ok=True)
res = {}
for rot, pasta in GRUPOS:
    arq = Path(pasta) / "metricas.json"
    if arq.exists():
        res[rot] = json.loads(arq.read_text(encoding="utf-8"))
    else:
        print(f"AVISO: {arq} nao encontrado; grupo '{rot}' fica de fora")
if "referencia (dia, limpo)" not in res:
    raise SystemExit("ERRO: preciso de avaliacao_test\\metricas.json (teste principal) como referencia")
ref = {m: res["referencia (dia, limpo)"]["modelos"][m]["mAP50-95"] for m in MODELOS}

linhas, txt = [], ["mAP50-95 por condicao [IC 95%] e variacao em relacao ao dia limpo", ""]
cab = f"{'condicao':>24}{'imagens':>8}{'objetos':>8}" + "".join(f"{m:>30}" for m in MODELOS) + "   ordem"
txt.append(cab)
for rot, d in res.items():
    cel, ordem = [], sorted(MODELOS, key=lambda m: -d["modelos"][m]["mAP50-95"])
    for m in MODELOS:
        r = d["modelos"][m]
        v = r["mAP50-95"]
        ic = r.get("ic95_mAP50-95", [float("nan")] * 2)
        queda = (v - ref[m]) / ref[m] * 100
        cel.append(f"{v:.3f} [{ic[0]:.3f};{ic[1]:.3f}] {queda:+5.0f}%")
        linhas.append([rot, d["imagens"], d["objetos"], m, round(v, 4), round(ic[0], 4), round(ic[1], 4),
                       round(r["mAP50"], 4), round(r["coco"]["AP_pequenos"], 4), round(queda, 1)])
    txt.append(f"{rot:>24}{d['imagens']:>8}{d['objetos']:>8}" + "".join(f"{c:>30}" for c in cel)
               + "   " + " > ".join(ordem))
txt += ["", "Variacao (%) = (mAP da condicao - mAP no dia limpo) / mAP no dia limpo.",
        "Ordem = classificacao dos modelos pelo mAP50-95 naquela condicao."]
with open(saida / "condicoes.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["condicao", "imagens", "objetos", "modelo", "mAP50-95", "ic95_inf", "ic95_sup", "mAP50",
                "AP_pequenos", "variacao_pct_vs_dia_limpo"])
    w.writerows(linhas)
(saida / "condicoes.txt").write_text("\n".join(txt) + "\n", encoding="utf-8")
print("\n".join(txt))
print(f"\nSalvo em {saida.resolve()}")
