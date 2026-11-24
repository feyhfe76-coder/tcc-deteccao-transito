"""Cruza clima (weather) x periodo do dia (timeofday) nas imagens anotadas do BDD100K.

Uso (venv ativado):
    python tabela_clima_periodo.py C:\\bdd100k

Le as anotacoes de treino e de validacao oficiais (um JSON por imagem ou um
JSON unico por conjunto) e grava tabela_clima_periodo.txt e .csv na pasta atual.
Nao altera nada no dataset.
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

if len(sys.argv) < 2:
    sys.exit("Informe a pasta do dataset, ex.: python tabela_clima_periodo.py C:\\bdd100k")
raiz = Path(sys.argv[1])


def atributos(grupo):
    """Devolve a lista de (weather, timeofday) das imagens anotadas de um conjunto."""
    pasta = [p for p in raiz.rglob(grupo) if p.is_dir() and "labels" in p.parts]
    for p in pasta:
        arqs = sorted(p.glob("*.json"))
        if arqs:
            out = []
            for a in arqs:
                at = json.loads(a.read_text(encoding="utf-8")).get("attributes", {})
                out.append((at.get("weather"), at.get("timeofday")))
            return out, f"{len(arqs)} arquivos em {p}"
    unico = list(raiz.rglob(f"*labels_images_{grupo}.json"))
    if unico:
        dados = json.loads(unico[0].read_text(encoding="utf-8"))
        return ([(d.get("attributes", {}).get("weather"), d.get("attributes", {}).get("timeofday"))
                 for d in dados], f"arquivo unico {unico[0]}")
    sys.exit(f"ERRO: nao encontrei anotacoes de '{grupo}' dentro de {raiz}")


def tabela(pares, titulo):
    c = Counter(pares)
    climas = sorted({w for w, _ in c}, key=lambda w: -sum(n for (a, _), n in c.items() if a == w))
    horas = sorted({t for _, t in c}, key=lambda t: -sum(n for (_, b), n in c.items() if b == t))
    linhas = [[titulo] + [str(h) for h in horas] + ["Total"]]
    for w in climas:
        v = [c[(w, h)] for h in horas]
        linhas.append([str(w)] + v + [sum(v)])
    tot = [sum(c[(w, h)] for w in climas) for h in horas]
    linhas.append(["Total"] + tot + [sum(tot)])
    return linhas, c


saida_txt, saida_csv, soma = [], [], Counter()
for grupo, nome in (("train", "Treino oficial"), ("val", "Validacao oficial")):
    pares, origem = atributos(grupo)
    linhas, c = tabela(pares, nome)
    soma += c
    saida_txt.append(f"{nome}: {len(pares)} imagens anotadas ({origem})")
    saida_txt.append(f"  clear + daytime: {c[('clear', 'daytime')]}")
    saida_csv += linhas + [[]]
    larg = [max(len(str(l[i])) for l in linhas) for i in range(len(linhas[0]))]
    saida_txt += ["  " + "  ".join(str(x).rjust(larg[i]) if i else str(x).ljust(larg[i])
                                    for i, x in enumerate(l)) for l in linhas] + [""]

linhas, _ = tabela(list(soma.elements()), "Treino + validacao")
saida_csv += linhas
larg = [max(len(str(l[i])) for l in linhas) for i in range(len(linhas[0]))]
saida_txt.append(f"Treino + validacao: {sum(soma.values())} imagens anotadas")
saida_txt.append(f"  clear + daytime: {soma[('clear', 'daytime')]}   (esperado: 14241)")
saida_txt += ["  " + "  ".join(str(x).rjust(larg[i]) if i else str(x).ljust(larg[i])
                                for i, x in enumerate(l)) for l in linhas]

Path("tabela_clima_periodo.txt").write_text("\n".join(saida_txt) + "\n", encoding="utf-8")
with open("tabela_clima_periodo.csv", "w", newline="", encoding="utf-8-sig") as f:
    csv.writer(f, delimiter=";").writerows(saida_csv)
print("\n".join(saida_txt))
print("\nGravado: tabela_clima_periodo.txt e tabela_clima_periodo.csv")
