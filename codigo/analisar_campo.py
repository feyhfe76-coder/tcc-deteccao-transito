"""Ensaio de campo: taxa de detecção do alvo por distância, para os três modelos.

Os vídeos devem estar numa pasta e ter a distância no nome, por exemplo:
    pessoa_10m.mp4, pessoa_20m.mp4, ..., pessoa_45m.mp4

Etapas:
  1) ANOTAÇÃO (uma vez por vídeo): abre o quadro do meio do vídeo e você desenha a caixa em volta
     do alvo com o mouse; ENTER ou ESPAÇO confirma, C cancela e redesenha. As caixas ficam salvas em
     anotacoes.json (rodando de novo, elas são reaproveitadas; apague o arquivo para refazer).
  2) DETECÇÃO: roda os três modelos em quadros espaçados de cada vídeo, cada um com o seu limiar
     de confiança (ponto de operação da Seção 3.6.3).
  3) RESULTADO: o alvo conta como detectado num quadro se houver uma caixa da classe certa com
     IoU >= 0,50 em relação à caixa anotada. Taxa de detecção = quadros detectados / quadros.
     Alcance = maior distância com taxa >= 50%.

O vídeo é usado como foi gravado (sem cortar nem girar), o que preserva o tamanho real do alvo.

Uso (venv ativado, na pasta C:\\tcc, com testar_fotos.py e treinar_frcnn.py na mesma pasta):
    python analisar_campo.py --videos C:\\tcc\\campo\\videos
    python analisar_campo.py --videos C:\\tcc\\campo\\videos --passo 1     (todos os quadros; mais lento)
    python analisar_campo.py --videos C:\\tcc\\campo\\videos --ignorar-inicio 5 --recorte paisagem --imgsz 960
        (variações de formato; use variacoes_campo.py para rodar todas de uma vez)
    python analisar_campo.py --videos C:\\tcc\\campo\\videos --ignorar-inicio 5   (descarta os 5 s iniciais,
        em que a mão do ajudante cobre a câmera; as anotações já feitas continuam valendo)

Saída em C:\\tcc\\campo_resultados\\ : resumo.txt, taxa_por_distancia.csv, quadros.csv,
fig_taxa_distancia.pdf/.png e exemplo_<distancia>.jpg (alvo anotado + caixas de cada modelo).
"""
import argparse
import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import RTDETR, YOLO

from testar_fotos import desenhar, modelo_frcnn, modelo_ultralytics

COR_MPL = {"RT-DETR-L": "#2a78d6", "YOLO26m": "#eb6834", "Faster R-CNN v2": "#1baf7a"}
MARCA = {"RT-DETR-L": "o", "YOLO26m": "s", "Faster R-CNN v2": "^"}


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def ler_quadros(arq, passo):
    cap = cv2.VideoCapture(str(arq))  # o OpenCV aplica a rotação gravada pelo celular
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    quadros, i = [], 0
    while True:
        ok, q = cap.read()
        if not ok:
            break
        if i % passo == 0:
            quadros.append((i, q))
        i += 1
    cap.release()
    return quadros, fps, i


def _janela(nome, w, h):
    """Janela no tamanho da imagem, no canto superior esquerdo e na frente das outras."""
    cv2.namedWindow(nome, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(nome, int(w), int(h))
    cv2.moveWindow(nome, 10, 10)
    try:
        cv2.setWindowProperty(nome, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass


def _roi(janela, img):
    """Abre a janela até que uma caixa seja desenhada (um ENTER sem caixa não encerra o programa)."""
    while True:
        cv2.waitKey(300)  # descarta teclas que sobraram da janela anterior
        _janela(janela, img.shape[1], img.shape[0])
        x, y, bw, bh = cv2.selectROI(janela, img, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(janela)
        cv2.waitKey(1)
        if bw > 0 and bh > 0:
            return x, y, bw, bh
        print("   Nenhuma caixa desenhada nesta janela; ela vai abrir de novo. "
              "Arraste com o botão esquerdo do mouse e só depois tecle ENTER.", flush=True)


def escolher_faixa(quadro, caixas, altura_tela=850):
    """Escolha da faixa paisagem (16:9, largura total), a mesma para todos os vídeos, como uma câmera fixa.
    Mova o mouse para posicionar a faixa, clique para fixá-la; ENTER aceita (só se todos os alvos
    couberem), R solta a faixa para reposicionar."""
    H, W = quadro.shape[:2]
    hc = round(W * 9 / 16)
    s = min(1.0, altura_tela / H)
    base = cv2.resize(quadro, (round(W * s), round(H * s)))
    for x1, y1, x2, y2 in caixas:  # alvos de todos os vídeos (podem estar em alturas diferentes)
        cv2.rectangle(base, (int(x1 * s), int(y1 * s)), (int(x2 * s), int(y2 * s)), (0, 255, 0), 2)
    est = {"y0": max(0, round(min(c[1] for c in caixas) - hc / 3)), "fixo": False}

    def mouse(ev, x, y, *_):
        if ev == cv2.EVENT_MOUSEMOVE and not est["fixo"]:
            est["y0"] = int(min(max(y / s - hc / 2, 0), H - hc))
        elif ev == cv2.EVENT_LBUTTONDOWN:
            est["y0"] = int(min(max(y / s - hc / 2, 0), H - hc))
            est["fixo"] = True

    nome = "FAIXA PAISAGEM - mouse posiciona, clique fixa, ENTER aceita, R solta"
    cv2.waitKey(300)
    _janela(nome, base.shape[1], base.shape[0])
    cv2.setMouseCallback(nome, mouse)
    print("   Faixa paisagem: mova o mouse, clique para fixar e tecle ENTER (R solta a faixa).", flush=True)
    while True:
        img = base.copy()
        y0 = est["y0"]
        dentro = all(c[1] >= y0 and c[3] <= y0 + hc for c in caixas)
        escurecer = np.zeros_like(img, bool)
        escurecer[:int(y0 * s)] = True
        escurecer[int((y0 + hc) * s):] = True
        img[escurecer] = (img[escurecer] * 0.35).astype(np.uint8)
        cor = (0, 255, 255) if dentro else (0, 0, 255)
        cv2.rectangle(img, (0, int(y0 * s)), (img.shape[1] - 1, int((y0 + hc) * s)), cor, 2)
        msg = "fixada - ENTER aceita" if est["fixo"] else "clique para fixar"
        if not dentro:
            msg = "algum alvo (verde) ficou fora da faixa"
        cv2.putText(img, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)
        cv2.imshow(nome, img)
        t = cv2.waitKey(30) & 0xFF
        if t in (ord("r"), ord("R")):
            est["fixo"] = False
        elif t in (13, 10, 32) and est["fixo"] and dentro:
            break
    cv2.destroyWindow(nome)
    cv2.waitKey(1)
    return {"y0": est["y0"], "altura": hc, "largura": W, "altura_video": H}


def anotar(quadro, titulo, altura_tela=850):
    """Três etapas: (1) na imagem inteira, marque uma ÁREA folgada em volta do alvo;
    (2) a área aparece ampliada e você desenha a caixa justa (da cabeça aos pés);
    (3) conferência: a caixa aparece sobre a imagem; ENTER aceita, R refaz."""
    h, w = quadro.shape[:2]
    s = min(1.0, altura_tela / h)
    peq = cv2.resize(quadro, (round(w * s), round(h * s)))
    while True:
        print(f"   [{titulo}] 1/3: marque uma área FOLGADA em volta do alvo e tecle ENTER.", flush=True)
        x, y, bw, bh = _roi(f"1-3 AREA - {titulo}", peq)
        X1, Y1 = int(x / s), int(y / s)
        X2, Y2 = min(w, int((x + bw) / s) + 1), min(h, int((y + bh) / s) + 1)
        recorte = quadro[Y1:Y2, X1:X2]
        z = min(altura_tela / recorte.shape[0], 1500 / recorte.shape[1])
        amp = cv2.resize(recorte, (round(recorte.shape[1] * z), round(recorte.shape[0] * z)),
                         interpolation=cv2.INTER_NEAREST)
        print(f"   [{titulo}] 2/3: desenhe a caixa JUSTA no alvo (cabeça aos pés) e tecle ENTER.", flush=True)
        x, y, bw, bh = _roi(f"2-3 CAIXA - {titulo}", amp)
        caixa = [X1 + x / z, Y1 + y / z, X1 + (x + bw) / z, Y1 + (y + bh) / z]
        conf_img = amp.copy()
        cv2.rectangle(conf_img, (int(x), int(y)), (int(x + bw), int(y + bh)), (0, 255, 0), 1)
        cv2.putText(conf_img, "ENTER = aceitar   R = refazer", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2)
        nome = f"3-3 CONFERIR - {titulo}"
        cv2.waitKey(300)
        _janela(nome, conf_img.shape[1], conf_img.shape[0])
        cv2.imshow(nome, conf_img)
        while True:
            tecla = cv2.waitKey(0) & 0xFF
            if tecla in (13, 10, 32, ord("r"), ord("R")):
                break
        cv2.destroyWindow(nome)
        cv2.waitKey(1)
        if tecla in (13, 10, 32):
            return caixa
        print("   Refazendo esta anotação.", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--saida", default="campo_resultados")
    ap.add_argument("--classe", default="person")
    ap.add_argument("--passo", type=int, default=3, help="usar 1 quadro a cada N")
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--rtdetr", default=r"treinos\rtdetr_l_640\weights\best.pt")
    ap.add_argument("--yolo", default=r"treinos\yolo26m_640\weights\best.pt")
    ap.add_argument("--frcnn", default=r"treinos\frcnn_r50v2_640_passos\best.pt")
    ap.add_argument("--conf-rtdetr", type=float, default=0.5425)
    ap.add_argument("--conf-yolo", type=float, default=0.3303)
    ap.add_argument("--conf-frcnn", type=float, default=0.7918)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--conf-coleta", type=float, default=0.05,
                    help="os modelos rodam com este limiar baixo e todas as caixas da classe-alvo sao salvas; "
                         "os resultados principais usam os limiares de cada modelo")
    ap.add_argument("--reanotar", type=float, nargs="*", default=[],
                    help="distancias (m) cuja caixa deve ser desenhada de novo, ex.: --reanotar 35 45")
    ap.add_argument("--recorte", choices=["nenhum", "paisagem"], default="nenhum",
                    help="paisagem: recorta do vídeo vertical uma faixa 16:9 (largura total), na mesma posição "
                         "para todos os vídeos (escolhida com o mouse uma vez e salva em faixa_paisagem.json)")
    ap.add_argument("--refazer-faixa", action="store_true", help="escolher de novo a faixa paisagem")
    ap.add_argument("--anotacoes", default=None,
                    help="arquivo de anotações a usar (padrão: <saida>/anotacoes.json); permite reaproveitar as caixas")
    ap.add_argument("--ignorar-inicio", type=float, default=0.0,
                    help="segundos iniciais de cada vídeo descartados da análise (ex.: 5)")
    ap.add_argument("--ignorar-fim", type=float, default=0.0,
                    help="segundos finais de cada vídeo descartados da análise")
    ap.add_argument("--reanotar-tudo", action="store_true", help="desenhar de novo as caixas de todos os vídeos")
    args = ap.parse_args()

    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    saida = Path(args.saida)
    saida.mkdir(exist_ok=True)
    videos = []
    for p in Path(args.videos).iterdir():
        m = re.search(r"(\d+(?:[.,]\d+)?)\s*m", p.stem)
        if p.suffix.lower() in (".mp4", ".mov", ".3gp", ".avi") and m:
            videos.append((float(m.group(1).replace(",", ".")), p))
    videos.sort()
    if not videos:
        raise SystemExit("Nenhum vídeo com a distância no nome (ex.: pessoa_10m.mp4).")
    print(f"Dispositivo: {dev} | vídeos: {[v.name for _, v in videos]}")

    # ---- 1) anotação
    arq_anot = Path(args.anotacoes) if args.anotacoes else saida / "anotacoes.json"
    anot = json.loads(arq_anot.read_text(encoding="utf-8")) if arq_anot.exists() else {}
    cache = {}
    for dist, p in videos:
        quadros, fps, total = ler_quadros(p, args.passo)
        cache[p.name] = (quadros, fps, total)
        if p.name not in anot or dist in args.reanotar or args.reanotar_tudo:
            meio = quadros[len(quadros) // 2][1]
            anot[p.name] = {"distancia_m": dist, "caixa": anotar(meio, f"{p.name} ({dist:g} m)"),
                            "quadro_anotado": quadros[len(quadros) // 2][0]}
            arq_anot.write_text(json.dumps(anot, indent=2), encoding="utf-8")
    for dist, p in videos:  # registro visual de cada anotação (recorte ampliado)
        q = dict(cache[p.name][0]).get(anot[p.name]["quadro_anotado"], cache[p.name][0][len(cache[p.name][0]) // 2][1])
        x1, y1, x2, y2 = [int(round(v)) for v in anot[p.name]["caixa"]]
        m = max(20, (y2 - y1))
        rec = q[max(0, y1 - m):y2 + m, max(0, x1 - m):x2 + m].copy()
        cv2.rectangle(rec, (x1 - max(0, x1 - m), y1 - max(0, y1 - m)), (x2 - max(0, x1 - m), y2 - max(0, y1 - m)), (0, 255, 0), 1)
        z = 400 / rec.shape[0]
        cv2.imwrite(str(saida / f"anotacao_{dist:g}m.png"),
                    cv2.resize(rec, (round(rec.shape[1] * z), 400), interpolation=cv2.INTER_NEAREST))
    print("Anotações salvas em", arq_anot.resolve())
    faixa = None
    if args.recorte == "paisagem":
        arq_faixa = arq_anot.parent / "faixa_paisagem.json"
        if arq_faixa.exists() and not args.refazer_faixa:
            faixa = json.loads(arq_faixa.read_text(encoding="utf-8"))
        else:
            d_max, p_max = videos[-1]  # o alvo mais distante: quadro de fundo para escolher a faixa
            q = cache[p_max.name][0][len(cache[p_max.name][0]) // 2][1]
            faixa = escolher_faixa(q, [anot[p.name]["caixa"] for _, p in videos])
            arq_faixa.write_text(json.dumps(faixa, indent=2), encoding="utf-8")
        print(f"Faixa paisagem: linhas {faixa['y0']} a {faixa['y0'] + faixa['altura']} ({arq_faixa.resolve()})")

    # ---- 2) modelos
    modelos = [
        ("RT-DETR-L", *modelo_ultralytics(RTDETR, args.rtdetr, args.conf_coleta, args.imgsz, dev), args.conf_rtdetr),
        ("YOLO26m", *modelo_ultralytics(YOLO, args.yolo, args.conf_coleta, args.imgsz, dev), args.conf_yolo),
        ("Faster R-CNN v2", *modelo_frcnn(args.frcnn, args.conf_coleta, dev, args.imgsz), args.conf_coleta),
    ]

    modelos[2] = modelos[2][:3] + (args.conf_frcnn,)
    linhas_q, resumo, sens = [], [], []
    arq_det = open(saida / "deteccoes.jsonl", "w", encoding="utf-8")
    for dist, p in videos:
        quadros, fps, total = cache[p.name]
        ini, fim = args.ignorar_inicio * fps, total - args.ignorar_fim * fps
        quadros = [(n, q) for n, q in quadros if ini <= n < fim]
        if not quadros:
            raise SystemExit(f"{p.name}: nenhum quadro sobrou após --ignorar-inicio/--ignorar-fim")
        ns = [n for n, _ in quadros]  # exemplo: o quadro anotado, se ficou na análise; senão, o do meio
        q_ex = anot[p.name]["quadro_anotado"] if anot[p.name]["quadro_anotado"] in ns else ns[len(ns) // 2]
        gt = list(anot[p.name]["caixa"])
        H, W = quadros[0][1].shape[:2]
        if faixa:  # faixa 16:9 fixa, a mesma em todos os vídeos
            hc, y0 = faixa["altura"], faixa["y0"]
            if (W, H) != (faixa["largura"], faixa["altura_video"]) or gt[1] < y0 or gt[3] > y0 + hc:
                raise SystemExit(f"{p.name}: resolução diferente ou alvo fora da faixa; use --refazer-faixa")
            quadros = [(n, q[y0:y0 + hc]) for n, q in quadros]
            gt = [gt[0], gt[1] - y0, gt[2], gt[3] - y0]
            H = hc
        alt_px = gt[3] - gt[1]
        alt_rede = alt_px * args.imgsz / max(H, W)
        lin = {"distancia_m": dist, "video": p.name, "quadros_usados": len(quadros),
               "trecho_s": f"{quadros[0][0] / fps:.1f}-{quadros[-1][0] / fps:.1f}", "fps": round(fps, 1),
               "resolucao": f"{W}x{H}", "altura_alvo_px": round(alt_px, 1), "altura_alvo_na_rede_px": round(alt_rede, 1),
               # o RT-DETR (Ultralytics) estica a imagem para um quadrado imgsz x imgsz, sem manter a proporção
               "altura_alvo_na_rede_rtdetr_px": round(alt_px * args.imgsz / H, 1)}
        exemplos = []
        for nome, prever, nomes, conf in modelos:
            k = nomes.index(args.classe)
            acertos, confs = 0, []
            cont = {"limiar_iou30": 0, "conf025_iou50": 0}
            conf_vista = []  # maior confianca de uma caixa da classe-alvo sobre o alvo (IoU >= 0,30), qualquer limiar
            for qi, (n_q, q) in enumerate(quadros):
                caixas, cs, ks = prever(q)
                alvo = [(float(iou(c, gt)), float(s)) for c, s, kk in zip(caixas, cs, ks) if kk == k]
                arq_det.write(json.dumps({"distancia_m": dist, "quadro": n_q, "modelo": nome,
                                          "deteccoes_classe_alvo": [[round(v, 4), round(s, 4)] for v, s in alvo]}) + "\n")
                validas = [(v, s) for v, s in alvo if s >= conf]
                melhor_iou, melhor_conf = max(validas, default=(0.0, 0.0))
                det = melhor_iou >= args.iou
                acertos += det
                if det:
                    confs.append(melhor_conf)
                cont["limiar_iou30"] += any(v >= 0.30 for v, s in validas)
                cont["conf025_iou50"] += any(v >= args.iou and s >= 0.25 for v, s in alvo)
                sobre = [s for v, s in alvo if v >= 0.30]
                if sobre:
                    conf_vista.append(max(sobre))
                linhas_q.append([dist, p.name, n_q, nome, round(melhor_iou, 3), round(melhor_conf, 3), int(det)])
                if n_q == q_ex:
                    m = cs >= conf
                    exemplos.append(desenhar(q, caixas[m], cs[m], ks[m], nomes, f"{nome} (conf >= {conf:.2f})"))
            n = len(quadros)
            sens.append([dist, nome, round(acertos / n, 3), round(cont["limiar_iou30"] / n, 3),
                         round(cont["conf025_iou50"] / n, 3), round(len(conf_vista) / n, 3),
                         round(float(np.median(conf_vista)), 3) if conf_vista else ""])
            lin[f"taxa_{nome}"] = round(acertos / len(quadros), 3)
            lin[f"conf_media_{nome}"] = round(float(np.mean(confs)), 3) if confs else ""
        # exemplo: quadro do meio, com a caixa anotada em branco tracejado em cada painel
        paineis = []
        for img in exemplos:
            img = img.copy()
            off = img.shape[0] - H  # faixa de título acima da imagem
            x1, y1, x2, y2 = [int(v) for v in gt]
            for t in range(0, max(1, x2 - x1), 12):
                cv2.line(img, (x1 + t, y1 + off), (min(x2, x1 + t + 6), y1 + off), (255, 255, 255), 2)
                cv2.line(img, (x1 + t, y2 + off), (min(x2, x1 + t + 6), y2 + off), (255, 255, 255), 2)
            for t in range(0, max(1, y2 - y1), 12):
                cv2.line(img, (x1, y1 + off + t), (x1, min(y2, y1 + t + 6) + off), (255, 255, 255), 2)
                cv2.line(img, (x2, y1 + off + t), (x2, min(y2, y1 + t + 6) + off), (255, 255, 255), 2)
            paineis.append(cv2.resize(img, (round(img.shape[1] * 900 / img.shape[0]), 900)))
        cv2.imwrite(str(saida / f"exemplo_{dist:g}m.jpg"), np.hstack(paineis))
        resumo.append(lin)
        print(f"{dist:>5g} m | alvo {alt_px:.0f} px ({alt_rede:.0f} px na rede) | " +
              " | ".join(f"{n} {lin[f'taxa_{n}']:.0%}" for n, *_ in modelos), flush=True)

    arq_det.close()
    with open(saida / "sensibilidade.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["distancia_m", "modelo", "taxa_principal(limiar_modelo,IoU>=0,50)", "taxa(limiar_modelo,IoU>=0,30)",
                    "taxa(conf>=0,25,IoU>=0,50)", "fracao_quadros_com_caixa_sobre_alvo(conf>=0,05,IoU>=0,30)",
                    "mediana_conf_da_caixa_sobre_alvo"])
        w.writerows(sens)
    # conferencia da anotacao: altura deveria cair com 1/distancia
    ref = resumo[0]
    for r in resumo:
        esperado = ref["altura_alvo_px"] * ref["distancia_m"] / r["distancia_m"]
        r["altura_esperada_px"] = round(esperado, 1)
        r["desvio_altura_pct"] = round((r["altura_alvo_px"] / esperado - 1) * 100, 1)

    # ---- 3) saídas
    with open(saida / "quadros.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["distancia_m", "video", "quadro", "modelo", "melhor_iou", "conf", "detectado"])
        w.writerows(linhas_q)
    with open(saida / "taxa_por_distancia.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(resumo[0]))
        w.writeheader()
        w.writerows(resumo)
    txt = [f"ENSAIO DE CAMPO | classe-alvo: {args.classe} | IoU >= {args.iou:.2f} | 1 quadro a cada {args.passo} | "
           f"recorte: {args.recorte} | imgsz: {args.imgsz}",
           f"Limiares: RT-DETR-L {args.conf_rtdetr}, YOLO26m {args.conf_yolo}, Faster R-CNN v2 {args.conf_frcnn}",
           f"Descartados: {args.ignorar_inicio:g} s iniciais e {args.ignorar_fim:g} s finais de cada vídeo "
           f"(trecho analisado: " + ", ".join(f"{r['distancia_m']:g} m {r['trecho_s']} s" for r in resumo) + ")", ""]
    for r in resumo:
        txt.append(f"{r['distancia_m']:>5g} m  {r['resolucao']}  alvo {r['altura_alvo_px']:>6} px "
                   f"({r['altura_alvo_na_rede_px']:>5} px na rede; RT-DETR {r['altura_alvo_na_rede_rtdetr_px']:>5} px)  " +
                   "  ".join(f"{n}: {r[f'taxa_{n}']:.0%}" for n, *_ in modelos))
    txt += ["", "Conferência da anotação (altura do alvo deveria cair com 1/distância; desvio > ~8% sugere caixa ou distância imprecisa):"]
    txt += [f"   {r['distancia_m']:>5g} m: anotada {r['altura_alvo_px']} px, esperada {r['altura_esperada_px']} px "
            f"({r['desvio_altura_pct']:+.1f}%)" for r in resumo]
    txt += ["", "Sensibilidade (ver sensibilidade.csv): taxa principal | limiar do modelo e IoU>=0,30 | conf>=0,25 e IoU>=0,50 | "
            "quadros com alguma caixa da classe sobre o alvo (conf>=0,05) e mediana dessa confiança:"]
    txt += [f"   {d:>5g} m {m:>16}: {a:.0%} | {b:.0%} | {c:.0%} | {e:.0%} (conf mediana {g})" for d, m, a, b, c, e, g in sens]
    txt.append("")
    for n, *_ in modelos:
        ok = [r["distancia_m"] for r in resumo if r[f"taxa_{n}"] >= 0.5]
        alc = max(ok) if ok else None
        nao_mono = any(r[f"taxa_{n}"] >= 0.5 for r in resumo if alc is not None and r["distancia_m"] > alc)
        txt.append(f"Alcance (taxa >= 50%) {n}: {f'{alc:g} m' if alc is not None else 'nenhuma distância'}"
                   + (" (atenção: taxa não monotônica)" if nao_mono else ""))
    (saida / "resumo.txt").write_text("\n".join(txt) + "\n", encoding="utf-8")
    print("\n" + "\n".join(txt))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    ax.axhline(0.5, color="#52514e", lw=1, ls="--")
    for n, *_ in modelos:
        ax.plot([r["distancia_m"] for r in resumo], [r[f"taxa_{n}"] for r in resumo], marker=MARCA[n],
                color=COR_MPL[n], lw=2, ms=7, label=n)
    ax.set_xlabel("Distância entre a câmera e o alvo (m)")
    ax.set_ylabel("Taxa de detecção")
    ax.set_ylim(-0.03, 1.03)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e4e3df", lw=0.6)
    ax.legend(frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(saida / f"fig_taxa_distancia.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\nResultados em {saida.resolve()}")


if __name__ == "__main__":
    main()
