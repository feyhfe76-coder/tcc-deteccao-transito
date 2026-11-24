"""Medição de velocidade (latência por imagem) dos três modelos, com o MESMO código em qualquer máquina:
notebook com GPU, notebook só com CPU e Raspberry Pi.

Protocolo (igual ao de avaliar_teste.py): lote 1, imgsz 640, conf 0,25, até 300 detecções; o tempo inclui
pré-processamento + inferência + pós-processamento, com a imagem JÁ CARREGADA na memória (a leitura do
disco e a descompressão do JPEG ficam FORA do cronômetro; os arquivos ficam na memória e cada imagem é
descomprimida logo antes de ser medida, o que poupa RAM no Raspberry). Aquecimento com as 20 primeiras imagens. A exatidão não é medida aqui.

Cuidados para a medição ser justa: em cada repetição os modelos rodam em ORDEM DIFERENTE (rodízio), com
pausa entre eles; o script registra hardware, versões, temperatura, uso de swap e (no Raspberry) se houve
redução de velocidade por temperatura ou subtensão.

PASSO 1 (notebook, uma vez): separar as MESMAS 200 imagens da medição original (mesmo sorteio de
avaliar_teste.py) numa pasta para levar ao Raspberry:
    python medir_latencia.py --preparar
    -> cria latencia_imagens_200\\ (200 .jpg + lista.txt com a ordem) e modelos_latencia\\ (os 3 modelos)

PASSO 2 (medir):
  Notebook, GPU (todas as 1.764 imagens do teste, FP32 e FP16, 3 repetições):
    python medir_latencia.py --imagens C:\\bdd_tcc\\images\\test --device cuda --fp16 --repeticoes 3 --rotulo gpu_rtx4050
  Notebook, só CPU:
    python medir_latencia.py --imagens C:\\bdd_tcc\\images\\test --device cpu --repeticoes 3 --rotulo cpu_i7_13620h
  Raspberry Pi (as 200 imagens, 1 repetição):
    python3 medir_latencia.py --imagens latencia_imagens_200 --pasta-modelos modelos_latencia --device cpu --rotulo rpi4
  Só estimar o tempo total (aquecimento + 10 imagens por modelo) e sair:
    ... --estimar

Saída em latencia_<rotulo>\\ :
  resumo.txt                texto com as tabelas e os avisos
  resumo.csv                por repetição, modelo e formato: média, mediana, p95, mín., máx., FPS, memória
  tempos_por_imagem.csv     o tempo de CADA imagem (permite recalcular sobre qualquer subconjunto)
  deteccoes_amostra.json    as detecções das 10 primeiras imagens (para conferir que todas as máquinas
                            detectam a mesma coisa; não cronometrado)
  maquina.json              hardware, sistema, versões, número de threads, energia
"""
import argparse
import csv
import gc
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

MODELOS = [("RT-DETR-L", "rtdetr", "rtdetr_l.pt"), ("YOLO26m", "yolo", "yolo26m.pt"),
           ("Faster R-CNN v2", "frcnn", "frcnn_r50v2.pt")]
PADRAO = {"rtdetr": Path("treinos/rtdetr_l_640/weights/best.pt"),
          "yolo": Path("treinos/yolo26m_640/weights/best.pt"),
          "frcnn": Path("treinos/frcnn_r50v2_640_passos/best.pt")}
ARQS = {"r50": fasterrcnn_resnet50_fpn, "r50v2": fasterrcnn_resnet50_fpn_v2}


# ------------------------------------------------------------------ informações da máquina
def _cmd(args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=15).stdout.strip()
    except Exception:
        return ""


def ler_arquivo(p):
    try:
        return Path(p).read_text(errors="ignore").strip("\x00\n ")
    except Exception:
        return ""


def eh_raspberry():
    return "raspberry" in ler_arquivo("/proc/device-tree/model").lower()


USA_GPU = False  # definido em main(); a temperatura da GPU só é lida quando a medição é na GPU


def temperatura():
    """Medição na GPU: temperatura da GPU NVIDIA. Na CPU: temperatura do processador no Linux/Raspberry
    (no Windows, None: o Windows não expõe esse sensor de forma simples)."""
    if USA_GPU:
        t = _cmd(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"])
        if t:
            try:
                return float(t.splitlines()[0])
            except ValueError:
                pass
    t = ler_arquivo("/sys/class/thermal/thermal_zone0/temp")
    try:
        return round(int(t) / 1000, 1)
    except ValueError:
        return None


def throttled():
    """Raspberry: 0x0 = sem problemas. Bits 0-3 = agora; 16-19 = ocorreu desde a ligação
    (0/16 subtensão, 1/17 frequência limitada, 2/18 redução por temperatura, 3/19 limite suave de temperatura)."""
    s = _cmd(["vcgencmd", "get_throttled"])
    return s.split("=")[-1] if s else None


def swap_usada_mib():
    m = {}
    for linha in ler_arquivo("/proc/meminfo").splitlines():
        k, _, v = linha.partition(":")
        m[k] = v.strip().split()[0] if v.strip() else "0"
    if "SwapTotal" not in m:
        return None
    return round((int(m["SwapTotal"]) - int(m["SwapFree"])) / 1024, 1)


def memoria_total_gib():
    if sys.platform == "win32":
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        ms = MS()
        ms.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
        return round(ms.ullTotalPhys / 2**30, 1)
    for linha in ler_arquivo("/proc/meminfo").splitlines():
        if linha.startswith("MemTotal"):
            return round(int(linha.split()[1]) / 2**20, 1)
    return None


def na_tomada():
    if sys.platform != "win32":
        return None
    import ctypes

    class PS(ctypes.Structure):
        _fields_ = [("ACLineStatus", ctypes.c_byte), ("BatteryFlag", ctypes.c_byte),
                    ("BatteryLifePercent", ctypes.c_byte), ("SystemStatusFlag", ctypes.c_byte),
                    ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]
    ps = PS()
    ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(ps))
    return {0: False, 1: True}.get(ps.ACLineStatus)


def nucleos_fisicos():
    """Núcleos físicos (sem contar hyper-threading); usado como padrão de threads na CPU."""
    if sys.platform == "win32":
        n = _cmd(["powershell", "-NoProfile", "-Command",
                  "(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum"])
        return int(n) if n.strip().isdigit() else os.cpu_count()
    pares = set()
    fis = nucleo = None
    for linha in ler_arquivo("/proc/cpuinfo").splitlines() + [""]:
        k, _, v = linha.partition(":")
        k = k.strip()
        if k == "physical id":
            fis = v.strip()
        elif k == "core id":
            nucleo = v.strip()
        elif not linha.strip() and nucleo is not None:
            pares.add((fis, nucleo))
            fis = nucleo = None
    return len(pares) or os.cpu_count()


def nome_processador():
    if sys.platform == "win32":
        n = _cmd(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"])
        return n or platform.processor()
    for linha in ler_arquivo("/proc/cpuinfo").splitlines():
        if linha.lower().startswith("model name"):
            return linha.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def info_maquina(args, dev):
    import ultralytics
    info = {"data_hora": datetime.now().isoformat(timespec="seconds"), "rotulo": args.rotulo,
            "sistema": platform.platform(), "maquina": ler_arquivo("/proc/device-tree/model") or platform.node(),
            "processador": nome_processador(), "nucleos_logicos": os.cpu_count(),
            "ram_total_GiB": memoria_total_gib(), "python": platform.python_version(),
            "torch": torch.__version__, "torchvision": torchvision.__version__,
            "ultralytics": ultralytics.__version__, "numpy": np.__version__, "opencv": cv2.__version__,
            "dispositivo": str(dev), "nucleos_fisicos": nucleos_fisicos(), "threads_torch": args.threads,
            "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
            "na_tomada": na_tomada(), "throttled_inicio": throttled(), "temperatura_inicio": temperatura()}
    if sys.platform == "win32":
        info["plano_energia"] = _cmd(["powercfg", "/getactivescheme"])
    if dev.type == "cuda":
        info["driver_nvidia"] = _cmd(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    return info


# ------------------------------------------------------------------ modelos
def carregar(tipo, caminho, dev, imgsz):
    """Devolve (f_cronometrada, f_completa). f_cronometrada é IDÊNTICA à de avaliar_teste.py."""
    if tipo in ("rtdetr", "yolo"):
        from ultralytics import RTDETR, YOLO
        m = (RTDETR if tipo == "rtdetr" else YOLO)(str(caminho))

        def prever(im, meia):
            extra = {"half": True} if meia else {}
            return m.predict(im, imgsz=imgsz, conf=0.25, max_det=300, device=dev, verbose=False, **extra)[0]

        def f(im, meia):
            return prever(im, meia).boxes.xyxy.cpu()

        def completa(im, meia):
            b = prever(im, meia).boxes
            return b.xyxy.float().cpu().numpy(), b.conf.float().cpu().numpy(), b.cls.cpu().numpy().astype(int)
        return f, completa, m

    ck = torch.load(caminho, map_location="cpu", weights_only=False)
    # mesma arquitetura de treinar_frcnn.criar_modelo, sem baixar pesos do COCO (os pesos vêm do arquivo)
    m = ARQS[ck["arq"]](weights=None, weights_backbone=None, min_size=ck["imgsz"], max_size=ck["imgsz"],
                        box_score_thresh=0.001, box_detections_per_img=300)
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features,
                                                  len(ck["nomes"]) + 1)
    m.load_state_dict(ck["modelo"])
    m = m.to(dev).eval()
    m.roi_heads.score_thresh = 0.25
    usa_gpu = dev.type == "cuda"

    @torch.no_grad()
    def bruto(im, meia):
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        s = ck["imgsz"] / max(h, w)
        rgb = cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(rgb).permute(2, 0, 1).to(dev).float().div_(255)
        with torch.autocast(device_type=dev.type, enabled=meia and usa_gpu):
            o = m([t])[0]
        return o, s

    def f(im, meia):
        o, s = bruto(im, meia)
        return (o["boxes"].float() / s).cpu()

    def completa(im, meia):
        o, s = bruto(im, meia)
        return ((o["boxes"].float() / s).cpu().numpy(), o["scores"].float().cpu().numpy(),
                (o["labels"].cpu() - 1).numpy().astype(int))
    return f, completa, m


def abrir(dados):
    return cv2.imdecode(np.frombuffer(dados, np.uint8), cv2.IMREAD_COLOR)  # igual a cv2.imread


def medir(f, imgs, meia, dev, threads, aquec=20, limite=None):
    usa_gpu = dev.type == "cuda"
    sinc = torch.cuda.synchronize if usa_gpu else (lambda: None)
    torch.set_num_threads(threads)
    for dados in imgs[:aquec]:
        f(abrir(dados), meia)
    # A Ultralytics muda o número de threads do PyTorch na 1ª previsão (select_device: min(8, núcleos - 1)).
    # Reimpõe o valor escolhido DEPOIS do aquecimento, para os três modelos medirem nas mesmas condições.
    torch.set_num_threads(threads)
    tempos = []
    for dados in imgs[:limite] if limite else imgs:
        im = abrir(dados)  # fora do cronômetro
        sinc()
        t0 = time.perf_counter()
        f(im, meia)
        sinc()
        tempos.append((time.perf_counter() - t0) * 1000)
    return tempos


def estat(t):
    t = np.asarray(t)
    return {"n": int(len(t)), "media_ms": float(t.mean()), "mediana_ms": float(np.median(t)),
            "p95_ms": float(np.percentile(t, 95)), "min_ms": float(t.min()), "max_ms": float(t.max()),
            "fps": float(1000 / t.mean())}


# ------------------------------------------------------------------ preparo (notebook)
def preparar(args):
    import yaml
    raiz = Path(args.data).parent
    nomes_cls = list(yaml.safe_load(Path(args.data).read_text())["names"].values())
    todas = sorted((raiz / "images" / "test").glob("*.jpg"))  # mesmo sorteio de avaliar_teste.py
    escolhidas = random.Random(0).sample(todas, min(200, len(todas)))
    dst = Path("latencia_imagens_200")
    dst.mkdir(exist_ok=True)
    for p in escolhidas:
        shutil.copy2(p, dst / p.name)
    (dst / "lista.txt").write_text("\n".join(p.name for p in escolhidas) + "\n", encoding="utf-8")
    pm = Path("modelos_latencia")
    pm.mkdir(exist_ok=True)
    for _, tipo, nome in MODELOS:
        caminho = Path(getattr(args, tipo)) if getattr(args, tipo) else PADRAO[tipo]
        if not caminho.exists():
            raise SystemExit(f"Modelo não encontrado: {caminho.resolve()}")
        shutil.copy2(caminho, pm / nome)
    print(f"{len(escolhidas)} de {len(todas)} imagens do teste copiadas para {dst.resolve()} (ordem em lista.txt)")
    print(f"Modelos copiados para {pm.resolve()}: {[n for *_, n in MODELOS]}")
    print(f"Classes: {nomes_cls}")


# ------------------------------------------------------------------ principal
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preparar", action="store_true", help="separar as 200 imagens e os modelos (notebook)")
    ap.add_argument("--data", default=r"C:\bdd_tcc\data.yaml")
    ap.add_argument("--imagens", help="pasta das imagens (com lista.txt, usa essa ordem; senão, ordem alfabética)")
    ap.add_argument("--pasta-modelos", help="pasta com rtdetr_l.pt, yolo26m.pt e frcnn_r50v2.pt")
    ap.add_argument("--rtdetr")
    ap.add_argument("--yolo")
    ap.add_argument("--frcnn")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--fp16", action="store_true", help="na GPU, mede também em FP16")
    ap.add_argument("--repeticoes", type=int, default=1)
    ap.add_argument("--threads", type=int, default=None,
                    help="threads do PyTorch (padrão: número de núcleos físicos), iguais para os três modelos")
    ap.add_argument("--pausa", type=float, default=30, help="segundos de pausa entre modelos")
    ap.add_argument("--n-imagens", type=int, default=None, help="usar só as N primeiras imagens")
    ap.add_argument("--estimar", action="store_true", help="medir 10 imagens por modelo, estimar e sair")
    ap.add_argument("--preaquecimento", type=float, default=None,
                    help="segundos processando imagens SEM cronometrar antes da 1ª medição da sessão "
                         "(padrão: 120 na GPU, 0 na CPU). Evita o efeito de início de sessão da GPU.")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--rotulo", default=None, help="nome da máquina na saída (ex.: gpu_rtx4050, rpi4)")
    args = ap.parse_args()

    if args.preparar:
        preparar(args)
        return
    if not args.imagens:
        raise SystemExit("Informe --imagens (ou use --preparar no notebook).")
    global USA_GPU
    USA_GPU = args.device == "cuda"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda pedido, mas a GPU não está disponível neste PyTorch.")
    args.threads = args.threads or nucleos_fisicos()
    torch.set_num_threads(args.threads)
    dev = torch.device(args.device)
    args.rotulo = args.rotulo or (("rpi" if eh_raspberry() else platform.node()) + f"_{args.device}")

    pasta = Path(args.imagens)
    lista = pasta / "lista.txt"
    nomes = (lista.read_text(encoding="utf-8").split() if lista.exists()
             else sorted(p.name for p in pasta.glob("*.jpg")))
    if args.n_imagens:
        nomes = nomes[:args.n_imagens]
    if not nomes:
        raise SystemExit(f"Nenhuma imagem em {pasta.resolve()}")
    caminhos = {}
    for rot, tipo, arq in MODELOS:
        c = (Path(args.pasta_modelos) / arq if args.pasta_modelos else
             Path(getattr(args, tipo)) if getattr(args, tipo) else PADRAO[tipo])
        if not c.exists():
            raise SystemExit(f"Modelo não encontrado para {rot}: {c.resolve()}")
        caminhos[rot] = (tipo, c)
    print(f"Carregando {len(nomes)} imagens na memória...", flush=True)
    imgs = [(pasta / n).read_bytes() for n in nomes]  # arquivos JPEG na memória (descomprimidos na medição)
    if any(abrir(b) is None for b in imgs[:5]):
        raise SystemExit("Alguma imagem não pôde ser lida.")

    saida = Path(f"latencia_{args.rotulo}")
    saida.mkdir(exist_ok=True)
    info = info_maquina(args, dev)
    info.update({"n_imagens": len(nomes), "pasta_imagens": str(pasta.resolve()), "repeticoes": args.repeticoes,
                 "pausa_s": args.pausa, "imgsz": args.imgsz, "conf": 0.25, "aquecimento": 20})
    print(json.dumps(info, ensure_ascii=False, indent=2))
    if info["na_tomada"] is False:
        print("AVISO: o notebook está na BATERIA. Ligue na tomada para a medição valer.")

    formatos = [("FP32", False)] + ([("FP16", True)] if args.fp16 and dev.type == "cuda" else [])
    combos = [(rot, fmt, meia) for rot, *_ in MODELOS for fmt, meia in formatos]

    if args.estimar:
        print("\nEstimativa (aquecimento de 5 + 10 imagens por modelo; o aquecimento real é de 20):")
        total = 0.0
        for rot, fmt, meia in combos:
            tipo, c = caminhos[rot]
            f, c_, m = carregar(tipo, c, dev, args.imgsz)
            t = medir(f, imgs, meia, dev, args.threads, aquec=5, limite=10)
            seg = np.mean(t) / 1000 * (len(imgs) + 20) * args.repeticoes
            total += seg + args.pausa * args.repeticoes
            print(f"  {rot:>16} {fmt}: {np.median(t):8.1f} ms/imagem -> ~{seg / 60:.1f} min "
                  f"({len(imgs)} imagens x {args.repeticoes} repetição(ões))", flush=True)
            del f, c_, m
            gc.collect()
        print(f"  TOTAL estimado: ~{total / 60:.0f} min (inclui as pausas). Threads: {args.threads}. "
              f"Temperatura agora: {temperatura()} | throttled: {throttled()}")
        return

    linhas_img, resumo, amostra, avisos = [], [], {}, []
    args.preaquecimento = (120.0 if dev.type == "cuda" else 0.0) if args.preaquecimento is None else args.preaquecimento
    info["preaquecimento_s"] = args.preaquecimento
    if args.preaquecimento > 0:  # GPU "acordada" antes de medir (frequências e memória já estabilizadas)
        rot, fmt, meia = combos[0]
        print(f"\nPré-aquecimento da sessão: {args.preaquecimento:.0f} s com {rot} {fmt} (não cronometrado)...",
              flush=True)
        f, c_, m = carregar(caminhos[rot][0], caminhos[rot][1], dev, args.imgsz)
        torch.set_num_threads(args.threads)
        fim, i = time.time() + args.preaquecimento, 0
        while time.time() < fim:
            f(abrir(imgs[i % len(imgs)]), meia)
            i += 1
        print(f"   {i} imagens processadas no pré-aquecimento.", flush=True)
        del f, c_, m  # libera TODO o modelo (antes, uma referência ficava viva e inflava a memória medida)
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    for r in range(args.repeticoes):
        ordem = combos[r % len(combos):] + combos[:r % len(combos)]  # rodízio da ordem a cada repetição
        for k, (rot, fmt, meia) in enumerate(ordem):
            if r or k:
                print(f"   pausa de {args.pausa:.0f} s...", flush=True)
                time.sleep(args.pausa)
            tipo, c = caminhos[rot]
            print(f"\n[repetição {r + 1}/{args.repeticoes}] {rot} {fmt} ({len(imgs)} imagens)...", flush=True)
            if dev.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            antes = {"temp": temperatura(), "swap": swap_usada_mib(), "thr": throttled()}
            f, completa, m = carregar(tipo, c, dev, args.imgsz)
            t0 = time.time()
            tempos = medir(f, imgs, meia, dev, args.threads)
            threads_usadas = torch.get_num_threads()
            dur = time.time() - t0
            depois = {"temp": temperatura(), "swap": swap_usada_mib(), "thr": throttled()}
            e = estat(tempos)
            e.update({"repeticao": r + 1, "ordem": k + 1, "modelo": rot, "formato": fmt,
                      "pico_alocado_MiB": round(torch.cuda.max_memory_allocated() / 2**20) if dev.type == "cuda" else None,
                      "pico_reservado_MiB": round(torch.cuda.max_memory_reserved() / 2**20) if dev.type == "cuda" else None,
                      "temp_antes": antes["temp"], "temp_depois": depois["temp"],
                      "swap_antes_MiB": antes["swap"], "swap_depois_MiB": depois["swap"],
                      "throttled_depois": depois["thr"], "threads": threads_usadas, "duracao_s": round(dur, 1)})
            resumo.append(e)
            linhas_img += [{"repeticao": r + 1, "modelo": rot, "formato": fmt, "posicao": i + 1, "imagem": n,
                            "ms": round(t, 3)} for i, (n, t) in enumerate(zip(nomes, tempos))]
            if r == 0:  # detecções das 10 primeiras imagens (não cronometrado)
                amostra[f"{rot} {fmt}"] = {}
                for n, dados in zip(nomes[:10], imgs[:10]):
                    b, s, cl = completa(abrir(dados), meia)
                    amostra[f"{rot} {fmt}"][n] = [[round(float(v), 1) for v in bb] + [round(float(sc), 3), int(cc)]
                                                  for bb, sc, cc in zip(b, s, cl)]
            if (depois["swap"] or 0) > (antes["swap"] or 0) + 50:
                avisos.append(f"{rot} {fmt}, repetição {r + 1}: swap subiu de {antes['swap']} para "
                              f"{depois['swap']} MiB (tempo pode estar distorcido pela memória).")
            if depois["thr"] not in (None, "0x0"):
                avisos.append(f"{rot} {fmt}, repetição {r + 1}: throttled={depois['thr']} (houve redução de "
                              "velocidade por temperatura ou subtensão; ver docstring de throttled()).")
            print(f"   mediana {e['mediana_ms']:.1f} ms | p95 {e['p95_ms']:.1f} ms | média {e['media_ms']:.1f} ms | "
                  f"{e['fps']:.2f} FPS | temp {antes['temp']}->{depois['temp']} | {dur / 60:.1f} min", flush=True)
            del f, completa, m
            gc.collect()
            # salva a cada modelo (se a medição for interrompida, o que já foi feito não se perde)
            with open(saida / "resumo.csv", "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(resumo[0]))
                w.writeheader()
                w.writerows(resumo)
            with open(saida / "tempos_por_imagem.csv", "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(linhas_img[0]))
                w.writeheader()
                w.writerows(linhas_img)

    info["throttled_fim"] = throttled()
    info["temperatura_fim"] = temperatura()
    (saida / "maquina.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    (saida / "deteccoes_amostra.json").write_text(json.dumps(amostra), encoding="utf-8")

    txt = [f"LATÊNCIA POR IMAGEM | {args.rotulo} | {info['processador']} | GPU: {info['gpu']}",
           f"{len(nomes)} imagens, {args.repeticoes} repetição(ões), lote 1, conf 0,25, imgsz {args.imgsz}, "
           f"aquecimento 20 | torch {info['torch']} | ultralytics {info['ultralytics']} | "
           f"threads {info['threads_torch']} | tomada: {info['na_tomada']}", ""]
    txt.append(f"{'modelo':>16} {'formato':>7} {'rep':>4} {'média':>8} {'mediana':>8} {'p95':>8} {'FPS':>7} "
               f"{'aloc. MiB':>9} {'temp':>11}")
    for e in resumo:
        txt.append(f"{e['modelo']:>16} {e['formato']:>7} {e['repeticao']:>4} {e['media_ms']:>8.1f} "
                   f"{e['mediana_ms']:>8.1f} {e['p95_ms']:>8.1f} {e['fps']:>7.2f} {str(e['pico_alocado_MiB']):>9} "
                   f"{str(e['temp_antes']) + '->' + str(e['temp_depois']):>11}")
    if args.repeticoes > 1:
        txt += ["", "Variação entre repetições (mediana: menor / maior / diferença):"]
        for rot, fmt, _ in combos:
            med = [e["mediana_ms"] for e in resumo if e["modelo"] == rot and e["formato"] == fmt]
            txt.append(f"{rot:>16} {fmt:>7}: {min(med):.1f} / {max(med):.1f} / {max(med) - min(med):.1f} ms")
        txt += ["", "Todas as repetições juntas:"]
        for rot, fmt, _ in combos:
            t = [l["ms"] for l in linhas_img if l["modelo"] == rot and l["formato"] == fmt]
            e = estat(t)
            txt.append(f"{rot:>16} {fmt:>7}: média {e['media_ms']:.1f} | mediana {e['mediana_ms']:.1f} | "
                       f"p95 {e['p95_ms']:.1f} | {e['fps']:.2f} FPS")
    txt += ["", "AVISOS:"] + (avisos or ["nenhum"])
    txt.append(f"throttled no início/fim: {info['throttled_inicio']} / {info['throttled_fim']}")
    (saida / "resumo.txt").write_text("\n".join(txt) + "\n", encoding="utf-8")
    print("\n" + "\n".join(txt))
    print(f"\nArquivos em {saida.resolve()}")


if __name__ == "__main__":
    main()
