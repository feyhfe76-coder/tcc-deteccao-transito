"""Sonda de hardware no dataset real: descobre o maior lote que cabe na GPU
e estima o tempo de treino. Padrao: RT-DETR-L em 640 px.

Uso (venv ativado):
    python sonda_rtdetr.py
    python sonda_rtdetr.py --imgsz 960 --lotes 4 2 1
    python sonda_rtdetr.py --model yolo26s.pt --lotes 32 16 8

Como mede: para cada lote (do maior para o menor) treina 2 epocas em 10% das
imagens de treino, sem validacao. O tempo usado e o da 2a epoca. Cada tentativa
roda em um processo separado. Para no primeiro lote que couber com folga.

Saida: tabela na tela e sonda_<modelo>_<imgsz>.json
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


def worker(args, batch, out):
    import torch
    from ultralytics import RTDETR, YOLO

    res = {"lote_pedido": batch}
    try:
        cls = RTDETR if "rtdetr" in args.model.lower() else YOLO
        model = cls(args.model)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        model.train(
            data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=batch,
            fraction=args.fraction, amp=not args.no_amp, val=False, plots=False,
            workers=args.workers, seed=0, project="runs_sonda",
            name=f"{Path(args.model).stem}_{args.imgsz}_b{batch}", exist_ok=True,
        )
        wall = time.perf_counter() - t0
        tr = model.trainer
        total = torch.cuda.get_device_properties(0).total_memory / 2**20
        reservado = torch.cuda.max_memory_reserved() / 2**20
        n_img = len(tr.train_loader.dataset)
        ultima, loss_nan = wall / args.epochs, False
        csv_path = Path(tr.save_dir) / "results.csv"
        if csv_path.exists():
            with open(csv_path, newline="") as f:
                rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f)]
            if rows and "time" in rows[0] and len(rows) >= 2:
                ultima = float(rows[-1]["time"]) - float(rows[-2]["time"])
            loss_nan = any("loss" in k and v.lower() in ("nan", "inf")
                           for r in rows for k, v in r.items())
        res.update(
            status="ok", lote_usado=int(tr.batch_size), imagens_usadas=n_img,
            pico_alocado_MiB=round(torch.cuda.max_memory_allocated() / 2**20),
            pico_reservado_MiB=round(reservado), vram_total_MiB=round(total),
            no_limite=bool(reservado > 0.92 * total), loss_nan=bool(loss_nan),
            s_ultima_epoca=round(ultima, 1), s_por_img=ultima / n_img,
            tempo_total_tentativa_s=round(wall, 1),
        )
    except Exception as e:
        msg = str(e)
        res.update(status="falta_memoria" if "out of memory" in msg.lower() else "erro",
                   erro=msg[:400])
    Path(out).write_text(json.dumps(res))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="rtdetr-l.pt")
    ap.add_argument("--data", default=r"C:\bdd_tcc\data.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--lotes", type=int, nargs="+", default=[8, 4, 2])
    ap.add_argument("--fraction", type=float, default=0.1, help="fracao do treino usada na sonda")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-amp", action="store_true", help="desliga a precisao mista")
    ap.add_argument("--n-train", type=int, default=10605, help="imagens do treino completo")
    ap.add_argument("--epocas-finais", type=int, default=100, help="epocas previstas no treino completo")
    ap.add_argument("--_worker", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--_out", default="_sonda_tmp.json", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._worker is not None:
        worker(args, args._worker, args._out)
        return

    resultados = []
    saida = Path(f"sonda_{Path(args.model).stem}_{args.imgsz}{'_semamp' if args.no_amp else ''}.json")
    tmp = Path(args._out)
    repassar = ["--model", args.model, "--data", args.data, "--imgsz", str(args.imgsz),
                "--fraction", str(args.fraction), "--epochs", str(args.epochs),
                "--workers", str(args.workers)] + (["--no-amp"] if args.no_amp else [])
    for b in args.lotes:
        print(f"\n===== {args.model} | {args.imgsz} px | lote {b} =====", flush=True)
        tmp.unlink(missing_ok=True)
        subprocess.run([sys.executable, __file__, *repassar, "--_worker", str(b), "--_out", str(tmp)])
        r = json.loads(tmp.read_text()) if tmp.exists() else {"lote_pedido": b, "status": "processo_caiu"}
        if r.get("status") == "ok":
            r["horas_estimadas_treino"] = round(r["s_por_img"] * args.n_train * args.epocas_finais / 3600, 1)
        resultados.append(r)
        saida.write_text(json.dumps(resultados, indent=2))
        if r.get("status") == "ok" and not r["no_limite"] and not r["loss_nan"]:
            break
    tmp.unlink(missing_ok=True)

    print(f"\nResultado ({args.model}, {args.imgsz} px, AMP {'desligado' if args.no_amp else 'ligado'}):")
    print(f"{'lote':>5}{'usado':>7}{'status':>15}{'pico MiB':>10}{'total':>7}{'limite?':>9}"
          f"{'NaN?':>6}{'s/epoca':>9}{'s/img':>8}{'horas*':>8}")
    for r in resultados:
        print(f"{r['lote_pedido']:>5}{str(r.get('lote_usado', '-')):>7}{r['status']:>15}"
              f"{str(r.get('pico_reservado_MiB', '-')):>10}{str(r.get('vram_total_MiB', '-')):>7}"
              f"{str(r.get('no_limite', '-')):>9}{str(r.get('loss_nan', '-')):>6}"
              f"{str(r.get('s_ultima_epoca', '-')):>9}"
              f"{(format(r['s_por_img'], '.3f') if 's_por_img' in r else '-'):>8}"
              f"{str(r.get('horas_estimadas_treino', '-')):>8}")
        if r["status"] != "ok":
            print("      ", r.get("erro", ""))
    print(f"\n* horas estimadas para {args.n_train} imagens x {args.epocas_finais} epocas, "
          f"sem contar a validacao de cada epoca.\nResultados salvos em {saida}")


if __name__ == "__main__":
    main()
