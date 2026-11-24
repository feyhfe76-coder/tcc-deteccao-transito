"""Treino completo do RT-DETR-L no dataset do TCC, com retomada automatica.

Uso (venv ativado, na pasta C:\\tcc):
    python treinar_rtdetr.py

Para interromper: Ctrl+C no terminal (uma vez) e aguarde a mensagem.
Para continuar:   rode exatamente o mesmo comando. Ele retoma da ultima epoca salva.

O que fica em C:\\tcc\\treinos\\rtdetr_l_640\\ :
    weights\\last.pt          checkpoint da ultima epoca (usado na retomada)
    weights\\best.pt          melhor modelo na validacao (o que vai para o teste)
    results.csv              metricas de cada epoca (Ultralytics)
    registro_epocas.csv      hora, memoria de GPU e nota de cada epoca
    sessoes.csv              inicio, fim e duracao de cada sessao de treino
    CONCLUIDO.txt            criado quando o treino termina

Receita (ver a explicacao na conversa):
    AdamW, lr 1e-4, weight decay 1e-4, lote 4 com acumulo de 4 passos (lote efetivo 16),
    100 epocas no maximo, parada antecipada com paciencia 20, 640 px, precisao mista.
"""
import argparse
import csv
import ctypes
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import torch
from ultralytics import RTDETR


def agora():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def anexar(csv_path, cabecalho, linha):
    novo = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if novo:
            w.writerow(cabecalho)
        w.writerow(linha)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"C:\bdd_tcc\data.yaml")
    ap.add_argument("--model", default="rtdetr-l.pt")
    ap.add_argument("--nome", default="rtdetr_l_640")
    ap.add_argument("--projeto", default="treinos")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--nbs", type=int, default=16, help="lote efetivo (acumulo de gradiente)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--lr0", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default=None, help="deixe vazio para usar a GPU")
    args = ap.parse_args()

    run_dir = Path(args.projeto).resolve() / args.nome
    pesos = run_dir / "weights"
    last, copia = pesos / "last.pt", pesos / "last_copia.pt"
    feito = run_dir / "CONCLUIDO.txt"

    if feito.exists():
        print(f"Este treino ja terminou:\n{feito.read_text(encoding='utf-8')}")
        print(f"Para treinar de novo, use outro --nome ou apague a pasta {run_dir}")
        return

    # impede o Windows de suspender por inatividade enquanto o treino roda
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)

    # ---------- carregar: retomada ou inicio ----------
    retomando, model = False, None
    for ckpt in (last, copia):
        if ckpt.exists():
            try:
                model = RTDETR(str(ckpt))
                retomando = True
                print(f"\n>>> Retomando do checkpoint {ckpt}\n")
                break
            except Exception as e:  # arquivo corrompido por interrupcao durante a gravacao
                print(f"AVISO: nao foi possivel ler {ckpt} ({str(e)[:80]}). Tentando a copia.")
    if model is None:
        model = RTDETR(args.model)
        print(f"\n>>> Iniciando treino novo em {run_dir}\n")

    # ---------- registros proprios ----------
    sessao = {"inicio": agora(), "t0": time.time(), "primeira": None, "ultima": None}
    usa_gpu = torch.cuda.is_available() and args.device != "cpu"

    def ao_iniciar_epoca(trainer):
        sessao["em_curso"] = trainer.epoch + 1

    def ao_fim_da_epoca(trainer):
        ep = sessao.get("em_curso")
        if ep is None or sessao["ultima"] == ep:
            return  # a avaliacao final chama este ponto de novo
        sessao["primeira"] = sessao["primeira"] or ep
        sessao["ultima"] = ep
        anexar(run_dir / "registro_epocas.csv",
               ["data_hora", "epoca", "segundos_desde_inicio_da_sessao", "pico_gpu_reservado_MiB",
                "pico_gpu_alocado_MiB", "nota_mAP50-95", "melhor_nota_ate_aqui"],
               [agora(), ep, round(time.time() - sessao["t0"]),
                round(torch.cuda.max_memory_reserved() / 2**20) if usa_gpu else "",
                round(torch.cuda.max_memory_allocated() / 2**20) if usa_gpu else "",
                round(float(trainer.fitness), 5) if trainer.fitness is not None else "",
                round(float(trainer.best_fitness), 5) if trainer.best_fitness is not None else ""])

    def ao_salvar(trainer):
        if trainer.last.exists():  # copia de seguranca do checkpoint recem-gravado
            tmp = trainer.last.with_name("last_copia.tmp")
            shutil.copy2(trainer.last, tmp)
            tmp.replace(trainer.last.with_name("last_copia.pt"))

    model.add_callback("on_train_epoch_start", ao_iniciar_epoca)
    model.add_callback("on_fit_epoch_end", ao_fim_da_epoca)
    model.add_callback("on_model_save", ao_salvar)

    # ---------- treinar ----------
    motivo = "erro"
    try:
        if retomando:
            model.train(resume=True)
        else:
            extra = {"device": args.device} if args.device else {}
            model.train(
                data=args.data, epochs=args.epochs, patience=args.patience,
                imgsz=args.imgsz, batch=args.batch, nbs=args.nbs,
                optimizer="AdamW", lr0=args.lr0, weight_decay=args.weight_decay,
                warmup_bias_lr=0.0, amp=True, seed=0, workers=args.workers,
                project=str(run_dir.parent), name=run_dir.name, exist_ok=True, **extra,
            )
        motivo = "concluido"
    except KeyboardInterrupt:
        motivo = "interrompido"
    finally:
        duracao = time.time() - sessao["t0"]
        if run_dir.exists():
            anexar(run_dir / "sessoes.csv",
                   ["inicio", "fim", "duracao_s", "primeira_epoca", "ultima_epoca", "motivo",
                    "pico_gpu_reservado_MiB"],
                   [sessao["inicio"], agora(), round(duracao), sessao["primeira"], sessao["ultima"],
                    motivo, round(torch.cuda.max_memory_reserved() / 2**20) if usa_gpu else ""])
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

    if motivo == "concluido":
        total = 0
        with open(run_dir / "sessoes.csv", newline="", encoding="utf-8") as f:
            total = sum(int(r["duracao_s"]) for r in csv.DictReader(f))
        feito.write_text(f"Terminado em {agora()}\nUltima epoca executada: {sessao['ultima']}\n"
                         f"Tempo total somando as sessoes: {total / 3600:.2f} h\n"
                         f"Melhor modelo: {pesos / 'best.pt'}\n", encoding="utf-8")
        print(f"\n>>> TREINO CONCLUIDO.\n{feito.read_text(encoding='utf-8')}")
    elif motivo == "interrompido":
        print(f"\n>>> Treino interrompido apos {duracao / 3600:.2f} h nesta sessao "
              f"(ultima epoca completa: {sessao['ultima']}).\n"
              ">>> Para continuar, rode o mesmo comando de novo.")


if __name__ == "__main__":
    main()
