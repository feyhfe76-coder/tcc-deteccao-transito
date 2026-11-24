"""Treino completo do YOLO no dataset do TCC, com retomada automatica.

Uso (venv ativado, na pasta C:\\tcc):
    python treinar_yolo.py

Para interromper: Ctrl+C no terminal (uma vez), de preferencia logo depois que
                  a linha de uma nova epoca aparecer.
Para continuar:   rode exatamente o mesmo comando.

O que fica em C:\\tcc\\treinos\\yolo26m_640\\ :
    weights\\best.pt          melhor modelo na validacao (o que vai para o teste)
    weights\\last.pt          checkpoint da ultima epoca (usado na retomada)
    results.csv              metricas de cada epoca (Ultralytics)
    registro_epocas.csv      hora, duracao, memoria de GPU e nota de cada epoca
    sessoes.csv              sessoes encerradas normalmente (apenas informativo)
    CONCLUIDO.txt            criado quando o treino termina, com o tempo total

Receita: a que o modo automatico da Ultralytics 8.4.173 escolhe para treinos longos
(MuSGD, taxa 0,01, momentum 0,9), aqui fixada de forma explicita para nao mudar
entre retomadas. Lote 8, 640 px, precisao mista, 100 epocas no maximo e parada
antecipada com paciencia 20.

Protecao de memoria: se o uso da GPU passar de 95% da placa, o treino para com
um aviso, em vez de continuar muito lento por transbordo para a RAM.
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
from ultralytics import YOLO

LOTE_PADRAO = 8


class MemoriaNoLimite(RuntimeError):
    pass


def agora():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def anexar(csv_path, cabecalho, linha):
    novo = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if novo:
            w.writerow(cabecalho)
        w.writerow(linha)


def tempo_das_epocas(registro_csv):
    """Soma a duracao (treino + validacao) de cada epoca concluida, em todas as sessoes.
    Se uma epoca aparecer duas vezes, vale a ultima execucao."""
    dur = {}
    with open(registro_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            dur[int(r["epoca"])] = float(r["duracao_epoca_s"])
    return sum(dur.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"C:\bdd_tcc\data.yaml")
    ap.add_argument("--model", default="yolo26m.pt")
    ap.add_argument("--nome", default="yolo26m_640")
    ap.add_argument("--projeto", default="treinos")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch", type=int, default=None, help=f"padrao: {LOTE_PADRAO}")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0,
                    help="semente aleatoria; para medir a variacao entre treinos use 1, 2... com outro --nome")
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

    if os.name == "nt":  # impede o Windows de suspender por inatividade
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)

    # ---------- carregar: retomada ou inicio ----------
    retomando, model = False, None
    for ckpt in (last, copia):
        if ckpt.exists():
            try:
                model = YOLO(str(ckpt))
                retomando = True
                print(f"\n>>> Retomando do checkpoint {ckpt}\n")
                break
            except Exception as e:
                print(f"AVISO: nao foi possivel ler {ckpt} ({str(e)[:80]}). Tentando a copia.")
    if model is None:
        model = YOLO(args.model)
        print(f"\n>>> Iniciando treino novo em {run_dir}\n")

    # ---------- registros e protecao de memoria ----------
    sessao = {"inicio": agora(), "t0": time.time(), "primeira": None, "ultima": None, "lotes": 0}
    usa_gpu = torch.cuda.is_available() and args.device != "cpu"
    vram = torch.cuda.get_device_properties(0).total_memory / 2**20 if usa_gpu else 0

    def ao_iniciar_epoca(trainer):
        sessao["em_curso"] = trainer.epoch + 1
        sessao["t_epoca"] = time.time()

    def ao_fim_do_lote(trainer):
        sessao["lotes"] += 1
        if usa_gpu and sessao["lotes"] % 25 == 0:
            pico = torch.cuda.max_memory_reserved() / 2**20
            if pico > 0.95 * vram:
                raise MemoriaNoLimite(f"pico de {pico:.0f} MiB em uma placa de {vram:.0f} MiB")

    def ao_fim_da_epoca(trainer):
        ep = sessao.get("em_curso")
        if ep is None or sessao["ultima"] == ep:
            return  # a avaliacao final chama este ponto de novo
        sessao["primeira"] = sessao["primeira"] or ep
        sessao["ultima"] = ep
        anexar(run_dir / "registro_epocas.csv",
               ["data_hora", "epoca", "duracao_epoca_s", "pico_gpu_reservado_MiB",
                "pico_gpu_alocado_MiB", "nota_mAP50-95", "melhor_nota_ate_aqui"],
               [agora(), ep, round(time.time() - sessao["t_epoca"], 1),
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
    model.add_callback("on_train_batch_end", ao_fim_do_lote)
    model.add_callback("on_fit_epoch_end", ao_fim_da_epoca)
    model.add_callback("on_model_save", ao_salvar)

    # ---------- treinar ----------
    motivo, detalhe = "erro", ""
    try:
        if retomando:
            model.train(resume=True, **({"batch": args.batch} if args.batch else {}))
        else:
            extra = {"device": args.device} if args.device else {}
            model.train(
                data=args.data, epochs=args.epochs, patience=args.patience,
                imgsz=args.imgsz, batch=args.batch or LOTE_PADRAO,
                optimizer="MuSGD", lr0=0.01, momentum=0.9, warmup_bias_lr=0.0,
                amp=True, seed=args.seed, workers=args.workers,
                project=str(run_dir.parent), name=run_dir.name, exist_ok=True, **extra,
            )
        motivo = "concluido"
    except KeyboardInterrupt:
        motivo = "interrompido"
    except MemoriaNoLimite as e:
        motivo, detalhe = "memoria", str(e)
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
        total = tempo_das_epocas(run_dir / "registro_epocas.csv")
        feito.write_text(
            f"Terminado em {agora()}\nUltima epoca executada: {sessao['ultima']}\n"
            f"Tempo das epocas (treino + validacao, todas as sessoes): {total / 3600:.2f} h\n"
            f"Melhor modelo: {pesos / 'best.pt'}\n", encoding="utf-8")
        print(f"\n>>> TREINO CONCLUIDO.\n{feito.read_text(encoding='utf-8')}")
    elif motivo == "interrompido":
        print(f"\n>>> Treino interrompido (ultima epoca completa nesta sessao: {sessao['ultima']}).\n"
              ">>> Para continuar, rode o mesmo comando de novo.")
    elif motivo == "memoria":
        print(f"\n>>> TREINO PARADO: memoria da GPU no limite ({detalhe}).\n"
              ">>> Continuar assim deixaria o treino varias vezes mais lento.\n"
              ">>> Rode de novo com um lote menor, por exemplo:  python treinar_yolo.py --batch 6\n"
              ">>> Copie esta mensagem e envie antes de continuar.")


if __name__ == "__main__":
    main()
