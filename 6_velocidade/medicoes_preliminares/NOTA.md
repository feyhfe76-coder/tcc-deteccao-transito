# Medições preliminares de velocidade (NÃO usadas nas tabelas do trabalho)

Feitas com o mesmo `medir_latencia.py`. Ficam aqui porque mostram os efeitos que levaram ao protocolo final.

| Pasta | O que aconteceu | Consequência no protocolo |
|---|---|---|
| `latencia_gpu_rtx4050/` | Na 1ª repetição do 1º modelo (RT-DETR-L FP32), as primeiras ~550 imagens levaram ~85 ms, contra ~33 ms depois. A correlação dos tempos por imagem entre repetições é ~0: não depende das imagens. | Pré-aquecimento de 2 min na GPU antes da 1ª medição |
| `latencia_gpu_rtx4050_v2/` | O início ficou normal, mas no fim da sessão (~15 min finais) tudo ficou mais lento: RT-DETR-L 32 → 51 ms. A memória registrada está inflada em ~124 MiB por um erro do script (o modelo do pré-aquecimento não era liberado), corrigido depois. | Modo "Desempenho máximo" e tela sempre ligada; correção da liberação de memória |
| `latencia_cpu_i7_13620h/` | Degrau aos ~75 min: a partir daí, RT-DETR-L +28%, YOLO26m +30%, Faster +4%. Compatível com o Windows entrando em economia de energia (tela desligada). | Idem; a CPU foi medida de novo (`latencia_cpu_i7_13620h_v2`) |

As medições finais (`latencia_gpu_rtx4050_v3`, `latencia_cpu_i7_13620h_v2`, `latencia_rpi4`) não mostram esses efeitos: a variação das medianas entre repetições ficou em até 1,9 ms na GPU e até ~20 ms na CPU (< 6%).
