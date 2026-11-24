# Ensaio preliminar DESCARTADO

Faster R-CNN v2 treinado com o cronograma dos modelos da Ultralytics (queda linear da taxa ao longo de até 100 épocas, parada antecipada com paciência de 20), lote 8, taxa inicial de 0,01.

- Melhor resultado na validação já na **época 4** (mAP50-95 = 0,3356); parada na época 24 (4 + 20 sem melhora).
- A taxa ainda estava em ~0,0077 quando o treino parou: o cronograma linear mal tinha começado a reduzi-la.

A troca para o roteiro de referência da torchvision (`frcnn_r50v2_640_passos`) foi decidida **apenas com base na validação**; o conjunto de teste não foi usado nessa decisão.

Comando que reproduz este ensaio (são os padrões do script, exceto o lote):
    python treinar_frcnn.py --batch 8

Observação: o `CONCLUIDO.txt` desta pasta foi sobrescrito por engano por outro texto; o `results.csv` é a evidência.
