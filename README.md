# Comparação entre RT-DETR, YOLO26 e Faster R-CNN no BDD100K

Ambiente: Windows, Python 3.11, i7-13620H, RTX 4050 (6 GB).

| Pasta | Conteúdo |
| --- | --- |
| `codigo/` | códigos |
| `1_dados/` | resumo do conjunto montado |
| `2_sondas_de_hardware/` | teste de memória da placa |
| `3_treinos/` | registros dos treinos |
| `4_avaliacao/` | exatidão na validação e no teste |
| `5_analises/` | limiares, intervalos de confiança e outras condições |
| `6_velocidade/` | tempos de processamento |
| `7_ensaio_de_campo/` | ensaio com pedestre |

## Etapas

- Instalar bibliotecas
```
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

- Baixar o dataset: https://archive.org/download/bdd100k → extrair em `C:\bdd100k`

- Montar o conjunto (dia, tempo limpo, 5 classes)
```
python preparar_dataset.py C:\bdd100k C:\bdd_tcc
```
Resultado: `1_dados/resumo.txt`

- Treinar
```
python treinar_rtdetr.py
python treinar_yolo.py
python treinar_yolo.py --seed 1 --nome yolo26m_640_s1
python treinar_frcnn.py --agenda passos --batch 8
```
Resultado: `3_treinos/<modelo>/results.csv`

- Avaliar na validação
```
python avaliar_teste.py --split val --sem-latencia --bootstrap 0
```
Resultado: `4_avaliacao/avaliacao_val/resumo.txt`

- Avaliar no teste
```
python avaliar_teste.py
```
Resultado: `4_avaliacao/avaliacao_test/resumo.txt` e `coco.csv`

- Limiar de confiança de cada modelo
```
python analise_abc.py
```
Resultado: `5_analises/analise_abc/limiares.csv` e `operacao_por_classe.csv`

- Intervalos de confiança
```
python bootstrap_grupos.py
```
Resultado: `5_analises/bootstrap_grupos/diferencas.csv`

- Velocidade
```
python medir_latencia.py --preparar
python medir_latencia.py --imagens C:\bdd_tcc\images\test --device cuda --fp16 --repeticoes 3 --rotulo gpu_rtx4050_v3
python medir_latencia.py --imagens C:\bdd_tcc\images\test --device cpu --repeticoes 3 --rotulo cpu_i7_13620h_v2
python medir_latencia.py --imagens latencia_imagens_200 --pasta-modelos modelos_latencia --device cpu --rotulo rpi4 --pausa 120
```
O último comando é no Raspberry Pi 4. Resultado: `6_velocidade/latencia_*/resumo.txt` e `tempos_por_imagem.csv`

- Outras condições de clima e iluminação
```
python preparar_condicoes.py C:\bdd100k C:\bdd_condicoes
python avaliar_teste.py --data C:\bdd_condicoes\<condicao>\data.yaml --sem-latencia --saida avaliacao_<condicao>
python bootstrap_grupos.py --teste avaliacao_<condicao> --data C:\bdd_condicoes\<condicao>\data.yaml --saida bootstrap_<condicao>
python tabela_condicoes.py
```
`<condicao>`: `nublado_dia`, `chuva_dia`, `neve_dia`, `amanhecer_anoitecer`, `noite_limpo`, `noite_chuva`

Resultado: `5_analises/avaliacao_condicoes/condicoes.csv`

- Ensaio com pedestre
```
python analisar_campo.py --videos C:\tcc\campo\videos
python variacoes_campo.py --videos C:\tcc\campo\videos
```
Resultado: `7_ensaio_de_campo/nenhum_640/taxa_por_distancia.csv`

## Outros códigos

```
python treinar_frcnn.py --sonda
```
`tabela_clima_periodo.py`, `sonda_rtdetr.py`, `contagem_operacoes.py`, `testar_fotos.py`
