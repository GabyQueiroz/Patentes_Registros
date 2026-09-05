# Patentes e Registros de Software

Codigo-fonte do experimento para recomendacao de complementaridades tecnologicas
entre patentes e programas de computador de ICTs brasileiras, com base nos
microdados BADEPI 11.0 do INPI.

## Estrutura

- `recomendacao_icts/run_experimento.py`: pipeline completo de leitura,
  preparacao dos dados, embeddings, agrupamentos, pares candidatos, treinamento,
  avaliacao e geracao de saidas.
- `recomendacao_icts/requirements.txt`: dependencias Python.
- `recomendacao_icts/README.md`: instrucoes especificas de execucao.

## Observacao

Os microdados da BADEPI devem ser baixados diretamente da pagina oficial do INPI:
https://www.gov.br/inpi/pt-br/inpi-data/dados-e-series-temporais/badepi

Os arquivos brutos, resultados gerados, modelos serializados e figuras nao sao
versionados por padrao, pois podem ser grandes e devem ser recriados pelo script.

