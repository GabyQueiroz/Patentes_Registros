# Sistema de recomendacao de complementaridades tecnologicas entre patentes e programas de computador

Pipeline experimental baseado nos microdados BADEPI de patentes e registros de programas de computador do INPI.

## Objetivo

Identificar ICTs brasileiras, representar patentes e softwares por embeddings textuais, descobrir areas tecnologicas, construir pares patente-software e treinar modelos de recomendacao para apoiar a prospeccao de complementaridades tecnologicas.

## Execucao rapida

```powershell
$py='C:\Users\gabri\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $py .\recomendacao_icts\run_experimento.py --modo rapido
```

## Execucao mais completa

```powershell
$py='C:\Users\gabri\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $py .\recomendacao_icts\run_experimento.py --modo completo --max-patentes 120000 --max-softwares 44658
```

## Principais saidas

As saidas sao criadas em `recomendacao_icts/resultados/`:

- `relatorio_experimento.md`: resumo metodologico e principais resultados.
- `metricas_modelos.csv`: comparacao dos modelos e hiperparametros.
- `recomendacoes_topk.csv`: pares patente-software recomendados.
- `ativos_ict.csv`: ativos identificados como ICT.
- `topicos_*.csv`: agrupamentos tecnologicos.
- `fig_*.png`: figuras em formato academico.

## Observacao metodologica

A BADEPI registra ativos de propriedade intelectual. Ela nao comprova licenciamento ou transferencia efetiva. Portanto, o modelo estima complementaridades tecnologicas potenciais, nao transferencias realizadas.

