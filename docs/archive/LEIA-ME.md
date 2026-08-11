# Pacote PPI-only (pronto para rodar)

Copie o conteúdo desta pasta para a RAIZ do seu projeto (junto de `data/` e `ppi_v4/`):
- `ppi_only/`               pacote refatorado (testado)
- `configs/ppi_only/`       YAMLs (base + mock_*)
- `figures/`                diagramas SVG da arquitetura
- `instrucoes_experimentos.md`  contexto completo para o claude-code

Depois abra o terminal na raiz do projeto e cole o prompt curto (abaixo) no claude-code.

## Prompt curto para o claude-code
Leia o arquivo instrucoes_experimentos.md na íntegra e execute-o de forma autônoma. O pacote
ppi_only/ já está pronto e testado — NÃO reescreva a pipeline; seu trabalho é rodar. Verifique o
ambiente (§2), rode os gates de sanity e mockup (§6–§7) e pare/reporte se algum falhar; então
execute a matriz de ablações com `python -m ppi_only.run_ablations --domains bp,cc,mf`, salvando
todos os artefatos (§9) e gerando results/RUN_REPORT.md e results/ABLATIONS_summary.{csv,md}.
Reutilize metrics/losses/engine do pacote ppi_v4 (via ppi_only/compat.py). Não altere métrica,
loss, a base-de-treino do vetor, nem reintroduza os blocos removidos (equalize/dilate/enhance/
convexize/ProtT5). Corrija apenas problemas de ambiente, logando o que fizer.
