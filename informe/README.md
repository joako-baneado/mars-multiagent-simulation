# Informe Técnico y Resultados Experimentales

Esta carpeta contiene el informe académico completo de **LA AGENCIA** (`equipoA.py`), los datos experimentales en JSON y las figuras generadas automáticamente por la suite de benchmarks.

## Contenido

- [`Informe_EquipoA.pdf`](Informe_EquipoA.pdf) — Informe técnico compilado.
- [`Informe_EquipoA.tex`](Informe_EquipoA.tex) — Código fuente LaTeX del informe (incluye diagramas TikZ de arquitectura, FSM y secuencia FIPA-ACL).
- [`resultados_experimentos.json`](resultados_experimentos.json) — Resultados completos del torneo de 10 semillas, ablaciones (FIPA-ACL y CNP) y profiling de latencia (200 y 400 unidades).
- [`rl_calibracion.json`](rl_calibracion.json) — Historial de convergencia de la calibración paralela RL-CEM.
- [`comparativa_entrenamiento.json`](comparativa_entrenamiento.json) — Comparativa en semillas de validación no vistas (`101, 202, 303, 404`).
- `figuras/` — Gráficos generados con `matplotlib`:
  - `torneo_ratio.png`: Distribución de Ratio en el torneo de 10 partidas.
  - `cobertura_mapa.png`: Cobertura acumulada del mapa `60×45` bajo niebla de guerra.
  - `comms_ablation.png`: Impacto de filtros anti-saturación FIPA-ACL.
  - `cnp_ablation.png`: Ablación de subastas Contract Net Protocol (CNP).
  - `rl_calibracion.png`: Convergencia de RL-CEM.
  - `profiling_latencia.png`: Histograma de latencia por tick a 200 y 400 unidades.

## Recompilar el informe LaTeX

```bash
pdflatex Informe_EquipoA.tex
pdflatex Informe_EquipoA.tex
```
