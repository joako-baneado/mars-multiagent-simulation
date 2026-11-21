# Informe Técnico y Resultados Experimentales

Esta carpeta contiene el informe académico completo del ecosistema multiagente TB2 (**LA AGENCIA**, **RAVAGERS**, **COLMENA** y **CUERVO**), los datos experimentales en JSON y las figuras generadas por ambas suites de benchmarks (`benchmark_equipoP.py` y `benchmark_torneo.py`).

## Contenido

- [`Informe_EquipoA.pdf`](Informe_EquipoA.pdf) — Informe técnico compilado (8 páginas: arquitectura comparada de los 4 equipos, Suite 1 vs. `EquipoBase` y Suite 2 Torneo Oficial 4-Way + estrés 400 unidades).
- [`Informe_EquipoA.tex`](Informe_EquipoA.tex) — Código fuente LaTeX del informe (incluye diagramas TikZ de arquitectura, FSM y secuencia FIPA-ACL).
- [`resultados_experimentos.json`](resultados_experimentos.json) — Resultados completos de la Suite 1 (`benchmark_equipoP.py`): torneo de 10 semillas vs. `EquipoBase`, ablaciones (FIPA-ACL y CNP) y profiling de latencia (200 y 400 unidades).
- [`resultados_torneo_4way.json`](resultados_torneo_4way.json) — Resultados completos de la Suite 2 (`benchmark_torneo.py`): torneo oficial 4-Way de 8 partidas con rotación de esquinas, telemetría de protocolos y prueba de estrés concurrente a 400 unidades.
- [`rl_calibracion.json`](rl_calibracion.json) — Historial de convergencia de la calibración paralela RL-CEM.
- [`comparativa_entrenamiento.json`](comparativa_entrenamiento.json) — Comparativa en semillas de validación no vistas (`101, 202, 303, 404`).
- `figuras/` — Gráficos generados con `matplotlib`:
  - `torneo_ratio.png`: Distribución de Ratio en el torneo base de 10 partidas (LA AGENCIA vs. 3 `EquipoBase`).
  - `torneo_4way_ratio.png`: Distribución y evolución de Ratio en el Torneo Oficial 4-Way (8 partidas rotadas).
  - `torneo_4way_eficiencia.png`: Eficiencia energética (Hielo vs. Batería y mensajes FIPA-ACL) y escalabilidad 200 vs. 400 unidades.
  - `profiling_4way_400u.png`: Latencia por facción bajo estrés 4-Way con 400 unidades concurrentes frente al límite de 40 ms.
  - `cobertura_mapa.png`: Cobertura acumulada del mapa `60×45` bajo niebla de guerra.
  - `comms_ablation.png`: Impacto de los filtros anti-saturación FIPA-ACL.
  - `cnp_ablation.png`: Ablación de subastas Contract Net Protocol (CNP).
  - `rl_calibracion.png`: Convergencia de RL-CEM.
  - `profiling_latencia.png`: Histograma de latencia aislada por tick a 200 y 400 unidades.

## Recompilar el informe LaTeX

```bash
pdflatex Informe_EquipoA.tex
pdflatex Informe_EquipoA.tex
```
