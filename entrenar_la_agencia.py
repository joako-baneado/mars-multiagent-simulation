"""
entrenar_la_agencia.py
======================
Entrenamiento offline paralelo multi-core por Aprendizaje por Refuerzo
(Risk-Aware Cross-Entropy Method) sobre palancas energéticas y pesos U(s, a)
de LA AGENCIA (`equipoA.py`), con comparativa Sin Entrenar vs. Entrenado en
semillas de prueba no vistas.

NO modifica `simulador.py` ni `equipo_base.py` (verifica SHA-256).
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import hashlib
import io
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

TB2_DIR = Path(__file__).resolve().parent
if str(TB2_DIR) not in sys.path:
    sys.path.insert(0, str(TB2_DIR))

from equipo_base import GRID_H, GRID_W, EquipoBase, clamp
from equipoA import EquipoA
from simulador import BASES_EQUIPO, TICK_DT, SimuladorMarte

PARAMS_SIN_ENTRENAR: Dict[str, float] = {
    "w1": 3.40,
    "w2": 0.42,
    "w3": 1.65,
    "w4": 2.90,
    "chain_bonus": 1.35,
    "chain_radius": 12.0,
    "min_ice_ahorro": 3.0,
    "cooldown_hielo": 180.0,
    "cnp_min_cant": 3.0,
    "expl_paso_post": 2.0,
    "expl_tick_standby": 9999.0,
    "rover_tick_expl": 240.0,
}

LIMITES_RL: Dict[str, Tuple[float, float]] = {
    "w1": (2.0, 6.0),
    "w2": (0.15, 1.0),
    "w4": (2.0, 5.5),
    "chain_bonus": (1.1, 2.2),
    "chain_radius": (8.0, 16.0),
    "cooldown_hielo": (180.0, 500.0),
    "cnp_min_cant": (3.0, 5.0),
    "expl_paso_post": (3.0, 12.0),
    "expl_tick_standby": (180.0, 600.0),
    "rover_tick_expl": (80.0, 260.0),
}

STD_INICIAL: Dict[str, float] = {
    "w1": 0.9,
    "w2": 0.22,
    "w4": 0.75,
    "chain_bonus": 0.28,
    "chain_radius": 2.2,
    "cooldown_hielo": 90.0,
    "cnp_min_cant": 0.7,
    "expl_paso_post": 2.5,
    "expl_tick_standby": 110.0,
    "rover_tick_expl": 45.0,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def correr_partida(seed: int, params: Dict[str, float], max_ticks: int = 1200) -> Dict[str, Any]:
    random.seed(seed)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sim = SimuladorMarte()

    eq_a = EquipoA(params=params)
    eq_a.inicializar("A", BASES_EQUIPO["A"], (GRID_W, GRID_H))
    sim.equipos["A"] = eq_a
    for letra in ("B", "C", "D"):
        eb = EquipoBase()
        eb.nombre = f"Equipo {letra} (Base)"
        eb.inicializar(letra, BASES_EQUIPO[letra], (GRID_W, GRID_H))
        sim.equipos[letra] = eb

    while not sim.simulacion_terminada and sim.tick_count < max_ticks:
        sim.actualizar(TICK_DT)

    ranking = sim.obtener_ranking()
    pos_a = next(i + 1 for i, f in enumerate(ranking) if f["letra"] == "A")
    fila_a = next(f for f in ranking if f["letra"] == "A")
    prof = eq_a.obtener_resumen_profiling()

    return {
        "seed": seed,
        "pos": pos_a,
        "ratio": fila_a["ratio"],
        "recursos": fila_a["recursos"],
        "bateria": fila_a["bateria"],
        "activas": fila_a["activas"],
        "mensajes": prof["comms"]["enviados"],
        "mean_ms": prof["mean_ms"],
        "max_ms": prof["max_ms"],
    }


def _worker_evaluar_candidato(args: Tuple[int, Dict[str, float], Tuple[int, ...], int]) -> Tuple[int, float, float, Dict[str, float]]:
    idx, cand, seeds_train, max_ticks_train = args
    ratios = [correr_partida(sd, cand, max_ticks=max_ticks_train)["ratio"] for sd in seeds_train]
    media_r = statistics.mean(ratios)
    std_r = statistics.pstdev(ratios) if len(ratios) > 1 else 0.0
    score_robusto = media_r - 0.35 * std_r
    return idx, score_robusto, media_r, cand


def evaluar_conjunto(
    params: Dict[str, float],
    seeds: Tuple[int, ...],
    max_ticks: int = 1400,
    workers: int = 4,
) -> Dict[str, Any]:
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        resultados = list(
            pool.map(correr_partida, seeds, [params] * len(seeds), [max_ticks] * len(seeds))
        )
    return {
        "ratio_prom": statistics.mean(r["ratio"] for r in resultados),
        "recursos_prom": statistics.mean(r["recursos"] for r in resultados),
        "bateria_prom": statistics.mean(r["bateria"] for r in resultados),
        "activas_prom": statistics.mean(r["activas"] for r in resultados),
        "mensajes_prom": statistics.mean(r["mensajes"] for r in resultados),
        "mean_ms_prom": statistics.mean(r["mean_ms"] for r in resultados),
        "max_ms_max": max(r["max_ms"] for r in resultados),
        "victorias": sum(1 for r in resultados if r["pos"] == 1),
        "total_partidas": len(resultados),
        "detalle": resultados,
    }


def entrenar_cem_paralelo(
    generaciones: int = 5,
    poblacion: int = 8,
    seeds_train: Tuple[int, ...] = (11, 22, 33, 44),
    max_ticks_train: int = 1200,
    workers: int = 8,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    claves = list(LIMITES_RL.keys())
    media: Dict[str, float] = {
        k: (
            3.5 if k == "w1"
            else 0.35 if k == "w2"
            else 2.6 if k == "w4"
            else 1.5 if k == "chain_bonus"
            else 12.0 if k == "chain_radius"
            else 320.0 if k == "cooldown_hielo"
            else 4.0 if k == "cnp_min_cant"
            else 6.0 if k == "expl_paso_post"
            else 360.0 if k == "expl_tick_standby"
            else 160.0
        )
        for k in claves
    }
    std = dict(STD_INICIAL)

    rng = random.Random(2026)
    mejor_score = -1.0
    mejor_params = dict(PARAMS_SIN_ENTRENAR)
    mejor_params.update(media)
    historial: List[Dict[str, Any]] = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for gen in range(1, generaciones + 1):
            tareas = []
            for idx in range(poblacion):
                cand = dict(PARAMS_SIN_ENTRENAR)
                if idx == 0:
                    cand.update(media)
                else:
                    for k in claves:
                        lo, hi = LIMITES_RL[k]
                        cand[k] = round(clamp(rng.gauss(media[k], std[k]), lo, hi), 4)
                tareas.append((idx, cand, seeds_train, max_ticks_train))

            evaluados = list(pool.map(_worker_evaluar_candidato, tareas))
            evaluados.sort(key=lambda x: -x[1])

            elite_n = max(3, poblacion // 3)
            elite = [e[3] for e in evaluados[:elite_n]]
            if evaluados[0][1] > mejor_score:
                mejor_score = evaluados[0][1]
                mejor_params = dict(evaluados[0][3])

            for k in claves:
                vals = [e[k] for e in elite]
                media[k] = round(0.25 * media[k] + 0.75 * statistics.mean(vals), 4)
                std[k] = max(0.05 * (LIMITES_RL[k][1] - LIMITES_RL[k][0]), 0.35 * std[k] + 0.65 * statistics.pstdev(vals))

            info = {
                "gen": gen,
                "mejor_robusto_gen": round(evaluados[0][1], 5),
                "mejor_media_gen": round(evaluados[0][2], 5),
                "media_elite": round(statistics.mean(e[1] for e in evaluados[:elite_n]), 5),
                "mejor_global": round(mejor_score, 5),
            }
            historial.append(info)
            print(
                f"Gen {gen:02d} | robusto={info['mejor_robusto_gen']:.4f} | "
                f"ratio_medio={info['mejor_media_gen']:.4f} | mejor_global={mejor_score:.4f}",
                flush=True,
            )

    return mejor_params, historial


def main() -> None:
    t_ini = time.perf_counter()
    h_sim_0 = sha256_file(TB2_DIR / "simulador.py")
    h_base_0 = sha256_file(TB2_DIR / "equipo_base.py")
    workers = min(8, os.cpu_count() or 4)

    seeds_test = (101, 202, 303, 404)
    print(f"Usando {workers} núcleos CPU en paralelo.", flush=True)
    print("=== 1. Evaluando LA AGENCIA SIN ENTRENAR (4 semillas de test) ===", flush=True)
    eval_antes = evaluar_conjunto(PARAMS_SIN_ENTRENAR, seeds_test, max_ticks=1400, workers=min(4, workers))
    print(
        f"Sin entrenar -> Ratio={eval_antes['ratio_prom']:.5f} | "
        f"Hielo={eval_antes['recursos_prom']:.1f} | Bat={eval_antes['bateria_prom']:.1f}% | "
        f"Msgs={eval_antes['mensajes_prom']:.0f} | Victorias={eval_antes['victorias']}/4",
        flush=True,
    )

    print("\n=== 2. Entrenando LA AGENCIA con RL-CEM Paralelo (5 gen x 8 cand x 4 seeds) ===", flush=True)
    mejores_params, historial = entrenar_cem_paralelo(workers=workers)

    print("\n=== 3. Evaluando LA AGENCIA ENTRENADA (mismas 4 semillas de test) ===", flush=True)
    eval_despues = evaluar_conjunto(mejores_params, seeds_test, max_ticks=1400, workers=min(4, workers))
    print(
        f"Entrenado    -> Ratio={eval_despues['ratio_prom']:.5f} | "
        f"Hielo={eval_despues['recursos_prom']:.1f} | Bat={eval_despues['bateria_prom']:.1f}% | "
        f"Msgs={eval_despues['mensajes_prom']:.0f} | Victorias={eval_despues['victorias']}/4",
        flush=True,
    )

    assert sha256_file(TB2_DIR / "simulador.py") == h_sim_0
    assert sha256_file(TB2_DIR / "equipo_base.py") == h_base_0

    out_dir = TB2_DIR / "informe"
    out_dir.mkdir(parents=True, exist_ok=True)
    res_path = out_dir / "comparativa_entrenamiento.json"
    with res_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "tiempo_total_s": round(time.perf_counter() - t_ini, 2),
                "params_sin_entrenar": PARAMS_SIN_ENTRENAR,
                "params_entrenados": mejores_params,
                "historial_cem": historial,
                "sin_entrenar": eval_antes,
                "entrenado": eval_despues,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\n[OK] Completado en {time.perf_counter() - t_ini:.1f}s -> {res_path}")
    print("Mejores parámetros:", mejores_params)


if __name__ == "__main__":
    main()
