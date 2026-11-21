"""
benchmark_equipoP.py
====================
Suite de verificación funcional, torneo multi-semilla paralelo, ablaciones por
sesión, profiling de latencia (200 y 400 unidades) y generación de figuras
para el Informe Técnico de LA AGENCIA (`equipoA.py`).

IMPORTANTE: Este script importa `simulador` y `equipo_base` en modo estricto
de solo lectura y verifica sus hashes SHA-256 antes y después de ejecutarse.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import hashlib
import io
import json
import os
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

TB2_DIR = Path(__file__).resolve().parent
if str(TB2_DIR) not in sys.path:
    sys.path.insert(0, str(TB2_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from equipo_base import CAPACIDAD_ROVER, GRID_H, GRID_W, EquipoBase, clamp
from equipoA import DEFAULT_PARAMS, EquipoA
from simulador import BASES_EQUIPO, TICK_DT, SimuladorMarte, UnidadSim

INFORME_DIR = TB2_DIR / "informe"
FIGURAS_DIR = INFORME_DIR / "figuras"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verificar_integridad_archivos_base(hashes_iniciales: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    archivos = {
        "simulador.py": TB2_DIR / "simulador.py",
        "equipo_base.py": TB2_DIR / "equipo_base.py",
    }
    actuales = {nombre: sha256_file(ruta) for nombre, ruta in archivos.items()}
    if hashes_iniciales is not None:
        for nombre, h0 in hashes_iniciales.items():
            if actuales[nombre] != h0:
                raise RuntimeError(
                    f"VIOLACIÓN DE INTEGRIDAD: {nombre} fue modificado! "
                    f"Esperado={h0}, Actual={actuales[nombre]}"
                )
    return actuales


class Simulador400(SimuladorMarte):
    """
    Subclase en memoria para prueba de estrés con 400 unidades concurrentes
    (100 por equipo: 60 rovers, 30 exploradores, 10 recargadores), tal como
    describe el enunciado, sin modificar `simulador.py`.
    """

    def _generar_unidades(self) -> None:
        for letra in ("A", "B", "C", "D"):
            bx, by = BASES_EQUIPO[letra]

            def pos_spawn(radio_max: int) -> Tuple[int, int]:
                dx = random.randint(-radio_max, radio_max)
                dy = random.randint(-radio_max, radio_max)
                return (
                    int(clamp(bx + dx, 0, GRID_W - 1)),
                    int(clamp(by + dy, 0, GRID_H - 1)),
                )

            for i in range(60):
                p = pos_spawn(4)
                uid = f"{letra}_rover_{i}"
                u = UnidadSim(
                    id=uid,
                    equipo=letra,
                    tipo="rover",
                    pos=p,
                    prev_pos=p,
                    bateria=100.0,
                    carga=0,
                    capacidad=CAPACIDAD_ROVER,
                    move_interval=TICK_DT * 2.0,
                    interp_dur=TICK_DT * 2.0,
                )
                self.unidades.append(u)
                self.unidades_por_id[uid] = u

            for i in range(30):
                p = pos_spawn(3)
                uid = f"{letra}_expl_{i}"
                u = UnidadSim(
                    id=uid,
                    equipo=letra,
                    tipo="dron_explorador",
                    pos=p,
                    prev_pos=p,
                    bateria=100.0,
                    carga=0,
                    capacidad=0,
                    move_interval=TICK_DT,
                    interp_dur=TICK_DT,
                )
                self.unidades.append(u)
                self.unidades_por_id[uid] = u

            for i in range(10):
                p = pos_spawn(1)
                uid = f"{letra}_rec_{i}"
                u = UnidadSim(
                    id=uid,
                    equipo=letra,
                    tipo="dron_recarga",
                    pos=p,
                    prev_pos=p,
                    bateria=100.0,
                    carga=0,
                    capacidad=0,
                    carga_recarga=1,
                    move_interval=TICK_DT,
                    interp_dur=TICK_DT,
                )
                self.unidades.append(u)
                self.unidades_por_id[uid] = u


def ejecutar_simulacion(
    seed: int,
    letra_p: str = "A",
    params: Optional[Dict[str, float]] = None,
    unthrottled_comms: bool = False,
    disable_cnp: bool = False,
    max_ticks: int = 1600,
    usar_400: bool = False,
) -> Dict[str, Any]:
    random.seed(seed)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sim = Simulador400() if usar_400 else SimuladorMarte()

    for letra in ("A", "B", "C", "D"):
        if letra == letra_p:
            eq = EquipoA(
                params=params,
                unthrottled_comms=unthrottled_comms,
                disable_cnp=disable_cnp,
            )
            eq.nombre = "LA AGENCIA"
        else:
            eq = EquipoBase()
            eq.nombre = f"Equipo {letra} (Base)"
        eq.inicializar(letra, BASES_EQUIPO[letra], (GRID_W, GRID_H))
        sim.equipos[letra] = eq

    while not sim.simulacion_terminada and sim.tick_count < max_ticks:
        sim.actualizar(TICK_DT)

    ranking = sim.obtener_ranking()
    eq_p: EquipoA = sim.equipos[letra_p]  # type: ignore[assignment]
    pos_ranking = next(i + 1 for i, f in enumerate(ranking) if f["letra"] == letra_p)
    fila_p = next(f for f in ranking if f["letra"] == letra_p)

    if usar_400:
        activas_p = sum(1 for u in sim.unidades if u.equipo == letra_p and u.estado != "muerto")
        fila_p["activas"] = activas_p
        fila_p["muertas"] = 100 - activas_p

    return {
        "seed": seed,
        "letra_p": letra_p,
        "ticks": sim.tick_count,
        "fase_final_alcanzada": sim.fase_final,
        "hielo_inicial": sim.hielo_inicial,
        "hielo_restante": sum(sim.hielo.values()),
        "ranking": ranking,
        "posicion_p": pos_ranking,
        "fila_p": fila_p,
        "profiling": eq_p.obtener_resumen_profiling(),
        "cobertura_por_tick": dict(eq_p.cobertura_por_tick),
        "historial_consenso": list(eq_p.historial_consenso),
        "transiciones_fsm": dict(eq_p.transiciones_fsm),
        "latencias_turno_ms": list(eq_p.latencias_turno_ms.values()),
    }


def _worker_torneo(args: Tuple[int, str, Dict[str, float], int]) -> Dict[str, Any]:
    seed, lp, params_opt, max_ticks = args
    res = ejecutar_simulacion(seed=seed, letra_p=lp, params=params_opt, max_ticks=max_ticks)
    otros = [f["ratio"] for f in res["ranking"] if f["letra"] != lp]
    return {
        "seed": seed,
        "esquina": lp,
        "ticks": res["ticks"],
        "posicion_p": res["posicion_p"],
        "fila_p": res["fila_p"],
        "otros_ratios": otros,
        "ranking_completo": res["ranking"],
    }


def _worker_experimento(args: Tuple[str, Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    nombre, kwargs = args
    return nombre, ejecutar_simulacion(**kwargs)


# -----------------------------------------------------------------------------
# Pruebas funcionales y de integridad (--quick)
# -----------------------------------------------------------------------------
def correr_pruebas_funcionales() -> None:
    print("[1/2] Verificando herencia e integridad arquitectónica...")
    assert issubclass(EquipoA, EquipoBase), "EquipoA debe heredar de EquipoBase"

    eq = EquipoA()
    eq.inicializar("A", BASES_EQUIPO["A"], (GRID_W, GRID_H))
    assert eq.nombre == "LA AGENCIA"
    assert len(eq.hielo_conocido) == 0
    assert len(eq.restos_conocidos) == 0
    assert len(eq.solicitudes_recarga) == 0

    print("[2/2] Ejecutando partida rápida (seed=7, max_ticks=900)...")
    res = ejecutar_simulacion(seed=7, letra_p="A", max_ticks=900)
    prof = res["profiling"]
    assert prof["max_ms"] < 40.0, f"Latencia excedió 40ms: {prof['max_ms']:.2f} ms"
    assert prof["comms"]["enviados"] > 0, "Debe enviar mensajes FIPA-ACL"
    assert prof["consensos"] > 0, "Debe ejecutar consenso deliberativo AutoGen"
    assert prof["transiciones_fsm"] >= 3, "Debe registrar transiciones FSM"

    print(
        f"    -> OK! Ratio LA AGENCIA={res['fila_p']['ratio']:.4f} "
        f"(Hielo={res['fila_p']['recursos']}, Bat={res['fila_p']['bateria']:.1f}%, "
        f"Pos={res['posicion_p']}/4, max_ms={prof['max_ms']:.2f} ms)"
    )


# -----------------------------------------------------------------------------
# Gráficas de alta calidad para el informe
# -----------------------------------------------------------------------------
def generar_figuras(datos: Dict[str, Any]) -> None:
    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

    # 1. Calibración RL-CEM (desde comparativa_entrenamiento.json)
    comp = datos.get("comparativa_rl")
    if comp and comp.get("historial_cem"):
        hist = comp["historial_cem"]
        gens = [h["gen"] for h in hist]
        fig, ax = plt.subplots(figsize=(6.8, 3.6))
        ax.plot(gens, [h["mejor_global"] for h in hist], marker="o", lw=2.2, label="Mejor Robusto Global (J)")
        ax.plot(gens, [h["mejor_media_gen"] for h in hist], marker="s", lw=1.8, linestyle="--", label="Ratio Medio Mejor Cand.")
        ax.plot(gens, [h["media_elite"] for h in hist], marker="^", lw=1.6, linestyle=":", label="Media Élite Robusta")
        ax.set_title("Calibración Offline por Policy Search RL-CEM (LA AGENCIA)")
        ax.set_xlabel("Generación")
        ax.set_ylabel("Puntuación / Ratio")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURAS_DIR / "rl_calibracion.png", dpi=180)
        plt.close(fig)

    # 2. Cobertura del mapa vs ticks (Sesión 1)
    cov = datos["experimento_referencia"]["cobertura_por_tick"]
    if cov:
        ticks = sorted(int(k) for k in cov.keys())
        pcts = [(cov[str(t)] if str(t) in cov else cov[t]) / (GRID_W * GRID_H) * 100.0 for t in ticks]
        fig, ax = plt.subplots(figsize=(6.8, 3.6))
        ax.plot(ticks, pcts, color="#1f77b4", lw=2.2)
        ax.fill_between(ticks, pcts, alpha=0.15, color="#1f77b4")
        ax.set_title("Cobertura Acumulada del Mapa 60x45 bajo Niebla de Guerra (Sesión 1)")
        ax.set_xlabel("Tick de Simulación")
        ax.set_ylabel("Celdas Exploradas (%)")
        ax.set_ylim(0, 105)
        fig.tight_layout()
        fig.savefig(FIGURAS_DIR / "cobertura_mapa.png", dpi=180)
        plt.close(fig)

    # 3. Ablación Comunicaciones FIPA-ACL (Sesión 3)
    c_filt = datos["ablacion_comms"]["con_filtros"]
    c_unth = datos["ablacion_comms"]["sin_filtros"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.5))
    etiquetas = ["FIPA-ACL\nCon Filtros", "FIPA-ACL\nSin Filtros"]
    ax1.bar(etiquetas, [c_filt["mensajes_enviados"], c_unth["mensajes_enviados"]], color=["#2ca02c", "#d62728"])
    ax1.set_title("Mensajes Enviados")
    ax1.set_ylabel("Cantidad de mensajes")
    ax2.bar(etiquetas, [c_filt["ratio"], c_unth["ratio"]], color=["#2ca02c", "#d62728"])
    ax2.set_title("Impacto en Ratio Final")
    ax2.set_ylabel("Ratio")
    fig.suptitle("Sesión 3: Control Presupuestario y Filtros Anti-Saturación (-0.5%/msg)")
    fig.tight_layout()
    fig.savefig(FIGURAS_DIR / "comms_ablation.png", dpi=180)
    plt.close(fig)

    # 4. Ablación Contract Net Protocol (Sesión 4)
    cnp_on = datos["ablacion_cnp"]["con_cnp"]
    cnp_off = datos["ablacion_cnp"]["sin_cnp"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.5))
    et_cnp = ["Con CNP\n(Subastas)", "Sin CNP\n(Directo)"]
    ax1.bar(et_cnp, [cnp_on["recursos"], cnp_off["recursos"]], color=["#1f77b4", "#ff7f0e"])
    ax1.set_title("Hielo Depositado (u.)")
    ax2.bar(et_cnp, [cnp_on["ratio"], cnp_off["ratio"]], color=["#1f77b4", "#ff7f0e"])
    ax2.set_title("Ratio Competitivo")
    fig.suptitle("Sesión 4: Efecto de Subastas CNP en Eficiencia de Extracción")
    fig.tight_layout()
    fig.savefig(FIGURAS_DIR / "cnp_ablation.png", dpi=180)
    plt.close(fig)

    # 5. Torneo 10 semillas (Ratio por equipo)
    torneo = datos["torneo"]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    equipos_labels = ["LA AGENCIA", "Base 1", "Base 2", "Base 3"]
    ratios_por_rol = [
        [t["fila_p"]["ratio"] for t in torneo],
        [t["otros_ratios"][0] for t in torneo],
        [t["otros_ratios"][1] for t in torneo],
        [t["otros_ratios"][2] for t in torneo],
    ]
    bp = ax.boxplot(ratios_por_rol, tick_labels=equipos_labels, patch_artist=True)
    colores = ["#ff7819", "#5c7cfa", "#fcc419", "#e64980"]
    for patch, col in zip(bp["boxes"], colores):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)
    ax.set_title("Torneo Multi-Semilla (10 Partidas): Distribución de Ratio")
    ax.set_ylabel("Ratio = Recursos en Base / Batería Total Usada")
    fig.tight_layout()
    fig.savefig(FIGURAS_DIR / "torneo_ratio.png", dpi=180)
    plt.close(fig)

    # 6. Profiling de latencia 200 vs 400 unidades (Sesión 6)
    lat200 = datos["profiling_200"]["latencias_ms"]
    lat400 = datos["profiling_400"]["latencias_ms"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.hist(lat200, bins=25, alpha=0.65, label="200 unidades (50/eq)", color="#1f77b4")
    ax.hist(lat400, bins=25, alpha=0.65, label="400 unidades (100/eq)", color="#ff7f0e")
    ax.axvline(40.0, color="#d62728", linestyle="--", lw=2.0, label="Límite exigido (40 ms/turno)")
    ax.set_title("Sesión 6: Histograma de Latencia de Decisión por Turno")
    ax.set_xlabel("Tiempo total por turno de LA AGENCIA (ms)")
    ax.set_ylabel("Frecuencia (ticks)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURAS_DIR / "profiling_latencia.png", dpi=180)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Benchmark completo paralelo (--full)
# -----------------------------------------------------------------------------
def correr_benchmark_completo() -> None:
    INFORME_DIR.mkdir(parents=True, exist_ok=True)
    params_opt = dict(DEFAULT_PARAMS)
    workers = min(8, os.cpu_count() or 4)

    comp_path = INFORME_DIR / "comparativa_entrenamiento.json"
    comp_rl = None
    if comp_path.exists():
        with comp_path.open("r", encoding="utf-8") as f:
            comp_rl = json.load(f)

    print(f"\n[A] Torneo Multi-Semilla paralelo ({workers} workers, 10 partidas, rotando A/B/C/D)...")
    letras = ("A", "B", "C", "D")
    tareas_torneo = [(100 + i * 17, letras[i % 4], params_opt, 1500) for i in range(10)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        torneo_res = list(pool.map(_worker_torneo, tareas_torneo))

    for i, t in enumerate(torneo_res, start=1):
        print(
            f"  Partida {i:02d} (seed={t['seed']}, esq={t['esquina']}) -> Pos={t['posicion_p']}/4 | "
            f"Ratio={t['fila_p']['ratio']:.4f} | Hielo={t['fila_p']['recursos']} | "
            f"Bat={t['fila_p']['bateria']:.1f}% | MaxBase={max(t['otros_ratios']):.4f}"
        )

    print("\n[B] Ejecutando ablaciones (Sesión 3 y 4), referencia (200u) y estrés (400u) en paralelo...")
    tareas_exp = [
        ("ref", {"seed": 100, "letra_p": "A", "params": params_opt, "max_ticks": 1600}),
        ("sin_filtros", {"seed": 100, "letra_p": "A", "params": params_opt, "unthrottled_comms": True, "max_ticks": 1600}),
        ("sin_cnp", {"seed": 100, "letra_p": "A", "params": params_opt, "disable_cnp": True, "max_ticks": 1600}),
        ("stress_400", {"seed": 100, "letra_p": "A", "params": params_opt, "max_ticks": 1200, "usar_400": True}),
    ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        exp_map = dict(pool.map(_worker_experimento, tareas_exp))

    ref = exp_map["ref"]
    res_sin_filtros = exp_map["sin_filtros"]
    res_sin_cnp = exp_map["sin_cnp"]
    res_400 = exp_map["stress_400"]

    print(
        f"    -> Profiling 200u: mean={ref['profiling']['mean_ms']:.3f} ms, "
        f"p95={ref['profiling']['p95_ms']:.3f} ms, p99={ref['profiling']['p99_ms']:.3f} ms, "
        f"max={ref['profiling']['max_ms']:.3f} ms"
    )
    print(
        f"    -> Profiling 400u: mean={res_400['profiling']['mean_ms']:.3f} ms, "
        f"p95={res_400['profiling']['p95_ms']:.3f} ms, p99={res_400['profiling']['p99_ms']:.3f} ms, "
        f"max={res_400['profiling']['max_ms']:.3f} ms"
    )

    salida = {
        "parametros_usados": params_opt,
        "comparativa_rl": comp_rl,
        "torneo": torneo_res,
        "experimento_referencia": {
            "fila_p": ref["fila_p"],
            "profiling": ref["profiling"],
            "cobertura_por_tick": ref["cobertura_por_tick"],
            "historial_consenso": ref["historial_consenso"],
            "transiciones_fsm": ref["transiciones_fsm"],
        },
        "ablacion_comms": {
            "con_filtros": {
                "mensajes_enviados": ref["profiling"]["comms"]["enviados"],
                "bloqueados": ref["profiling"]["comms"]["bloqueados_filtro"],
                "costo_bateria_comms": ref["profiling"]["comms"]["enviados"] * 1.0,
                "ratio": ref["fila_p"]["ratio"],
                "bateria_total": ref["fila_p"]["bateria"],
            },
            "sin_filtros": {
                "mensajes_enviados": res_sin_filtros["profiling"]["comms"]["enviados"],
                "bloqueados": res_sin_filtros["profiling"]["comms"]["bloqueados_filtro"],
                "costo_bateria_comms": res_sin_filtros["profiling"]["comms"]["enviados"] * 1.0,
                "ratio": res_sin_filtros["fila_p"]["ratio"],
                "bateria_total": res_sin_filtros["fila_p"]["bateria"],
            },
        },
        "ablacion_cnp": {
            "con_cnp": {
                "recursos": ref["fila_p"]["recursos"],
                "bateria": ref["fila_p"]["bateria"],
                "ratio": ref["fila_p"]["ratio"],
                "cnp_stats": ref["profiling"]["cnp"],
            },
            "sin_cnp": {
                "recursos": res_sin_cnp["fila_p"]["recursos"],
                "bateria": res_sin_cnp["fila_p"]["bateria"],
                "ratio": res_sin_cnp["fila_p"]["ratio"],
                "cnp_stats": res_sin_cnp["profiling"]["cnp"],
            },
        },
        "profiling_200": {
            **ref["profiling"],
            "latencias_ms": ref["latencias_turno_ms"],
        },
        "profiling_400": {
            **res_400["profiling"],
            "latencias_ms": res_400["latencias_turno_ms"],
            "fila_p": res_400["fila_p"],
        },
    }

    json_path = INFORME_DIR / "resultados_experimentos.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Resultados guardados en {json_path}")

    generar_figuras(salida)
    print(f"[OK] Figuras generadas en {FIGURAS_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark y profiling para LA AGENCIA (equipoA.py)")
    parser.add_argument("--quick", action="store_true", help="Solo pruebas funcionales y de integridad")
    parser.add_argument("--full", action="store_true", help="Ejecutar suite completa + figuras")
    args = parser.parse_args()

    hashes_0 = verificar_integridad_archivos_base()
    print(f"SHA-256 inicial simulador.py:   {hashes_0['simulador.py']}")
    print(f"SHA-256 inicial equipo_base.py: {hashes_0['equipo_base.py']}")

    try:
        if args.quick or not args.full:
            correr_pruebas_funcionales()
        if args.full:
            correr_benchmark_completo()
    finally:
        verificar_integridad_archivos_base(hashes_0)
        print("Verificación final SHA-256: simulador.py y equipo_base.py 100% INTACTOS.")


if __name__ == "__main__":
    main()
