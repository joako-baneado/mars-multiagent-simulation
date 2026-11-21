"""
benchmark_torneo.py
===================
Suite de verificación funcional, integridad arquitectónica, torneo 4-way
con rotación de esquinas (LA AGENCIA vs RAVAGERS vs COLMENA vs CUERVO) y
prueba de estrés a 400 unidades (< 40 ms/turno).

NO modifica `simulador.py` ni `equipo_base.py` (verifica SHA-256 al inicio y al final).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import hashlib
import io
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

from equipo_base import CAPACIDAD_ROVER, GRID_H, GRID_W, EquipoBase, Percepcion, UnidadVista, clamp
from equipoA import EquipoA
from equipoB import EquipoB
from equipoC import EquipoC
from equipoD import EquipoD
from simulador import BASES_EQUIPO, TICK_DT, SimuladorMarte, UnidadSim, cargar_equipos_dinamicos


class _DummyAPI:
    """API mínima para pruebas unitarias aisladas sin red ni estado global."""

    def enviar_mensaje(self, dest_id: str, asunto: str, datos: Dict[str, Any]) -> bool:
        return True

    def enviar_a_recargador_cercano(self, asunto: str, datos: Dict[str, Any]) -> Optional[str]:
        return None

    def enviar_a_rover_ocioso_cercano(
        self, asunto: str, datos: Dict[str, Any], pos_ref: Optional[Tuple[int, int]] = None
    ) -> Optional[str]:
        return None

    def enviar_a_aliados_cercanos(
        self, radio: int, asunto: str, datos: Dict[str, Any], max_dest: int = 2
    ) -> int:
        return 0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verificar_integridad_archivos_base(
    hashes_iniciales: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
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
    """Subclase en memoria para prueba de estrés con 400 unidades concurrentes (100/equipo)."""

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


def _instanciar_equipos_rotados(rotacion: int) -> Dict[str, EquipoBase]:
    clases = [EquipoA, EquipoB, EquipoC, EquipoD]
    letras = ("A", "B", "C", "D")
    asignacion: Dict[str, EquipoBase] = {}
    for idx, letra in enumerate(letras):
        cls = clases[(idx + rotacion) % 4]
        inst = cls()
        inst.inicializar(letra, BASES_EQUIPO[letra], (GRID_W, GRID_H))
        asignacion[letra] = inst
    return asignacion


def ejecutar_partida_4way(
    seed: int,
    rotacion: int = 0,
    max_ticks: int = 1600,
    usar_400: bool = False,
) -> Dict[str, Any]:
    random.seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        sim = Simulador400() if usar_400 else SimuladorMarte()

    equipos = _instanciar_equipos_rotados(rotacion)
    for letra, eq in equipos.items():
        sim.equipos[letra] = eq

    while not sim.simulacion_terminada and sim.tick_count < max_ticks:
        sim.actualizar(TICK_DT)

    ranking = sim.obtener_ranking()
    por_nombre: Dict[str, Dict[str, Any]] = {}
    for pos_idx, fila in enumerate(ranking, start=1):
        letra = fila["letra"]
        eq = sim.equipos[letra]
        activas = sum(1 for u in sim.unidades if u.equipo == letra and u.estado != "muerto")
        total_u = 100 if usar_400 else 50
        prof = eq.obtener_resumen_profiling() if hasattr(eq, "obtener_resumen_profiling") else {}
        por_nombre[eq.nombre] = {
            "letra": letra,
            "posicion": pos_idx,
            "ratio": fila["ratio"],
            "recursos": fila["recursos"],
            "bateria": fila["bateria"],
            "activas": activas,
            "muertas": total_u - activas,
            "profiling": prof,
            "compartidos_vacios": (
                len(eq.hielo_conocido) == 0
                and len(eq.restos_conocidos) == 0
                and len(eq.solicitudes_recarga) == 0
            ),
        }

    return {
        "seed": seed,
        "rotacion": rotacion,
        "ticks": sim.tick_count,
        "fase_final": sim.fase_final,
        "hielo_restante": sum(sim.hielo.values()),
        "ranking": ranking,
        "por_nombre": por_nombre,
    }


def _worker_torneo(args: Tuple[int, int, int]) -> Dict[str, Any]:
    seed, rotacion, max_ticks = args
    return ejecutar_partida_4way(seed=seed, rotacion=rotacion, max_ticks=max_ticks, usar_400=False)


# -----------------------------------------------------------------------------
# Pruebas rápidas (--quick)
# -----------------------------------------------------------------------------
def _probar_hook_equipo_d() -> None:
    eq_d = EquipoD()
    eq_d.inicializar("D", BASES_EQUIPO["D"], (GRID_W, GRID_H))
    api = _DummyAPI()

    # Caso 1: batería sana (80%), carga=2 y 2 rivales adyacentes -> debe cancelar y alejarse
    u_alta = UnidadVista(
        id="D_rover_3",
        equipo="D",
        tipo="rover",
        pos=(30, 22),
        bateria=80.0,
        carga=2,
        capacidad=5,
        estado="consumiendo",
        en_tormenta=False,
    )
    perc_cong = Percepcion(
        tick=50,
        base_pos=BASES_EQUIPO["D"],
        grid_size=(GRID_W, GRID_H),
        fase_final=False,
        tiempo_final_restante=30.0,
        base_aliada_cerca=False,
        hielo_visible=[],
        restos_visible=[{"id": "r1", "pos": (30, 22), "equipo": "A", "tipo": "rover", "combustible": 20}],
        aliados_visible=[],
        rivales_visible=[
            {"id": "A_rover_1", "pos": (31, 22), "equipo": "A", "tipo": "rover", "bateria": 90.0, "estado": "activo"},
            {"id": "B_rover_2", "pos": (30, 23), "equipo": "B", "tipo": "rover", "bateria": 85.0, "estado": "activo"},
        ],
        tormentas_visible=[],
        mensajes=[],
    )
    acc1 = eq_d.decidir_unidad(u_alta, perc_cong, api)
    assert acc1[0] == "mover" and (acc1[1], acc1[2]) != (0, 0), f"Hook D debía evadir congestión, retornó {acc1}"

    # Caso 2: batería baja (35%) -> debe mantener consumo con ("esperar",)
    u_baja = UnidadVista(
        id="D_rover_4",
        equipo="D",
        tipo="rover",
        pos=(30, 22),
        bateria=35.0,
        carga=2,
        capacidad=5,
        estado="consumiendo",
        en_tormenta=False,
    )
    acc2 = eq_d.decidir_unidad(u_baja, perc_cong, api)
    assert acc2 == ("esperar",), f"Hook D con batería baja debía mantener consumo, retornó {acc2}"

    # Caso 3: veta rica ganable cercana -> debe interrumpir y moverse hacia la veta
    u_veta = UnidadVista(
        id="D_rover_5",
        equipo="D",
        tipo="rover",
        pos=(30, 22),
        bateria=80.0,
        carga=1,
        capacidad=5,
        estado="consumiendo",
        en_tormenta=False,
    )
    perc_veta = Percepcion(
        tick=52,
        base_pos=BASES_EQUIPO["D"],
        grid_size=(GRID_W, GRID_H),
        fase_final=False,
        tiempo_final_restante=30.0,
        base_aliada_cerca=False,
        hielo_visible=[(31, 22, 4)],
        restos_visible=[{"id": "r2", "pos": (30, 22), "equipo": "B", "tipo": "rover", "combustible": 10}],
        aliados_visible=[],
        rivales_visible=[
            {"id": "A_rover_1", "pos": (34, 22), "equipo": "A", "tipo": "rover", "bateria": 90.0, "estado": "activo"},
        ],
        tormentas_visible=[],
        mensajes=[],
    )
    acc3 = eq_d.decidir_unidad(u_veta, perc_veta, api)
    assert acc3[0] == "mover" and (acc3[1], acc3[2]) != (0, 0), f"Hook D debía pivotar a veta rica, retornó {acc3}"


def correr_pruebas_rapidas() -> None:
    print("[1/4] Verificando herencia, nombres y carga dinámica...")
    assert issubclass(EquipoC, EquipoBase)
    assert issubclass(EquipoD, EquipoBase)
    dinamicos = cargar_equipos_dinamicos()
    assert dinamicos["C"].nombre == "COLMENA", f"Nombre C inesperado: {dinamicos['C'].nombre}"
    assert dinamicos["D"].nombre == "CUERVO", f"Nombre D inesperado: {dinamicos['D'].nombre}"

    print("[2/4] Verificando hook exclusivo de EquipoD en consumiendo...")
    _probar_hook_equipo_d()

    print("[3/4] Ejecutando partida 4-way (seed=100, max_ticks=1400)...")
    res = ejecutar_partida_4way(seed=100, rotacion=0, max_ticks=1400, usar_400=False)

    print("[4/4] Validando Regla de Integridad, las 6 sesiones y latencia < 40 ms...")
    for nombre in ("COLMENA", "CUERVO"):
        info = res["por_nombre"][nombre]
        prof = info["profiling"]
        assert info["compartidos_vacios"], f"{nombre} violó aislamiento usando diccionarios de EquipoBase"
        assert prof["transiciones_fsm"] >= 3, f"{nombre} sin transiciones FSM suficientes"
        assert prof["comms"]["enviados"] > 0, f"{nombre} no envió mensajes FIPA-ACL"
        assert prof["votaciones"] > 0, f"{nombre} no ejecutó votaciones de Elección Social"
        assert prof["max_ms"] < 40.0, f"{nombre} excedió 40 ms/turno: {prof['max_ms']:.2f} ms"

    prof_c = res["por_nombre"]["COLMENA"]["profiling"]
    prof_d = res["por_nombre"]["CUERVO"]["profiling"]
    assert prof_c["vickrey"]["cfp_lote_enviados"] > 0 or prof_c["comms"]["lotes_emitidos"] > 0, (
        "COLMENA debe ejecutar subastas Vickrey / lotes"
    )
    assert (
        prof_d["cnp"]["cfp_hielo"] + prof_d["cnp"]["cfp_carrona"] + prof_d["cnp"]["cfp_rescate"]
    ) > 0, "CUERVO debe ejecutar subastas CNP"

    print(f"    Partida terminada en tick={res['ticks']} (fase_final={res['fase_final']}):")
    for fila in res["ranking"]:
        p = res["por_nombre"][fila["nombre"]]["profiling"]
        print(
            f"      {fila['letra']} ({fila['nombre']:10s}): Ratio={fila['ratio']:.5f} | "
            f"Hielo={fila['recursos']:3d} | Bat={fila['bateria']:6.1f}% | "
            f"Act={fila['activas']:2d} | max_ms={p.get('max_ms', 0.0):.2f}"
        )
    print(f"    COLMENA -> lotes={prof_c['comms']['lotes_emitidos']}, vickrey={prof_c['vickrey']}, votaciones={prof_c['votaciones']}")
    print(f"    CUERVO  -> cnp={prof_d['cnp']}, hook_d={prof_d['hook_d']}, votaciones={prof_d['votaciones']}")


# -----------------------------------------------------------------------------
# Torneo completo 4-Way y estrés 400 unidades (--full)
# -----------------------------------------------------------------------------
def correr_torneo_completo() -> None:
    seeds = [100, 117, 134, 151, 168, 185, 202, 219]
    tareas = [(sd, idx % 4, 1600) for idx, sd in enumerate(seeds)]
    workers = min(8, os.cpu_count() or 4)

    print(f"\n[A] Torneo 4-Way en paralelo ({workers} workers, 8 semillas con rotación de esquinas)...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        partidas = list(pool.map(_worker_torneo, tareas))

    facciones = ("LA AGENCIA", "RAVAGERS", "COLMENA", "CUERVO")
    acum: Dict[str, List[Dict[str, Any]]] = {n: [] for n in facciones}

    for i, res in enumerate(partidas, start=1):
        resumen_line = " | ".join(
            f"{f['nombre'][:3]}:{f['ratio']:.4f}({idx}º)"
            for idx, f in enumerate(res["ranking"], start=1)
        )
        print(f"  Partida {i:02d} (seed={res['seed']}, rot={res['rotacion']}, ticks={res['ticks']:4d}): {resumen_line}")
        for nom in facciones:
            acum[nom].append(res["por_nombre"][nom])

    print("\n=== RESUMEN TORNEO 4-WAY (200 unidades, 8 partidas con rotación) ===")
    print(
        f"{'Facción':12s} | {'1.º':>4s} | {'PosMed':>6s} | {'RatioMed':>8s} | {'Min-Max Ratio':>15s} | "
        f"{'Hielo':>6s} | {'Bat%':>7s} | {'Msgs':>6s} | {'p99_ms':>7s} | {'max_ms':>7s}"
    )
    print("-" * 105)
    for nom in facciones:
        items = acum[nom]
        wins = sum(1 for x in items if x["posicion"] == 1)
        pos_med = statistics.mean(x["posicion"] for x in items)
        ratios = [x["ratio"] for x in items]
        hielo_med = statistics.mean(x["recursos"] for x in items)
        bat_med = statistics.mean(x["bateria"] for x in items)
        msgs_vals = [
            x["profiling"].get("comms", {}).get("enviados", x["profiling"].get("mensajes", 0))
            for x in items
        ]
        msgs_med = statistics.mean(msgs_vals) if msgs_vals else 0.0
        p99_max = max(x["profiling"].get("p99_ms", 0.0) for x in items)
        max_ms = max(x["profiling"].get("max_ms", 0.0) for x in items)
        print(
            f"{nom:12s} | {wins:4d} | {pos_med:6.2f} | {statistics.mean(ratios):8.5f} | "
            f"{min(ratios):.4f}-{max(ratios):.4f}  | {hielo_med:6.1f} | {bat_med:7.1f} | "
            f"{msgs_med:6.0f} | {p99_max:7.3f} | {max_ms:7.3f}"
        )

    print("\n[B] Prueba de estrés 400 unidades concurrentes (100/equipo, seed=100, max_ticks=1200)...")
    res400 = ejecutar_partida_4way(seed=100, rotacion=0, max_ticks=1200, usar_400=True)
    for nom in facciones:
        info = res400["por_nombre"][nom]
        p = info["profiling"]
        print(
            f"  {nom:12s} -> Ratio={info['ratio']:.5f} | Hielo={info['recursos']:3d} | "
            f"mean={p.get('mean_ms', 0.0):.3f} ms | p95={p.get('p95_ms', 0.0):.3f} ms | "
            f"p99={p.get('p99_ms', 0.0):.3f} ms | max={p.get('max_ms', 0.0):.3f} ms"
        )
        if nom in ("COLMENA", "CUERVO"):
            assert p.get("p99_ms", 999.0) < 40.0, f"{nom} excedió p99 < 40ms en 400u: {p.get('p99_ms')}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark 4-way y verificación de equipoC/equipoD")
    parser.add_argument("--quick", action="store_true", help="Pruebas funcionales rápidas")
    parser.add_argument("--full", action="store_true", help="Torneo 8 semillas + estrés 400 unidades")
    args = parser.parse_args()

    hashes_0 = verificar_integridad_archivos_base()
    print(f"SHA-256 simulador.py:   {hashes_0['simulador.py']}")
    print(f"SHA-256 equipo_base.py: {hashes_0['equipo_base.py']}")

    try:
        if args.quick or not args.full:
            correr_pruebas_rapidas()
        if args.full:
            correr_torneo_completo()
    finally:
        verificar_integridad_archivos_base(hashes_0)
        print("Verificación final SHA-256: simulador.py y equipo_base.py 100% INTACTOS.")


if __name__ == "__main__":
    main()
