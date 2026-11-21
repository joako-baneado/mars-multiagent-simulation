"""
simulador.py
============
Simulación multiagente competitiva en Marte (60x45) usando Pygame.

Controles durante la ejecución:
  - ESPACIO : Pausar / Reanudar
  - + / -   : Cambiar velocidad de visualización (1x, 2x, 4x)
  - R       : Reiniciar partida
  - ESC     : Salir
"""

from __future__ import annotations

import importlib
import math
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pygame

from equipo_base import (
    CAPACIDAD_ROVER,
    GRID_H,
    GRID_W,
    UMBRAL_BATERIA_BAJA,
    EquipoBase,
    Percepcion,
    UnidadVista,
    clamp,
    distancia_chebyshev,
)

# -----------------------------------------------------------------------------
# Configuración visual y constantes de simulación
# -----------------------------------------------------------------------------
TICK_DT = 0.12  # Segundos por tick lógico (~8.33 ticks/s)
TORMENTA_RADIO = 5.0
TORMENTA_PULSO_INTERVALO = 0.85  # Cada cuántos seg de sim aplica empuje y -3% bat
DURACION_FASE_FINAL = 30.0
TICKS_CONSUMO_RESTOS = 4

COLORES_EQUIPO = {
    "A": (255, 120, 25),   # Naranja fuego
    "B": (55, 130, 255),   # Azul cobalto
    "C": (245, 205, 30),   # Amarillo ámbar
    "D": (235, 55, 165),   # Magenta
}

BASES_EQUIPO = {
    "A": (0, 0),
    "B": (GRID_W - 1, 0),
    "C": (0, GRID_H - 1),
    "D": (GRID_W - 1, GRID_H - 1),
}

COLOR_BLANCO = (246, 246, 246)
COLOR_NEGRO = (18, 18, 22)
COLOR_MARRON = (145, 85, 40)
COLOR_ROJO_ALERTA = (255, 35, 35)
COLOR_VERDE_RECARGA = (40, 255, 95)


# -----------------------------------------------------------------------------
# Estructuras internas del simulador
# -----------------------------------------------------------------------------
@dataclass
class UnidadSim:
    id: str
    equipo: str
    tipo: str  # "rover" | "dron_explorador" | "dron_recarga"
    pos: Tuple[int, int]
    prev_pos: Tuple[int, int]
    bateria: float = 100.0
    carga: int = 0
    capacidad: int = 0
    estado: str = "activo"  # "activo" | "esperando_recarga" | "consumiendo" | "muerto"
    carga_recarga: int = 0
    move_timer: float = 0.0
    move_interval: float = TICK_DT
    interp_t0: float = 0.0
    interp_dur: float = TICK_DT
    angulo: float = 0.0
    consumo_ticks: int = 0
    consumo_target_id: Optional[str] = None
    consumo_modo: Any = "propia"
    en_tormenta: bool = False
    bandeja_mensajes: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RestosSim:
    id: str
    pos: Tuple[int, int]
    equipo_origen: str
    tipo_origen: str
    combustible: int  # 20 rover, 10 dron


@dataclass
class TormentaSim:
    id: str
    modo: str  # "orbita_bases" | "centro_aleatorio" | "aleatorio"
    fx: float
    fy: float
    vx: float = 0.0
    vy: float = 0.0
    radio: float = TORMENTA_RADIO
    waypoint_idx: int = 0
    angulo_orbita: float = 0.0
    fase_rotacion: float = 0.0


class CommsAPI:
    """API controlada pasada a cada unidad para mensajería con costo real de batería."""

    def __init__(self, sim: "SimuladorMarte", emisor: UnidadSim) -> None:
        self.sim = sim
        self.emisor = emisor

    def enviar_mensaje(self, dest_id: str, asunto: str, datos: Dict[str, Any]) -> bool:
        return self.sim.enviar_mensaje_unidad(self.emisor, dest_id, asunto, datos)

    def enviar_a_recargador_cercano(self, asunto: str, datos: Dict[str, Any]) -> Optional[str]:
        candidatos = [
            u for u in self.sim.unidades
            if u.equipo == self.emisor.equipo
            and u.tipo == "dron_recarga"
            and u.estado != "muerto"
            and u.bateria > 2.0
        ]
        if not candidatos:
            return None

        # Priorizar recargadores disponibles con carga de recarga lista
        candidatos.sort(
            key=lambda u: (
                0 if u.carga_recarga > 0 else 1,
                distancia_chebyshev(self.emisor.pos, u.pos),
            )
        )
        destino = candidatos[0]
        if self.sim.enviar_mensaje_unidad(self.emisor, destino.id, asunto, datos):
            return destino.id
        return None

    def enviar_a_rover_ocioso_cercano(
        self,
        asunto: str,
        datos: Dict[str, Any],
        pos_ref: Optional[Tuple[int, int]] = None,
    ) -> Optional[str]:
        ref = pos_ref if pos_ref is not None else self.emisor.pos
        candidatos = [
            u for u in self.sim.unidades
            if u.equipo == self.emisor.equipo
            and u.tipo == "rover"
            and u.estado == "activo"
            and u.bateria > UMBRAL_BATERIA_BAJA
            and u.carga < u.capacidad
        ]
        if not candidatos:
            return None
        candidatos.sort(key=lambda u: (0 if u.carga == 0 else 1, distancia_chebyshev(ref, u.pos)))
        destino = candidatos[0]
        if self.sim.enviar_mensaje_unidad(self.emisor, destino.id, asunto, datos):
            return destino.id
        return None

    def enviar_a_aliados_cercanos(
        self,
        radio: int,
        asunto: str,
        datos: Dict[str, Any],
        max_dest: int = 2,
    ) -> int:
        candidatos = [
            u for u in self.sim.unidades
            if u.equipo == self.emisor.equipo
            and u.id != self.emisor.id
            and u.estado != "muerto"
            and distancia_chebyshev(self.emisor.pos, u.pos) <= radio
        ]
        candidatos.sort(key=lambda u: distancia_chebyshev(self.emisor.pos, u.pos))
        enviados = 0
        for dest in candidatos[:max_dest]:
            if self.emisor.bateria <= 0.5:
                break
            if self.sim.enviar_mensaje_unidad(self.emisor, dest.id, asunto, datos):
                enviados += 1
        return enviados


# -----------------------------------------------------------------------------
# Carga dinámica de equipos equipoA.py .. equipoD.py
# -----------------------------------------------------------------------------
def cargar_equipos_dinamicos() -> Dict[str, EquipoBase]:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)

    equipos: Dict[str, EquipoBase] = {}
    for letra in ("A", "B", "C", "D"):
        mod_name = f"equipo{letra}"
        try:
            if mod_name in sys.modules:
                modulo = importlib.reload(sys.modules[mod_name])
            else:
                modulo = importlib.import_module(mod_name)

            clase_elegida = None
            for attr_name in dir(modulo):
                obj = getattr(modulo, attr_name)
                if (
                    isinstance(obj, type)
                    and issubclass(obj, EquipoBase)
                    and obj is not EquipoBase
                ):
                    clase_elegida = obj
                    if attr_name.lower() == f"equipo{letra.lower()}":
                        break

            if clase_elegida is not None:
                equipos[letra] = clase_elegida()
            else:
                inst = EquipoBase()
                inst.nombre = f"Equipo {letra} (Base)"
                equipos[letra] = inst
        except Exception as exc:
            print(f"[AVISO] No se pudo cargar {mod_name}.py ({exc}). Usando EquipoBase.")
            inst = EquipoBase()
            inst.nombre = f"Equipo {letra} (Fallback)"
            equipos[letra] = inst

    return equipos


# -----------------------------------------------------------------------------
# Motor de simulación
# -----------------------------------------------------------------------------
class SimuladorMarte:
    def __init__(self) -> None:
        self.equipos: Dict[str, EquipoBase] = cargar_equipos_dinamicos()
        self.unidades: List[UnidadSim] = []
        self.unidades_por_id: Dict[str, UnidadSim] = {}
        self.hielo: Dict[Tuple[int, int], int] = {}
        self.hielo_inicial: int = 0
        self.restos: Dict[str, RestosSim] = {}
        self.restos_consumidos: List[Tuple[Tuple[int, int], str]] = []  # (pos, equipo_consumidor)
        self.tormentas: List[TormentaSim] = []

        self.stats_equipo: Dict[str, Dict[str, float]] = {}
        self.tiempo_sim: float = 0.0
        self.tick_count: int = 0
        self.acumulador_tick: float = 0.0
        self.acumulador_tormenta: float = 0.0

        self.fase_final: bool = False
        self.tiempo_final_restante: float = DURACION_FASE_FINAL
        self.simulacion_terminada: bool = False

        # Partículas ambientales de Marte para efecto pseudorrealista
        self.particulas_polvo: List[List[float]] = []
        self.huellas_rovers: List[Tuple[Tuple[int, int], float]] = []

        self.reiniciar()

    def reiniciar(self) -> None:
        self.equipos = cargar_equipos_dinamicos()
        self.unidades.clear()
        self.unidades_por_id.clear()
        self.hielo.clear()
        self.restos.clear()
        self.restos_consumidos.clear()
        self.tormentas.clear()
        self.huellas_rovers.clear()

        self.tiempo_sim = 0.0
        self.tick_count = 0
        self.acumulador_tick = 0.0
        self.acumulador_tormenta = 0.0
        self.fase_final = False
        self.tiempo_final_restante = DURACION_FASE_FINAL
        self.simulacion_terminada = False

        self.stats_equipo = {
            letra: {
                "recursos_en_base": 0.0,
                "bateria_usada": 0.0,
            }
            for letra in ("A", "B", "C", "D")
        }

        for letra, eq in self.equipos.items():
            eq.inicializar(letra, BASES_EQUIPO[letra], (GRID_W, GRID_H))

        self._generar_hielo()
        self._generar_unidades()
        self._generar_tormentas_iniciales()

        self.particulas_polvo = [
            [
                random.uniform(0.0, GRID_W),
                random.uniform(0.0, GRID_H),
                random.uniform(1.5, 4.0),
                random.uniform(-0.3, 0.4),
            ]
            for _ in range(90)
        ]

    def _generar_hielo(self) -> None:
        """Distribuye >= 600 u. de hielo (>= 150 u. por cuadrante de equipo) en celdas de 1-5 u."""
        cuadrantes = [
            (3, GRID_W // 2 - 1, 3, GRID_H // 2 - 1),
            (GRID_W // 2, GRID_W - 4, 3, GRID_H // 2 - 1),
            (3, GRID_W // 2 - 1, GRID_H // 2, GRID_H - 4),
            (GRID_W // 2, GRID_W - 4, GRID_H // 2, GRID_H - 4),
        ]
        total = 0
        for x0, x1, y0, y1 in cuadrantes:
            acum_cuadrante = 0
            intentos = 0
            while acum_cuadrante < 165 and intentos < 1000:
                intentos += 1
                x = random.randint(x0, x1)
                y = random.randint(y0, y1)
                # Evitar bases
                if any(distancia_chebyshev((x, y), b) <= 4 for b in BASES_EQUIPO.values()):
                    continue
                if (x, y) in self.hielo:
                    continue
                val = random.randint(1, 5)
                self.hielo[(x, y)] = val
                acum_cuadrante += val
                total += val

        # Extra de hielo disputado en el centro
        for _ in range(25):
            x = random.randint(GRID_W // 2 - 7, GRID_W // 2 + 7)
            y = random.randint(GRID_H // 2 - 5, GRID_H // 2 + 5)
            if (x, y) not in self.hielo:
                val = random.randint(2, 5)
                self.hielo[(x, y)] = val
                total += val

        self.hielo_inicial = total

    def _generar_unidades(self) -> None:
        """Crea exactamente 50 unidades por equipo: 30 rovers, 15 drones expl., 5 drones recarga."""
        for letra in ("A", "B", "C", "D"):
            bx, by = BASES_EQUIPO[letra]

            def pos_spawn(radio_max: int) -> Tuple[int, int]:
                dx = random.randint(-radio_max, radio_max)
                dy = random.randint(-radio_max, radio_max)
                return (
                    int(clamp(bx + dx, 0, GRID_W - 1)),
                    int(clamp(by + dy, 0, GRID_H - 1)),
                )

            # 30 Rovers (más lentos: se mueven cada 2 ticks)
            for i in range(30):
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

            # 15 Drones de exploración (2x más rápidos que rovers)
            for i in range(15):
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

            # 5 Drones de recarga (salen de base al ser solicitados)
            for i in range(5):
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

    def _generar_tormentas_iniciales(self) -> None:
        cx, cy = GRID_W / 2.0, GRID_H / 2.0
        # T1: orbita bordes pasando por las 4 bases
        self.tormentas.append(
            TormentaSim(id="T1", modo="orbita_bases", fx=cx, fy=cy, waypoint_idx=0)
        )
        # T2: gira aleatoria cerca del centro
        self.tormentas.append(
            TormentaSim(id="T2", modo="centro_aleatorio", fx=cx, fy=cy, angulo_orbita=0.0)
        )
        # T3: movimiento totalmente aleatorio
        ang = random.uniform(0, math.tau)
        self.tormentas.append(
            TormentaSim(
                id="T3",
                modo="aleatorio",
                fx=cx,
                fy=cy,
                vx=math.cos(ang) * 4.5,
                vy=math.sin(ang) * 4.5,
            )
        )

    def _consumir_bateria(self, u: UnidadSim, cantidad: float) -> None:
        if u.estado == "muerto" or cantidad <= 0:
            return
        gasto_real = min(u.bateria, cantidad)
        u.bateria -= gasto_real
        self.stats_equipo[u.equipo]["bateria_usada"] += gasto_real
        if u.bateria <= 0.0001:
            self._matar_unidad(u)

    def _matar_unidad(self, u: UnidadSim) -> None:
        if u.estado == "muerto":
            return
        u.bateria = 0.0
        u.estado = "muerto"
        u.carga = 0
        combustible = 20 if u.tipo == "rover" else 10
        rid = f"restos_{u.id}_{self.tick_count}"
        self.restos[rid] = RestosSim(
            id=rid,
            pos=u.pos,
            equipo_origen=u.equipo,
            tipo_origen=u.tipo,
            combustible=combustible,
        )

    def enviar_mensaje_unidad(
        self, emisor: UnidadSim, dest_id: str, asunto: str, datos: Dict[str, Any]
    ) -> bool:
        if emisor.estado == "muerto" or emisor.bateria <= 0.0:
            return False
        dest = self.unidades_por_id.get(dest_id)
        if dest is None or dest.estado == "muerto" or dest.bateria <= 0.0:
            return False

        # -0.5% emisor y -0.5% receptor
        self._consumir_bateria(emisor, 0.5)
        self._consumir_bateria(dest, 0.5)
        if dest.estado != "muerto":
            dest.bandeja_mensajes.append(
                {"emisor_id": emisor.id, "asunto": asunto, "datos": datos}
            )
        return True

    def _en_tormenta(self, pos: Tuple[int, int]) -> bool:
        px, py = pos
        for t in self.tormentas:
            if math.hypot(px - t.fx, py - t.fy) <= t.radio:
                return True
        return False

    def _construir_percepcion(self, u: UnidadSim) -> Percepcion:
        radio_vision = 5 if u.tipo == "rover" else (9 if u.tipo == "dron_explorador" else 7)
        ux, uy = u.pos

        hielo_vis = []
        for (hx, hy), cant in self.hielo.items():
            if cant > 0 and max(abs(hx - ux), abs(hy - uy)) <= radio_vision:
                hielo_vis.append((hx, hy, cant))
        hielo_vis.sort(key=lambda h: abs(h[0] - ux) + abs(h[1] - uy))

        restos_vis = []
        for r in self.restos.values():
            if max(abs(r.pos[0] - ux), abs(r.pos[1] - uy)) <= radio_vision:
                restos_vis.append(
                    {
                        "id": r.id,
                        "pos": r.pos,
                        "equipo": r.equipo_origen,
                        "tipo": r.tipo_origen,
                        "combustible": r.combustible,
                    }
                )

        aliados_vis = []
        rivales_vis = []
        for otro in self.unidades:
            if otro.estado == "muerto":
                continue
            if max(abs(otro.pos[0] - ux), abs(otro.pos[1] - uy)) <= radio_vision:
                info = {
                    "id": otro.id,
                    "pos": otro.pos,
                    "equipo": otro.equipo,
                    "tipo": otro.tipo,
                    "bateria": otro.bateria,
                    "estado": otro.estado,
                }
                if otro.equipo == u.equipo:
                    aliados_vis.append(info)
                else:
                    rivales_vis.append(info)

        tormentas_vis = [
            (t.fx, t.fy, t.radio)
            for t in self.tormentas
            if math.hypot(t.fx - ux, t.fy - uy) <= radio_vision + t.radio + 3.0
        ]

        msgs = list(u.bandeja_mensajes)
        u.bandeja_mensajes.clear()

        base_pos = BASES_EQUIPO[u.equipo]
        return Percepcion(
            tick=self.tick_count,
            base_pos=base_pos,
            grid_size=(GRID_W, GRID_H),
            fase_final=self.fase_final,
            tiempo_final_restante=self.tiempo_final_restante,
            base_aliada_cerca=distancia_chebyshev(u.pos, base_pos) <= 2,
            hielo_visible=hielo_vis,
            restos_visible=restos_vis,
            aliados_visible=aliados_vis,
            rivales_visible=rivales_vis,
            tormentas_visible=tormentas_vis,
            mensajes=msgs,
        )

    def _actualizar_tormentas(self, dt: float) -> None:
        cx, cy = GRID_W / 2.0, GRID_H / 2.0
        waypoints_bases = [(3.0, 3.0), (GRID_W - 4.0, 3.0), (GRID_W - 4.0, GRID_H - 4.0), (3.0, GRID_H - 4.0)]

        for t in self.tormentas:
            t.fase_rotacion += dt * 3.5

            if t.modo == "orbita_bases":
                tx, ty = waypoints_bases[t.waypoint_idx % len(waypoints_bases)]
                dx, dy = tx - t.fx, ty - t.fy
                dist = math.hypot(dx, dy)
                vel = 6.2
                if dist < 1.2:
                    t.waypoint_idx = (t.waypoint_idx + 1) % len(waypoints_bases)
                elif dist > 0:
                    t.fx += (dx / dist) * vel * dt
                    t.fy += (dy / dist) * vel * dt

            elif t.modo == "centro_aleatorio":
                t.angulo_orbita += dt * random.uniform(0.9, 1.8) * random.choice([1.0, 1.0, 1.0, -0.4])
                radio_orb = 7.5 + 4.0 * math.sin(self.tiempo_sim * 0.7 + t.angulo_orbita * 0.5)
                target_x = cx + math.cos(t.angulo_orbita) * radio_orb
                target_y = cy + math.sin(t.angulo_orbita) * (radio_orb * 0.78)
                dx, dy = target_x - t.fx, target_y - t.fy
                dist = math.hypot(dx, dy)
                if dist > 0.1:
                    paso = min(dist, 5.2 * dt)
                    t.fx += (dx / dist) * paso
                    t.fy += (dy / dist) * paso

            else:  # "aleatorio"
                if random.random() < dt * 1.8:
                    ang = math.atan2(t.vy, t.vx) + random.uniform(-1.3, 1.3)
                    spd = random.uniform(3.5, 6.0)
                    t.vx = math.cos(ang) * spd
                    t.vy = math.sin(ang) * spd

                t.fx += t.vx * dt
                t.fy += t.vy * dt
                if t.fx < 2.0:
                    t.fx = 2.0
                    t.vx = abs(t.vx)
                elif t.fx > GRID_W - 3.0:
                    t.fx = GRID_W - 3.0
                    t.vx = -abs(t.vx)
                if t.fy < 2.0:
                    t.fy = 2.0
                    t.vy = abs(t.vy)
                elif t.fy > GRID_H - 3.0:
                    t.fy = GRID_H - 3.0
                    t.vy = -abs(t.vy)

    def _aplicar_pulso_tormentas(self) -> None:
        """Efectos en radio=5: empuje 1 celda hacia afuera y -3% batería adicional por exposición."""
        for u in self.unidades:
            if u.estado == "muerto":
                continue
            for t in self.tormentas:
                dx = u.pos[0] - t.fx
                dy = u.pos[1] - t.fy
                dist = math.hypot(dx, dy)
                if dist <= t.radio:
                    # -3% adicional por exposición
                    self._consumir_bateria(u, 3.0)
                    if u.estado == "muerto":
                        break
                    # Empuje 1 celda hacia afuera
                    if dist < 0.25:
                        px, py = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
                    else:
                        px = 1 if dx > 0.25 else (-1 if dx < -0.25 else 0)
                        py = 1 if dy > 0.25 else (-1 if dy < -0.25 else 0)
                        if px == 0 and py == 0:
                            px = 1 if dx >= 0 else -1
                    nx = int(clamp(u.pos[0] + px, 0, GRID_W - 1))
                    ny = int(clamp(u.pos[1] + py, 0, GRID_H - 1))
                    if (nx, ny) != u.pos:
                        u.prev_pos = u.pos
                        u.pos = (nx, ny)
                        u.interp_t0 = self.tiempo_sim
                        u.interp_dur = TICK_DT
                    break  # Máximo 1 empuje por pulso para evitar teletransportes extremos

    def _ejecutar_accion(self, u: UnidadSim, accion: Any) -> None:
        if u.estado == "muerto" or not isinstance(accion, tuple) or not accion:
            return

        tipo_acc = accion[0]

        # Si un rover está esperando recarga y no se recarga, permanece detenido
        if u.tipo == "rover" and u.estado == "esperando_recarga":
            return

        # Si estaba consumiendo carroña y decide moverse (ej. EquipoD huyendo), cancelar consumo
        if u.estado == "consumiendo" and tipo_acc == "mover":
            u.estado = "activo"
            u.consumo_ticks = 0
            u.consumo_target_id = None

        if tipo_acc == "solicitar_recarga" and u.tipo == "rover":
            u.estado = "esperando_recarga"
            return

        if tipo_acc == "mover" and len(accion) >= 3:
            dx = int(clamp(accion[1], -1, 1))
            dy = int(clamp(accion[2], -1, 1))
            if dx == 0 and dy == 0:
                return
            nx = int(clamp(u.pos[0] + dx, 0, GRID_W - 1))
            ny = int(clamp(u.pos[1] + dy, 0, GRID_H - 1))
            if (nx, ny) == u.pos:
                return

            en_torm = self._en_tormenta(u.pos) or self._en_tormenta((nx, ny))
            costo = 5.0 if en_torm else 1.0
            self._consumir_bateria(u, costo)
            if u.estado == "muerto":
                return

            if u.tipo == "rover":
                self.huellas_rovers.append((u.pos, self.tiempo_sim))

            u.prev_pos = u.pos
            u.pos = (nx, ny)
            u.interp_t0 = self.tiempo_sim
            u.interp_dur = u.move_interval
            u.angulo = math.atan2(ny - u.prev_pos[1], nx - u.prev_pos[0])

        elif tipo_acc == "extraer" and u.tipo == "rover":
            cant = self.hielo.get(u.pos, 0)
            if cant > 0 and u.carga < u.capacidad:
                self.hielo[u.pos] = cant - 1
                if self.hielo[u.pos] <= 0:
                    del self.hielo[u.pos]
                u.carga += 1

        elif tipo_acc == "depositar" and u.tipo == "rover":
            if distancia_chebyshev(u.pos, BASES_EQUIPO[u.equipo]) <= 2 and u.carga > 0:
                self.stats_equipo[u.equipo]["recursos_en_base"] += u.carga
                u.carga = 0

        elif tipo_acc == "recargar_en_base" and u.tipo in ("dron_explorador", "dron_recarga"):
            if distancia_chebyshev(u.pos, BASES_EQUIPO[u.equipo]) <= 2:
                u.bateria = 100.0
                if u.tipo == "dron_recarga":
                    u.carga_recarga = 1

        elif tipo_acc == "recargar_aliado" and u.tipo == "dron_recarga" and len(accion) >= 2:
            target_id = accion[1]
            aliado = self.unidades_por_id.get(target_id)
            if (
                aliado is not None
                and aliado.equipo == u.equipo
                and aliado.estado != "muerto"
                and u.carga_recarga > 0
                and distancia_chebyshev(u.pos, aliado.pos) <= 1
            ):
                aliado.bateria = 100.0
                aliado.estado = "activo"
                u.carga_recarga -= 1

        elif tipo_acc == "consumir_restos" and u.tipo == "rover" and len(accion) >= 2:
            rid = accion[1]
            modo = accion[2] if len(accion) >= 3 else "propia"
            r = self.restos.get(rid)
            if r is not None and distancia_chebyshev(u.pos, r.pos) <= 1:
                u.estado = "consumiendo"
                u.consumo_ticks = TICKS_CONSUMO_RESTOS
                u.consumo_target_id = rid
                u.consumo_modo = modo

    def _procesar_consumo_restos(self, u: UnidadSim) -> None:
        if u.estado != "consumiendo":
            return
        r = self.restos.get(u.consumo_target_id or "")
        if r is None:
            u.estado = "activo"
            u.consumo_ticks = 0
            u.consumo_target_id = None
            return

        u.consumo_ticks -= 1
        if u.consumo_ticks <= 0:
            combustible = float(r.combustible)
            modo = u.consumo_modo
            if isinstance(modo, tuple) and len(modo) == 2 and modo[0] == "transferir":
                aliado = self.unidades_por_id.get(modo[1])
                if aliado is not None and aliado.equipo == u.equipo and aliado.estado != "muerto":
                    aliado.bateria = min(100.0, aliado.bateria + combustible)
                    if aliado.estado == "esperando_recarga" and aliado.bateria > UMBRAL_BATERIA_BAJA:
                        aliado.estado = "activo"
                else:
                    u.bateria = min(100.0, u.bateria + combustible)
            else:
                u.bateria = min(100.0, u.bateria + combustible)

            # Registrar "X" del color del equipo consumidor y eliminar restos
            self.restos_consumidos.append((r.pos, u.equipo))
            del self.restos[r.id]
            u.estado = "activo"
            u.consumo_ticks = 0
            u.consumo_target_id = None

    def _tick_logico(self) -> None:
        self.tick_count += 1

        # Actualizar estado de tormenta en cada unidad
        for u in self.unidades:
            if u.estado != "muerto":
                u.en_tormenta = self._en_tormenta(u.pos)

        # Rovers actúan cada 2 ticks, Drones cada 1 tick (drones 2x más rápidos)
        for u in self.unidades:
            if u.estado == "muerto":
                continue

            es_turno_rover = (u.tipo == "rover" and self.tick_count % 2 == 0)
            es_turno_dron = (u.tipo in ("dron_explorador", "dron_recarga"))

            if not (es_turno_rover or es_turno_dron):
                continue

            if u.estado == "consumiendo":
                # Permitir que EquipoD pueda interrumpir consumo si ve rivales
                if u.equipo == "D":
                    perc = self._construir_percepcion(u)
                    if perc.rivales_visible:
                        accion = self.equipos[u.equipo].decidir_unidad(
                            UnidadVista(
                                id=u.id,
                                equipo=u.equipo,
                                tipo=u.tipo,
                                pos=u.pos,
                                bateria=u.bateria,
                                carga=u.carga,
                                capacidad=u.capacidad,
                                estado=u.estado,
                                en_tormenta=u.en_tormenta,
                                carga_recarga=u.carga_recarga,
                            ),
                            perc,
                            CommsAPI(self, u),
                        )
                        self._ejecutar_accion(u, accion)
                        continue
                self._procesar_consumo_restos(u)
                continue

            perc = self._construir_percepcion(u)
            vista = UnidadVista(
                id=u.id,
                equipo=u.equipo,
                tipo=u.tipo,
                pos=u.pos,
                bateria=u.bateria,
                carga=u.carga,
                capacidad=u.capacidad,
                estado=u.estado,
                en_tormenta=u.en_tormenta,
                carga_recarga=u.carga_recarga,
            )
            accion = self.equipos[u.equipo].decidir_unidad(vista, perc, CommsAPI(self, u))
            self._ejecutar_accion(u, accion)

    def actualizar(self, dt: float) -> None:
        if self.simulacion_terminada:
            return

        self.tiempo_sim += dt
        self._actualizar_tormentas(dt)

        # Actualizar partículas ambientales y desvanecer huellas
        for p in self.particulas_polvo:
            p[0] = (p[0] + p[2] * dt) % GRID_W
            p[1] = (p[1] + p[3] * dt) % GRID_H
        self.huellas_rovers = [
            (pos, t0) for (pos, t0) in self.huellas_rovers if self.tiempo_sim - t0 < 2.2
        ]

        # Pulsos de tormenta
        self.acumulador_tormenta += dt
        while self.acumulador_tormenta >= TORMENTA_PULSO_INTERVALO:
            self.acumulador_tormenta -= TORMENTA_PULSO_INTERVALO
            self._aplicar_pulso_tormentas()

        # Ticks lógicos
        self.acumulador_tick += dt
        while self.acumulador_tick >= TICK_DT:
            self.acumulador_tick -= TICK_DT
            self._tick_logico()

        # Verificar fin del hielo -> +2 tormentas en el centro y 30s exactos
        hielo_restante = sum(self.hielo.values())
        if not self.fase_final and hielo_restante <= 0:
            self.fase_final = True
            self.tiempo_final_restante = DURACION_FASE_FINAL
            cx, cy = GRID_W / 2.0, GRID_H / 2.0
            ang4 = random.uniform(0, math.tau)
            self.tormentas.append(
                TormentaSim(
                    id="T4",
                    modo="aleatorio",
                    fx=cx,
                    fy=cy,
                    vx=math.cos(ang4) * 5.8,
                    vy=math.sin(ang4) * 5.8,
                )
            )
            self.tormentas.append(
                TormentaSim(
                    id="T5",
                    modo="centro_aleatorio",
                    fx=cx,
                    fy=cy,
                    angulo_orbita=math.pi,
                )
            )

        if self.fase_final:
            self.tiempo_final_restante -= dt
            if self.tiempo_final_restante <= 0.0:
                self.tiempo_final_restante = 0.0
                self.simulacion_terminada = True

    def obtener_ranking(self) -> List[Dict[str, Any]]:
        filas = []
        for letra in ("A", "B", "C", "D"):
            rec = self.stats_equipo[letra]["recursos_en_base"]
            bat = self.stats_equipo[letra]["bateria_usada"]
            ratio = (rec / bat) if bat > 0.0001 else 0.0
            activas = sum(1 for u in self.unidades if u.equipo == letra and u.estado != "muerto")
            muertas = 50 - activas
            filas.append(
                {
                    "letra": letra,
                    "nombre": self.equipos[letra].nombre,
                    "color": COLORES_EQUIPO[letra],
                    "recursos": int(rec),
                    "bateria": bat,
                    "ratio": ratio,
                    "activas": activas,
                    "muertas": muertas,
                }
            )
        filas.sort(key=lambda f: (-f["ratio"], -f["recursos"], f["bateria"]))
        return filas


# -----------------------------------------------------------------------------
# Renderizado Pygame (Pseudorrealista + HUD + Popup Final)
# -----------------------------------------------------------------------------
class InterfazPygame:
    def __init__(self, sim: SimuladorMarte) -> None:
        pygame.init()
        pygame.font.init()
        pygame.display.set_caption("Simulación Multiagente en Marte — 60x45")

        info = pygame.display.Info()
        init_w = max(1100, info.current_w - 30) if info.current_w > 0 else 1366
        init_h = max(720, info.current_h - 70) if info.current_h > 0 else 800
        self.screen = pygame.display.set_mode((init_w, init_h), pygame.RESIZABLE)

        # Maximizar ventana al inicio de forma multiplataforma
        try:
            from pygame._sdl2.video import Window
            Window.from_display_module().maximize()
        except Exception:
            pass

        self.sim = sim
        self.clock = pygame.time.Clock()
        self.pausado = False
        self.velocidad = 1  # 1x por defecto (30s finales exactos)

        self.fondo_cache_size: Tuple[int, int] = (0, 0)
        self.fondo_marte: Optional[pygame.Surface] = None
        self.crateres_seed = self._generar_crateres_norm()

        self.font_xs = pygame.font.SysFont("dejavusans,arial,sans", 11)
        self.font_sm = pygame.font.SysFont("dejavusans,arial,sans", 13)
        self.font_md = pygame.font.SysFont("dejavusans,arial,sans", 16, bold=True)
        self.font_lg = pygame.font.SysFont("dejavusans,arial,sans", 22, bold=True)
        self.font_xl = pygame.font.SysFont("dejavusans,arial,sans", 28, bold=True)

        self.btn_reiniciar_rect = pygame.Rect(0, 0, 0, 0)
        self.btn_salir_rect = pygame.Rect(0, 0, 0, 0)

    def _generar_crateres_norm(self) -> List[Tuple[float, float, float, int]]:
        rng = random.Random(42)
        crateres = []
        for _ in range(38):
            crateres.append(
                (rng.uniform(0.05, 0.95), rng.uniform(0.05, 0.95), rng.uniform(0.015, 0.055), rng.randint(1, 3))
            )
        return crateres

    def _obtener_fondo_marte(self, w: int, h: int) -> pygame.Surface:
        if self.fondo_marte is not None and self.fondo_cache_size == (w, h):
            return self.fondo_marte

        surf = pygame.Surface((w, h))
        # Gradiente vertical cálido marciano
        for y in range(h):
            t = y / max(1, h - 1)
            r = int(84 + t * 34)
            g = int(35 + t * 18)
            b = int(24 + t * 10)
            pygame.draw.line(surf, (r, g, b), (0, y), (w, y))

        # Cráteres suaves pseudorrealistas
        for cxn, cyn, rn, tipo in self.crateres_seed:
            cx, cy = int(cxn * w), int(cyn * h)
            rad = max(6, int(rn * min(w, h)))
            sombra = pygame.Surface((rad * 2 + 4, rad * 2 + 4), pygame.SRCALPHA)
            pygame.draw.circle(sombra, (55, 22, 15, 75), (rad + 2, rad + 2), rad)
            pygame.draw.circle(sombra, (140, 68, 42, 60), (rad + 4, rad + 4), max(2, rad - 2), 2)
            if tipo == 2:
                pygame.draw.circle(sombra, (65, 28, 18, 50), (rad + 2, rad + 2), max(2, rad // 2))
            surf.blit(sombra, (cx - rad - 2, cy - rad - 2))

        # Grilla táctica 60x45 sutil
        cw = w / GRID_W
        ch = h / GRID_H
        grid_surf = pygame.Surface((w, h), pygame.SRCALPHA)
        for ix in range(GRID_W + 1):
            px = int(ix * cw)
            pygame.draw.line(grid_surf, (175, 95, 65, 32), (px, 0), (px, h), 1)
        for iy in range(GRID_H + 1):
            py = int(iy * ch)
            pygame.draw.line(grid_surf, (175, 95, 65, 32), (0, py), (w, py), 1)
        surf.blit(grid_surf, (0, 0))

        self.fondo_marte = surf
        self.fondo_cache_size = (w, h)
        return surf

    def _layout(self) -> Tuple[pygame.Rect, pygame.Rect, float]:
        win_w, win_h = self.screen.get_size()
        hud_w = max(310, min(380, int(win_w * 0.24)))
        margin = 10
        avail_w = max(300, win_w - hud_w - margin * 3)
        avail_h = max(240, win_h - margin * 2)

        cell = min(avail_w / GRID_W, avail_h / GRID_H)
        map_w = int(cell * GRID_W)
        map_h = int(cell * GRID_H)
        map_x = margin + (avail_w - map_w) // 2
        map_y = margin + (avail_h - map_h) // 2

        map_rect = pygame.Rect(map_x, map_y, map_w, map_h)
        hud_rect = pygame.Rect(win_w - hud_w - margin, margin, hud_w, win_h - margin * 2)
        return map_rect, hud_rect, cell

    def _dibujar_bases(self, map_rect: pygame.Rect, cell: float) -> None:
        pulso = 0.5 + 0.5 * math.sin(self.sim.tiempo_sim * 3.0)
        for letra, (bx, by) in BASES_EQUIPO.items():
            color = COLORES_EQUIPO[letra]
            cx = map_rect.x + (bx + 0.5) * cell
            cy = map_rect.y + (by + 0.5) * cell

            # Zona de plataforma 2.6 celdas
            pad_r = max(14, int(cell * 1.9))
            pad_surf = pygame.Surface((pad_r * 2 + 4, pad_r * 2 + 4), pygame.SRCALPHA)
            pygame.draw.circle(pad_surf, (*color, int(45 + 35 * pulso)), (pad_r + 2, pad_r + 2), pad_r)
            pygame.draw.circle(pad_surf, (*color, 210), (pad_r + 2, pad_r + 2), int(pad_r * 0.65), 2)
            self.screen.blit(pad_surf, (int(cx - pad_r - 2), int(cy - pad_r - 2)))

            # Núcleo de base
            nuc_r = max(6, int(cell * 0.55))
            pygame.draw.circle(self.screen, (28, 30, 38), (int(cx), int(cy)), nuc_r)
            pygame.draw.circle(self.screen, color, (int(cx), int(cy)), nuc_r, 2)
            lbl = self.font_xs.render(letra, True, (255, 255, 255))
            self.screen.blit(lbl, lbl.get_rect(center=(int(cx), int(cy))))

    def _dibujar_hielo(self, map_rect: pygame.Rect, cell: float) -> None:
        for (hx, hy), cant in self.sim.hielo.items():
            if cant <= 0:
                continue
            # Celeste más oscuro a mayor cantidad (1..5)
            t = (cant - 1) / 4.0
            r = int(175 - t * 165)
            g = int(235 - t * 110)
            b = int(255 - t * 45)

            rx = int(map_rect.x + hx * cell + 1)
            ry = int(map_rect.y + hy * cell + 1)
            rw = max(6, int(cell) - 2)
            rh = max(6, int(cell) - 2)
            rect = pygame.Rect(rx, ry, rw, rh)
            pygame.draw.rect(self.screen, (r, g, b), rect, border_radius=3)
            pygame.draw.rect(self.screen, (230, 250, 255), rect, 1, border_radius=3)

            txt_str = str(cant)
            sombra = self.font_xs.render(txt_str, True, (10, 35, 65))
            txt = self.font_xs.render(txt_str, True, (255, 255, 255))
            self.screen.blit(sombra, sombra.get_rect(center=(rect.centerx + 1, rect.centery + 1)))
            self.screen.blit(txt, txt.get_rect(center=rect.center))

    def _dibujar_restos_consumidos(self, map_rect: pygame.Rect, cell: float) -> None:
        s = max(4, int(cell * 0.30))
        for (rx, ry), eq in self.sim.restos_consumidos:
            cx = int(map_rect.x + (rx + 0.5) * cell)
            cy = int(map_rect.y + (ry + 0.5) * cell)
            color = COLORES_EQUIPO[eq]
            pygame.draw.line(self.screen, (15, 10, 10), (cx - s + 1, cy - s + 1), (cx + s + 1, cy + s + 1), 3)
            pygame.draw.line(self.screen, (15, 10, 10), (cx + s + 1, cy - s + 1), (cx - s + 1, cy + s + 1), 3)
            pygame.draw.line(self.screen, color, (cx - s, cy - s), (cx + s, cy + s), 3)
            pygame.draw.line(self.screen, color, (cx + s, cy - s), (cx - s, cy + s), 3)

    def _dibujar_triangulo(
        self, cx: float, cy: float, r: float, fill: Tuple[int, int, int], border: Optional[Tuple[int, int, int]] = None
    ) -> None:
        pts = [
            (cx, cy - r * 1.08),
            (cx - r * 0.94, cy + r * 0.78),
            (cx + r * 0.94, cy + r * 0.78),
        ]
        pygame.draw.polygon(self.screen, fill, pts)
        if border is not None:
            pygame.draw.polygon(self.screen, border, pts, 2)

    def _dibujar_unidades(self, map_rect: pygame.Rect, cell: float) -> None:
        ahora = self.sim.tiempo_sim

        # Huellas suaves de rovers
        for (hx, hy), t0 in self.sim.huellas_rovers:
            alpha = int(max(0, 1.0 - (ahora - t0) / 2.2) * 70)
            px = int(map_rect.x + (hx + 0.5) * cell)
            py = int(map_rect.y + (hy + 0.5) * cell)
            hs = pygame.Surface((6, 6), pygame.SRCALPHA)
            pygame.draw.circle(hs, (45, 18, 12, alpha), (3, 3), 2)
            self.screen.blit(hs, (px - 3, py - 3))

        # Ordenar: muertas debajo, luego rovers, luego drones encima
        orden = sorted(
            self.sim.unidades,
            key=lambda u: (0 if u.estado == "muerto" else (1 if u.tipo == "rover" else 2)),
        )

        for u in orden:
            if u.estado == "muerto":
                vx, vy = float(u.pos[0]), float(u.pos[1])
            else:
                t = clamp((ahora - u.interp_t0) / max(0.01, u.interp_dur), 0.0, 1.0)
                vx = u.prev_pos[0] + (u.pos[0] - u.prev_pos[0]) * t
                vy = u.prev_pos[1] + (u.pos[1] - u.prev_pos[1]) * t

            cx = map_rect.x + (vx + 0.5) * cell
            cy = map_rect.y + (vy + 0.5) * cell
            team_color = COLORES_EQUIPO[u.equipo]

            if u.estado == "muerto":
                r = max(5.0, cell * 0.36)
                if u.tipo == "rover":
                    pygame.draw.circle(self.screen, COLOR_NEGRO, (int(cx), int(cy)), int(r))
                    pygame.draw.circle(self.screen, team_color, (int(cx), int(cy)), max(2, int(r * 0.38)))
                else:
                    self._dibujar_triangulo(cx, cy, r * 1.05, COLOR_NEGRO)
                    pygame.draw.circle(
                        self.screen, team_color, (int(cx), int(cy + r * 0.16)), max(2, int(r * 0.32))
                    )
                continue

            # Sombra suave (drones proyectan sombra más desplazada por altura)
            off = 4 if u.tipo != "rover" else 2
            sombra = pygame.Surface((int(cell * 0.9) + 4, int(cell * 0.9) + 4), pygame.SRCALPHA)
            sr = max(4, int(cell * 0.34))
            pygame.draw.circle(sombra, (0, 0, 0, 75), (sombra.get_width() // 2, sombra.get_height() // 2), sr)
            self.screen.blit(sombra, (int(cx - sombra.get_width() // 2 + off), int(cy - sombra.get_height() // 2 + off)))

            cuerpo_color = COLOR_MARRON if u.estado == "consumiendo" else COLOR_BLANCO

            if u.tipo == "rover":
                r = max(5, int(cell * 0.36))
                pygame.draw.circle(self.screen, cuerpo_color, (int(cx), int(cy)), r)
                pygame.draw.circle(self.screen, team_color, (int(cx), int(cy)), r, 2)

                # Alerta batería baja esperando recarga: anillo rojo width=2
                if u.estado == "esperando_recarga":
                    pygame.draw.circle(self.screen, COLOR_ROJO_ALERTA, (int(cx), int(cy)), r + 3, 2)

                # Indicador sutil de carga de hielo encima
                if u.carga > 0:
                    pygame.draw.circle(self.screen, (50, 215, 255), (int(cx), int(cy)), max(2, int(r * 0.35)))

            else:
                r = max(5.5, cell * 0.40)
                self._dibujar_triangulo(cx, cy, r, cuerpo_color, team_color)
                c_inner_y = int(cy + r * 0.16)
                if u.tipo == "dron_explorador":
                    # Círculo negro pequeño centro
                    pygame.draw.circle(self.screen, COLOR_NEGRO, (int(cx), c_inner_y), max(2, int(r * 0.28)))
                else:
                    # Cuadrado verde luminoso pequeño centro
                    sq = max(4, int(r * 0.50))
                    pygame.draw.rect(
                        self.screen,
                        COLOR_VERDE_RECARGA,
                        pygame.Rect(int(cx - sq // 2), int(c_inner_y - sq // 2), sq, sq),
                    )

            # Barra de batería arriba
            bw = max(10, int(cell * 0.80))
            bh = 3
            bx = int(cx - bw // 2)
            by = int(cy - cell * 0.50 - 3)
            pygame.draw.rect(self.screen, (20, 20, 25), (bx - 1, by - 1, bw + 2, bh + 2))
            ratio_b = clamp(u.bateria / 100.0, 0.0, 1.0)
            b_col = (55, 225, 95) if ratio_b > 0.5 else ((245, 190, 40) if ratio_b > 0.25 else (255, 50, 50))
            pygame.draw.rect(self.screen, b_col, (bx, by, int(bw * ratio_b), bh))

    def _dibujar_tormentas(self, map_rect: pygame.Rect, cell: float) -> None:
        overlay = pygame.Surface((map_rect.w, map_rect.h), pygame.SRCALPHA)
        for t in self.sim.tormentas:
            cx = int((t.fx + 0.5) * cell)
            cy = int((t.fy + 0.5) * cell)
            rad = int(t.radio * cell)

            # Capas de polvo translúcidas dentro de radio=5 celdas
            pygame.draw.circle(overlay, (195, 95, 45, 42), (cx, cy), rad)
            pygame.draw.circle(overlay, (215, 115, 55, 55), (cx, cy), int(rad * 0.72))
            pygame.draw.circle(overlay, (235, 135, 70, 68), (cx, cy), int(rad * 0.42))
            pygame.draw.circle(overlay, (255, 165, 95, 160), (cx, cy), rad, 2)

            # Espirales de polvo rotando
            for k in range(6):
                ang = t.fase_rotacion + k * (math.tau / 6.0)
                px = cx + int(math.cos(ang) * rad * 0.65)
                py = cy + int(math.sin(ang) * rad * 0.65)
                pygame.draw.circle(overlay, (255, 205, 145, 135), (px, py), max(2, int(cell * 0.18)))

            # Círculo marcador central
            cr = max(5, int(cell * 0.42))
            pygame.draw.circle(overlay, (255, 240, 210, 235), (cx, cy), cr, 2)
            pygame.draw.circle(overlay, (255, 110, 30, 235), (cx, cy), max(2, cr // 2))

        self.screen.blit(overlay, (map_rect.x, map_rect.y))

    def _dibujar_particulas_polvo(self, map_rect: pygame.Rect, cell: float) -> None:
        part_surf = pygame.Surface((map_rect.w, map_rect.h), pygame.SRCALPHA)
        for px, py, _, _ in self.sim.particulas_polvo:
            x = int(px * cell)
            y = int(py * cell)
            pygame.draw.circle(part_surf, (245, 185, 140, 55), (x, y), 1)
        self.screen.blit(part_surf, (map_rect.x, map_rect.y))

    def _dibujar_hud(self, hud_rect: pygame.Rect) -> None:
        pygame.draw.rect(self.screen, (20, 22, 30), hud_rect, border_radius=8)
        pygame.draw.rect(self.screen, (65, 72, 95), hud_rect, 2, border_radius=8)

        x = hud_rect.x + 14
        y = hud_rect.y + 12
        w = hud_rect.w - 28

        # Título
        titulo = self.font_lg.render("MISIÓN MARTE 60x45", True, (245, 245, 250))
        self.screen.blit(titulo, (x, y))
        y += 30

        # Estado de fase / tiempo
        hielo_rest = sum(self.sim.hielo.values())
        if self.sim.fase_final:
            fase_txt = f"TORMENTA FINAL: {self.sim.tiempo_final_restante:04.1f}s"
            fase_col = (255, 85, 65)
        elif self.pausado:
            fase_txt = "EN PAUSA"
            fase_col = (255, 210, 65)
        else:
            fase_txt = f"RECOLECCIÓN ({self.velocidad}x)"
            fase_col = (95, 230, 145)

        self.screen.blit(self.font_md.render(fase_txt, True, fase_col), (x, y))
        y += 24

        mins = int(self.sim.tiempo_sim // 60)
        segs = int(self.sim.tiempo_sim % 60)
        stats_glob = [
            f"Tiempo: {mins:02d}:{segs:02d}   Tormentas: {len(self.sim.tormentas)}",
            f"Hielo restante: {hielo_rest} / {self.sim.hielo_inicial} u.",
        ]
        for linea in stats_glob:
            self.screen.blit(self.font_sm.render(linea, True, (200, 210, 225)), (x, y))
            y += 18

        # Barra de progreso de hielo
        prog = hielo_rest / max(1, self.sim.hielo_inicial)
        pygame.draw.rect(self.screen, (35, 40, 55), (x, y + 2, w, 8), border_radius=4)
        pygame.draw.rect(self.screen, (40, 195, 245), (x, y + 2, int(w * prog), 8), border_radius=4)
        y += 18

        pygame.draw.line(self.screen, (55, 60, 80), (x, y), (x + w, y), 1)
        y += 10

        # Tarjetas por equipo
        card_h = min(110, max(82, (hud_rect.bottom - y - 95) // 4))
        for letra in ("A", "B", "C", "D"):
            color = COLORES_EQUIPO[letra]
            rec = int(self.sim.stats_equipo[letra]["recursos_en_base"])
            bat = self.sim.stats_equipo[letra]["bateria_usada"]
            ratio = (rec / bat) if bat > 0.0001 else 0.0
            activas = sum(1 for u in self.sim.unidades if u.equipo == letra and u.estado != "muerto")
            muertas = 50 - activas

            card = pygame.Rect(x, y, w, card_h - 6)
            pygame.draw.rect(self.screen, (28, 32, 44), card, border_radius=6)
            pygame.draw.rect(self.screen, color, card, 2, border_radius=6)

            # Badge de color + nombre
            pygame.draw.rect(self.screen, color, (card.x + 8, card.y + 8, 12, 16), border_radius=3)
            nom = self.sim.equipos[letra].nombre[:24]
            self.screen.blit(self.font_md.render(nom, True, (245, 245, 250)), (card.x + 26, card.y + 7))

            l1 = f"Hielo: {rec} u.   |   Bat. usada: {bat:.1f}%"
            l2 = f"Activas: {activas}  |  Muertas: {muertas}  |  Ratio: {ratio:.3f}"
            self.screen.blit(self.font_sm.render(l1, True, (215, 225, 240)), (card.x + 10, card.y + 30))
            self.screen.blit(self.font_sm.render(l2, True, (185, 200, 220)), (card.x + 10, card.y + 50))

            y += card_h

        # Controles / Leyenda resumida abajo
        y_leg = hud_rect.bottom - 84
        pygame.draw.line(self.screen, (55, 60, 80), (x, y_leg), (x + w, y_leg), 1)
        leyenda = [
            "Círculo=Rover | Triáng=Dron (Negro=Expl, Verde=Rec)",
            "Anillo Rojo=Bat.Baja | Marrón=Carroña | X=Consumido",
            "[ESPACIO] Pausa  [+/-] Vel  [R] Reiniciar  [ESC] Salir",
        ]
        ly = y_leg + 8
        for txt in leyenda:
            self.screen.blit(self.font_xs.render(txt, True, (155, 168, 190)), (x, ly))
            ly += 19

    def _dibujar_popup_ranking(self) -> None:
        win_w, win_h = self.screen.get_size()
        velo = pygame.Surface((win_w, win_h), pygame.SRCALPHA)
        velo.fill((8, 10, 16, 195))
        self.screen.blit(velo, (0, 0))

        pw = min(760, win_w - 60)
        ph = min(460, win_h - 60)
        box = pygame.Rect((win_w - pw) // 2, (win_h - ph) // 2, pw, ph)
        pygame.draw.rect(self.screen, (24, 28, 40), box, border_radius=12)
        pygame.draw.rect(self.screen, (245, 195, 65), box, 3, border_radius=12)

        titulo = self.font_xl.render("RANKING FINAL — SIMULACIÓN COMPLETADA", True, (255, 220, 95))
        self.screen.blit(titulo, titulo.get_rect(centerx=box.centerx, top=box.top + 22))

        sub = self.font_sm.render("Orden descendente por Ratio = Recursos en Base / Batería Total Usada", True, (190, 200, 220))
        self.screen.blit(sub, sub.get_rect(centerx=box.centerx, top=box.top + 60))

        ranking = self.sim.obtener_ranking()
        ry = box.top + 102
        for idx, fila in enumerate(ranking, start=1):
            row_rect = pygame.Rect(box.x + 28, ry, pw - 56, 58)
            bg_col = (40, 46, 66) if idx == 1 else (30, 34, 48)
            pygame.draw.rect(self.screen, bg_col, row_rect, border_radius=8)
            pygame.draw.rect(self.screen, fila["color"], row_rect, 2, border_radius=8)

            pos_txt = self.font_lg.render(f"{idx}º", True, (255, 225, 95) if idx == 1 else (230, 235, 245))
            self.screen.blit(pos_txt, (row_rect.x + 14, row_rect.y + 15))

            pygame.draw.rect(self.screen, fila["color"], (row_rect.x + 56, row_rect.y + 16, 16, 26), border_radius=4)
            nom_txt = self.font_md.render(fila["nombre"], True, (250, 250, 255))
            self.screen.blit(nom_txt, (row_rect.x + 82, row_rect.y + 9))

            detalle = (
                f"Base: {fila['recursos']} u.  |  Bat. usada: {fila['bateria']:.1f}%  |  "
                f"Activas/Muertas: {fila['activas']}/{fila['muertas']}"
            )
            self.screen.blit(self.font_sm.render(detalle, True, (195, 208, 228)), (row_rect.x + 82, row_rect.y + 32))

            ratio_txt = self.font_lg.render(f"{fila['ratio']:.4f}", True, (100, 240, 160))
            self.screen.blit(ratio_txt, ratio_txt.get_rect(right=row_rect.right - 16, centery=row_rect.centery))

            ry += 68

        # Botones inferiores
        bw, bh = 210, 42
        self.btn_reiniciar_rect = pygame.Rect(box.centerx - bw - 14, box.bottom - 62, bw, bh)
        self.btn_salir_rect = pygame.Rect(box.centerx + 14, box.bottom - 62, bw, bh)

        pygame.draw.rect(self.screen, (45, 145, 95), self.btn_reiniciar_rect, border_radius=8)
        pygame.draw.rect(self.screen, (165, 55, 65), self.btn_salir_rect, border_radius=8)
        t_re = self.font_md.render("Reiniciar (R)", True, (255, 255, 255))
        t_sa = self.font_md.render("Salir (ESC)", True, (255, 255, 255))
        self.screen.blit(t_re, t_re.get_rect(center=self.btn_reiniciar_rect.center))
        self.screen.blit(t_sa, t_sa.get_rect(center=self.btn_salir_rect.center))

    def ejecutar(self) -> None:
        corriendo = True
        while corriendo:
            raw_dt = min(self.clock.tick(60) / 1000.0, 0.1)

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    corriendo = False
                elif ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        corriendo = False
                    elif ev.key == pygame.K_SPACE and not self.sim.simulacion_terminada:
                        self.pausado = not self.pausado
                    elif ev.key == pygame.K_r:
                        self.sim.reiniciar()
                        self.pausado = False
                    elif ev.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                        self.velocidad = min(4, self.velocidad * 2)
                    elif ev.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        self.velocidad = max(1, self.velocidad // 2)
                elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    if self.sim.simulacion_terminada:
                        if self.btn_reiniciar_rect.collidepoint(ev.pos):
                            self.sim.reiniciar()
                            self.pausado = False
                        elif self.btn_salir_rect.collidepoint(ev.pos):
                            corriendo = False

            if not self.pausado and not self.sim.simulacion_terminada:
                # Pasos pequeños para mantener exactitud de la física y los 30s finales
                sub_dt = raw_dt
                for _ in range(self.velocidad):
                    self.sim.actualizar(sub_dt)

            self.screen.fill((12, 13, 18))
            map_rect, hud_rect, cell = self._layout()

            fondo = self._obtener_fondo_marte(map_rect.w, map_rect.h)
            self.screen.blit(fondo, map_rect.topleft)

            self._dibujar_bases(map_rect, cell)
            self._dibujar_hielo(map_rect, cell)
            self._dibujar_restos_consumidos(map_rect, cell)
            self._dibujar_unidades(map_rect, cell)
            self._dibujar_tormentas(map_rect, cell)
            self._dibujar_particulas_polvo(map_rect, cell)
            pygame.draw.rect(self.screen, (115, 75, 60), map_rect, 2)

            self._dibujar_hud(hud_rect)

            if self.sim.simulacion_terminada:
                self._dibujar_popup_ranking()

            pygame.display.flip()

        pygame.quit()


def main() -> None:
    sim = SimuladorMarte()
    ui = InterfazPygame(sim)
    ui.ejecutar()


if __name__ == "__main__":
    main()