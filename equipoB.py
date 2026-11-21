"""
equipoB.py
==========
Inteligencia autónoma de RAVAGERS para la Simulación Multiagente Competitiva
en Marte (60x45), heredando estrictamente de `EquipoBase` (`equipo_base.py`).

Estrategia principal — Blitz Central Agresivo + Encadenamiento 5/5:
  - 65% de los drones exploradores barren velozmente hacia la zona central
    disputada (30, 22) +/- 10 para localizar vetas ricas (3-5 u.), mientras
    35% cubre el cuadrante propio. Tras completar su barrido de 3 waypoints,
    modulan su cadencia para disparar el Ratio.
  - 60% de los rovers (Grupo Blitz) avanzan al anillo central (radio 6..13)
    para capturar vetas ricas antes que los rivales; 40% (Grupo Cuadrante)
    limpia el cuadrante propio con viajes cortos.
  - Encadenamiento agresivo 5/5 (radio Chebyshev <= 12) antes de retornar al
    perímetro de depósito (distancia_chebyshev <= 2 de base_pos).
  - FIPA-ACL ultra-ligero: solo reporta vetas ricas (>= 3 u., cooldown 220) y
    restos de rover (+20 u. cercanos, cooldown 260) por `api`, manteniendo la
    memoria espacial 100% aislada por agente (`_MemoriaRavager`).
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from equipo_base import (
    GRID_H,
    GRID_W,
    UMBRAL_BATERIA_BAJA,
    EquipoBase,
    Percepcion,
    UnidadVista,
    clamp,
    distancia_chebyshev,
)

RADIO_VISION: Dict[str, int] = {
    "rover": 5,
    "dron_explorador": 9,
    "dron_recarga": 7,
}

TTL_HIELO = 125
TTL_RESTOS = 70
TTL_TORMENTA = 14
DECAY_LAMBDA = 0.015
MAX_EXPANSIONES_ASTAR = 24
UMBRAL_REPOSO_BASE = 140
UMBRAL_REPOSO_ROVER = 10
ELITE_ROVERS_COUNT = 6
ELITE_RECARGADORES_COUNT = 2


@dataclass
class _CreenciaHielo:
    pos: Tuple[int, int]
    cantidad: int
    tick_obs: int

    def confianza(self, tick_actual: int) -> float:
        return math.exp(-DECAY_LAMBDA * max(0, tick_actual - self.tick_obs))


@dataclass
class _CreenciaRestos:
    id: str
    pos: Tuple[int, int]
    combustible: int
    tick_obs: int


@dataclass
class _TrazaTormenta:
    fx: float
    fy: float
    vx: float
    vy: float
    radio: float
    tick_obs: int


@dataclass
class _MemoriaRavager:
    """Memoria espacial y de estado estrictamente privada para un agente."""

    uid: str
    tipo: str
    hielo: Dict[Tuple[int, int], _CreenciaHielo] = field(default_factory=dict)
    restos: Dict[str, _CreenciaRestos] = field(default_factory=dict)
    tormentas: List[_TrazaTormenta] = field(default_factory=list)
    waypoints: Optional[List[Tuple[int, int]]] = None
    patrulla_idx: int = 0
    ciclo_patrulla_completo: bool = False
    objetivo_hielo: Optional[Tuple[int, int]] = None
    objetivo_restos: Optional[Dict[str, Any]] = None
    rescate_actual: Optional[Dict[str, Any]] = None
    ticks_en_rescate: int = 0
    cooldowns: Dict[Any, int] = field(default_factory=dict)
    ultimo_ping_recarga: int = -999
    esperando_agree: bool = False
    en_reposo: bool = False
    modo_ultra_safe: bool = False
    hielo_depositado: int = 0


class EquipoB(EquipoBase):
    """
    Facción RAVAGERS: recolección masiva de hielo con Blitz Central Agresivo,
    encadenamiento 5/5, FIPA-ACL ultra-ligero y Modo Reposo de conservación.
    """

    nombre: str = "RAVAGERS"

    def __init__(self) -> None:
        super().__init__()
        self._agentes: Dict[str, _MemoriaRavager] = {}
        self.latencias_turno_ms: Dict[int, float] = {}
        self.mensajes_enviados: int = 0
        self._hielo_en_base: int = 0

    def inicializar(
        self,
        letra: str,
        base_pos: Tuple[int, int],
        grid_size: Tuple[int, int] = (GRID_W, GRID_H),
    ) -> None:
        super().inicializar(letra, base_pos, grid_size)
        self._agentes.clear()
        self.latencias_turno_ms.clear()
        self.mensajes_enviados = 0
        self._hielo_en_base = 0

    @staticmethod
    def _idx_uid(uid: str) -> int:
        try:
            return int(uid.split("_")[-1])
        except Exception:
            return abs(hash(uid)) % 100

    def _mem_privada(self, unidad: UnidadVista) -> _MemoriaRavager:
        if unidad.id not in self._agentes:
            self._agentes[unidad.id] = _MemoriaRavager(uid=unidad.id, tipo=unidad.tipo)
        return self._agentes[unidad.id]

    # -------------------------------------------------------------------------
    # Percepción, actualización de creencias y olvido por agente
    # -------------------------------------------------------------------------
    def _actualizar_percepcion(
        self, unidad: UnidadVista, percepcion: Percepcion, mem: _MemoriaRavager
    ) -> None:
        tick = percepcion.tick
        rv = RADIO_VISION.get(unidad.tipo, 5)
        ux, uy = unidad.pos

        hielo_vis: Set[Tuple[int, int]] = set()
        for hx, hy, cant in percepcion.hielo_visible:
            if cant > 0:
                pos = (hx, hy)
                hielo_vis.add(pos)
                mem.hielo[pos] = _CreenciaHielo(pos=pos, cantidad=cant, tick_obs=tick)

        borrar_h = [
            pos
            for pos, cr in mem.hielo.items()
            if (max(abs(pos[0] - ux), abs(pos[1] - uy)) <= rv and pos not in hielo_vis)
            or (tick - cr.tick_obs > TTL_HIELO)
            or (cr.confianza(tick) < 0.14)
        ]
        for pos in borrar_h:
            mem.hielo.pop(pos, None)
            if mem.objetivo_hielo == pos:
                mem.objetivo_hielo = None

        restos_vis: Set[str] = set()
        for r in percepcion.restos_visible:
            rid = r["id"]
            rpos = tuple(r["pos"])
            restos_vis.add(rid)
            mem.restos[rid] = _CreenciaRestos(
                id=rid,
                pos=rpos,
                combustible=int(r.get("combustible", 10)),
                tick_obs=tick,
            )

        borrar_r = [
            rid
            for rid, cr in mem.restos.items()
            if (max(abs(cr.pos[0] - ux), abs(cr.pos[1] - uy)) <= rv and rid not in restos_vis)
            or (tick - cr.tick_obs > TTL_RESTOS)
        ]
        for rid in borrar_r:
            mem.restos.pop(rid, None)
            if mem.objetivo_restos and mem.objetivo_restos.get("id") == rid:
                mem.objetivo_restos = None

        # Actualizar trazas de tormentas con velocidad estimada
        prev_t = [t for t in mem.tormentas if tick - t.tick_obs <= TTL_TORMENTA]
        nuevas_t: List[_TrazaTormenta] = []
        usadas: Set[int] = set()
        for tx, ty, rad in percepcion.tormentas_visible:
            mejor_i, mejor_d = -1, 6.5
            for i, pt in enumerate(prev_t):
                if i in usadas:
                    continue
                d = math.hypot(tx - pt.fx, ty - pt.fy)
                if d < mejor_d:
                    mejor_d, mejor_i = d, i
            vx, vy = 0.0, 0.0
            if mejor_i >= 0:
                usadas.add(mejor_i)
                pt = prev_t[mejor_i]
                dt = max(1, tick - pt.tick_obs)
                vx = clamp((tx - pt.fx) / dt, -1.5, 1.5)
                vy = clamp((ty - pt.fy) / dt, -1.5, 1.5)
            nuevas_t.append(_TrazaTormenta(fx=tx, fy=ty, vx=vx, vy=vy, radio=rad, tick_obs=tick))

        for i, pt in enumerate(prev_t):
            if i not in usadas and tick - pt.tick_obs <= 6:
                nuevas_t.append(
                    _TrazaTormenta(
                        fx=clamp(pt.fx + pt.vx, 0.0, self.grid_size[0] - 1.0),
                        fy=clamp(pt.fy + pt.vy, 0.0, self.grid_size[1] - 1.0),
                        vx=pt.vx,
                        vy=pt.vy,
                        radio=pt.radio,
                        tick_obs=pt.tick_obs,
                    )
                )
        mem.tormentas = nuevas_t

    # -------------------------------------------------------------------------
    # Comunicación FIPA-ACL ultra-ligera
    # -------------------------------------------------------------------------
    def _puede_emitir(self, mem: _MemoriaRavager, clave: Any, tick: int, cooldown: int) -> bool:
        ultimo = mem.cooldowns.get(clave, -9999)
        if tick - ultimo < cooldown:
            return False
        mem.cooldowns[clave] = tick
        return True

    def _enviar_acl_directo(
        self,
        api: Any,
        mem: _MemoriaRavager,
        dest_id: str,
        performative: str,
        ontology: str,
        conv_id: str,
        content: Dict[str, Any],
    ) -> bool:
        datos = {
            "performative": performative,
            "sender": mem.uid,
            "conversation_id": conv_id,
            "ontology": ontology,
            "content": content,
            **{k: v for k, v in content.items() if k in ("pos", "cantidad", "id", "rover_id")},
        }
        ok = api.enviar_mensaje(dest_id, f"{ontology}:{performative}", datos)
        if ok:
            self.mensajes_enviados += 1
        return ok

    def _procesar_mensajes(
        self, unidad: UnidadVista, percepcion: Percepcion, api: Any, mem: _MemoriaRavager
    ) -> None:
        tick = percepcion.tick
        for msg in percepcion.mensajes:
            emisor = msg.get("emisor_id", "")
            datos = msg.get("datos", {})
            perf = datos.get("performative", "INFORM")
            ont = datos.get("ontology", "resource-discovery")
            conv_id = datos.get("conversation_id", f"msg_{tick}")
            content = datos.get("content", datos)

            if ont == "resource-discovery" and perf == "INFORM":
                if "cantidad" in content and "pos" in content:
                    pos = tuple(content["pos"])
                    cant = int(content["cantidad"])
                    mem.hielo[pos] = _CreenciaHielo(pos=pos, cantidad=cant, tick_obs=tick)
                    if (
                        unidad.tipo == "rover"
                        and mem.objetivo_hielo is None
                        and unidad.carga < unidad.capacidad
                    ):
                        mem.objetivo_hielo = pos
                elif "id" in content and "pos" in content:
                    rid = content["id"]
                    rpos = tuple(content["pos"])
                    mem.restos[rid] = _CreenciaRestos(
                        id=rid,
                        pos=rpos,
                        combustible=int(content.get("combustible", 10)),
                        tick_obs=tick,
                    )
                    if (
                        unidad.tipo == "rover"
                        and mem.objetivo_restos is None
                        and unidad.bateria < 78.0
                        and unidad.carga == 0
                    ):
                        mem.objetivo_restos = {"id": rid, "pos": rpos}

            elif ont == "rescue-ops":
                if perf == "REQUEST" and unidad.tipo == "dron_recarga":
                    rover_id = content.get("rover_id", emisor)
                    rpos = tuple(content["pos"])
                    if mem.rescate_actual is None and unidad.carga_recarga > 0:
                        mem.rescate_actual = {"rover_id": rover_id, "pos": rpos}
                        mem.ticks_en_rescate = 0
                        if unidad.bateria > 12.0:
                            self._enviar_acl_directo(
                                api,
                                mem,
                                rover_id,
                                "AGREE",
                                "rescue-ops",
                                conv_id,
                                {"rover_id": rover_id, "rescatista": unidad.id},
                            )
                elif perf == "AGREE" and unidad.tipo == "rover":
                    mem.esperando_agree = False
                elif perf == "FAILURE" and unidad.tipo == "rover":
                    mem.esperando_agree = False
                    mem.ultimo_ping_recarga = -999

    # -------------------------------------------------------------------------
    # Navegación segura y evasión de tormentas
    # -------------------------------------------------------------------------
    def _punto_perimetro_base(self, pos: Tuple[int, int], base_pos: Tuple[int, int]) -> Tuple[int, int]:
        w, h = self.grid_size
        bx, by = base_pos
        tx = int(clamp(pos[0], bx - 2, bx + 2))
        ty = int(clamp(pos[1], by - 2, by + 2))
        return (int(clamp(tx, 0, w - 1)), int(clamp(ty, 0, h - 1)))

    def _riesgo_tormenta_segmento(
        self, origen: Tuple[int, int], destino: Tuple[int, int], mem: _MemoriaRavager, tick: int
    ) -> float:
        if not mem.tormentas:
            return 0.0
        pasos = max(1, distancia_chebyshev(origen, destino))
        muestras = min(4, pasos)
        riesgo = 0.0
        for i in range(1, muestras + 1):
            alpha = i / muestras
            mx = origen[0] + (destino[0] - origen[0]) * alpha
            my = origen[1] + (destino[1] - origen[1]) * alpha
            for t in mem.tormentas:
                if tick - t.tick_obs <= TTL_TORMENTA:
                    tx = t.fx + t.vx * min(3, i)
                    ty = t.fy + t.vy * min(3, i)
                    d = math.hypot(mx - tx, my - ty)
                    if d <= t.radio + 0.7:
                        riesgo += 1.0 + (t.radio + 0.7 - d) * 0.35
        return riesgo

    def _paso_seguro(
        self, origen: Tuple[int, int], destino: Tuple[int, int], mem: _MemoriaRavager, tick: int
    ) -> Tuple[int, int]:
        if origen == destino:
            return (0, 0)

        w, h = self.grid_size
        dx_dir = 0 if destino[0] == origen[0] else (1 if destino[0] > origen[0] else -1)
        dy_dir = 0 if destino[1] == origen[1] else (1 if destino[1] > origen[1] else -1)
        paso_recto = (origen[0] + dx_dir, origen[1] + dy_dir)

        cerca_torm = any(
            (tick - t.tick_obs <= TTL_TORMENTA)
            and (
                math.hypot(origen[0] - t.fx, origen[1] - t.fy) <= t.radio + 3.5
                or math.hypot(paso_recto[0] - t.fx, paso_recto[1] - t.fy) <= t.radio + 2.5
            )
            for t in mem.tormentas
        )
        if not cerca_torm:
            return (dx_dir, dy_dir)

        abierto: List[Tuple[float, float, Tuple[int, int]]] = []
        heapq.heappush(abierto, (float(distancia_chebyshev(origen, destino)), 0.0, origen))
        g_cost: Dict[Tuple[int, int], float] = {origen: 0.0}
        vino_de: Dict[Tuple[int, int], Tuple[int, int]] = {}
        cerrados: Set[Tuple[int, int]] = set()
        mejor_nodo = origen
        mejor_h = float(distancia_chebyshev(origen, destino))

        expansiones = 0
        while abierto and expansiones < MAX_EXPANSIONES_ASTAR:
            _, g_act, actual = heapq.heappop(abierto)
            if actual in cerrados:
                continue
            cerrados.add(actual)
            expansiones += 1

            h_act = float(distancia_chebyshev(actual, destino))
            if h_act < mejor_h and actual != origen:
                mejor_h = h_act
                mejor_nodo = actual
            if actual == destino:
                mejor_nodo = actual
                break

            ax, ay = actual
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = ax + dx, ay + dy
                    if not (0 <= nx < w and 0 <= ny < h):
                        continue
                    nxt = (nx, ny)
                    if nxt in cerrados:
                        continue
                    paso_c = 1.0
                    for t in mem.tormentas:
                        if tick - t.tick_obs <= TTL_TORMENTA:
                            d_t = math.hypot(nx - (t.fx + t.vx), ny - (t.fy + t.vy))
                            if d_t <= t.radio + 0.8:
                                paso_c += 12.0 + (t.radio + 1.0 - d_t) * 2.0
                    ng = g_act + paso_c
                    if ng < g_cost.get(nxt, float("inf")):
                        g_cost[nxt] = ng
                        vino_de[nxt] = actual
                        heapq.heappush(abierto, (ng + float(distancia_chebyshev(nxt, destino)), ng, nxt))

        if mejor_nodo != origen and mejor_nodo in vino_de:
            curr = mejor_nodo
            while vino_de.get(curr) != origen and curr in vino_de:
                curr = vino_de[curr]
            if vino_de.get(curr) == origen:
                return (curr[0] - origen[0], curr[1] - origen[1])

        return (dx_dir, dy_dir)

    def _evadir_tormenta_estatico(
        self, pos: Tuple[int, int], mem: _MemoriaRavager, tick: int, ancla: Optional[Tuple[int, int]] = None
    ) -> Optional[Tuple[int, int]]:
        en_peligro = any(
            math.hypot(pos[0] - t.fx, pos[1] - t.fy) <= t.radio + 0.5
            for t in mem.tormentas
            if tick - t.tick_obs <= TTL_TORMENTA
        )
        if not en_peligro:
            return None

        w, h = self.grid_size
        mejor_paso = None
        mejor_score = -float("inf")
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = pos[0] + dx, pos[1] + dy
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                d_min = min(
                    math.hypot(nx - t.fx, ny - t.fy) - t.radio
                    for t in mem.tormentas
                    if tick - t.tick_obs <= TTL_TORMENTA
                )
                sc = d_min * 4.0 - (distancia_chebyshev((nx, ny), ancla) * 0.35 if ancla else 0.0)
                if sc > mejor_score:
                    mejor_score = sc
                    mejor_paso = (dx, dy)
        return mejor_paso

    # -------------------------------------------------------------------------
    # Mecánicas obligatorias y lógica por rol
    # -------------------------------------------------------------------------
    def _manejar_bateria_baja_r(
        self, unidad: UnidadVista, percepcion: Percepcion, api: Any, mem: _MemoriaRavager
    ) -> Optional[Tuple[Any, ...]]:
        tick = percepcion.tick
        if unidad.tipo == "rover":
            if unidad.estado == "esperando_recarga" or unidad.bateria <= UMBRAL_BATERIA_BAJA:
                if not mem.esperando_agree and tick - mem.ultimo_ping_recarga >= 24:
                    mem.ultimo_ping_recarga = tick
                    mem.esperando_agree = True
                    datos = {
                        "performative": "REQUEST",
                        "sender": unidad.id,
                        "conversation_id": f"rescue_{unidad.id}_{tick}",
                        "ontology": "rescue-ops",
                        "content": {"rover_id": unidad.id, "pos": unidad.pos},
                        "rover_id": unidad.id,
                        "pos": unidad.pos,
                    }
                    if api.enviar_a_recargador_cercano("rescue-ops:REQUEST", datos):
                        self.mensajes_enviados += 1
                if unidad.estado != "esperando_recarga":
                    return ("solicitar_recarga",)
                return ("esperar",)
            return None

        if unidad.bateria <= UMBRAL_BATERIA_BAJA:
            if unidad.tipo == "dron_recarga" and mem.rescate_actual is not None:
                self._enviar_acl_directo(
                    api,
                    mem,
                    mem.rescate_actual["rover_id"],
                    "FAILURE",
                    "rescue-ops",
                    f"fail_{unidad.id}_{tick}",
                    {"motivo": "bateria_baja"},
                )
                mem.rescate_actual = None
            if percepcion.base_aliada_cerca:
                return ("recargar_en_base",)
            dest = self._punto_perimetro_base(unidad.pos, percepcion.base_pos)
            dx, dy = self._paso_seguro(unidad.pos, dest, mem, tick)
            return ("mover", dx, dy)
        return None

    def _reportar_y_evaluar_restos(
        self, unidad: UnidadVista, percepcion: Percepcion, api: Any, mem: _MemoriaRavager
    ) -> Optional[Tuple[Any, ...]]:
        tick = percepcion.tick

        # FIPA-ACL ultra-ligero en drones: solo vetas ricas (>= 3) o restos de rover cercanos
        if unidad.tipo in ("dron_explorador", "dron_recarga"):
            if percepcion.hielo_visible and not percepcion.fase_final:
                ricas = sorted(
                    [h for h in percepcion.hielo_visible if h[2] >= 3],
                    key=lambda h: (-h[2], distancia_chebyshev(unidad.pos, (h[0], h[1]))),
                )
                for hx, hy, cant in ricas:
                    pos = (hx, hy)
                    if self._puede_emitir(mem, ("hielo", pos), tick, cooldown=220):
                        datos = {
                            "performative": "INFORM",
                            "sender": unidad.id,
                            "conversation_id": f"ice_{hx}_{hy}_{tick}",
                            "ontology": "resource-discovery",
                            "content": {"pos": pos, "cantidad": cant},
                            "pos": pos,
                            "cantidad": cant,
                        }
                        if api.enviar_a_rover_ocioso_cercano("resource-discovery:INFORM", datos, pos_ref=pos):
                            self.mensajes_enviados += 1
                        break

            if percepcion.restos_visible and not percepcion.fase_final:
                restos = max(percepcion.restos_visible, key=lambda r: r.get("combustible", 10))
                rid = restos["id"]
                rpos = tuple(restos["pos"])
                comb = int(restos.get("combustible", 10))
                if (
                    comb >= 20
                    and distancia_chebyshev(rpos, percepcion.base_pos) <= 32
                    and self._puede_emitir(mem, ("restos", rid), tick, cooldown=260)
                ):
                    datos = {
                        "performative": "INFORM",
                        "sender": unidad.id,
                        "conversation_id": f"restos_{rid}_{tick}",
                        "ontology": "resource-discovery",
                        "content": {"id": rid, "pos": rpos, "combustible": comb},
                        "id": rid,
                        "pos": rpos,
                    }
                    if api.enviar_a_rover_ocioso_cercano("resource-discovery:INFORM", datos, pos_ref=rpos):
                        self.mensajes_enviados += 1
            return None

        # Rover: consumo/transferencia de restos cercanos sin enviar spam
        if not percepcion.restos_visible or percepcion.fase_final:
            return None

        restos = max(percepcion.restos_visible, key=lambda r: r.get("combustible", 10))
        rid = restos["id"]
        rpos = tuple(restos["pos"])

        aliado_critico = None
        for al in percepcion.aliados_visible:
            if (
                al["id"] != unidad.id
                and (al["estado"] == "esperando_recarga" or al["bateria"] < 60.0)
                and distancia_chebyshev(unidad.pos, tuple(al["pos"])) <= 7
            ):
                aliado_critico = al
                break

        if (unidad.bateria < 78.0 or aliado_critico is not None) and distancia_chebyshev(unidad.pos, rpos) <= 1:
            if unidad.bateria >= 80.0 and aliado_critico is not None:
                return ("consumir_restos", rid, ("transferir", aliado_critico["id"]))
            return ("consumir_restos", rid, "propia")

        return None

    def _generar_waypoints_blitz(self, unidad: UnidadVista) -> List[Tuple[int, int]]:
        """
        65% de los exploradores barren rápidamente el núcleo central disputado (30, 22);
        35% asegura el cuadrante propio con barrido compacto de 3 waypoints.
        """
        w, h = self.grid_size
        idx = self._idx_uid(unidad.id)
        cx, cy = w // 2, h // 2
        bx, by = self.base_pos

        if idx % 20 < 13:  # ~65% Grupo Blitz Central
            ang = ((idx % 10) / 10.0) * math.tau
            rx, ry = 8, 6
            p1 = (int(clamp(cx + math.cos(ang) * rx, 6, w - 7)), int(clamp(cy + math.sin(ang) * ry, 6, h - 7)))
            p2 = (int(clamp(cx - math.cos(ang) * rx, 6, w - 7)), int(clamp(cy - math.sin(ang) * ry, 6, h - 7)))
            p3 = (int(clamp((bx + cx) * 0.5 + (idx % 3 - 1) * 4, 6, w - 7)), int(clamp((by + cy) * 0.5, 6, h - 7)))
            wps = [p3, p1, p2]
        else:  # ~35% Grupo Cuadrante Propio
            x_min = 4 if bx < w // 2 else w // 2
            x_max = w // 2 if bx < w // 2 else w - 5
            y_min = 4 if by < h // 2 else h // 2
            y_max = h // 2 if by < h // 2 else h - 5
            slot = idx % 4
            mx = int(x_min + (x_max - x_min) * (0.28 + 0.45 * (slot % 2)))
            my = int(y_min + (y_max - y_min) * (0.28 + 0.45 * (slot // 2)))
            wps = [
                (mx, my),
                (int(clamp(mx + 8, x_min, x_max)), my),
                (mx, int(clamp(my + 7, y_min, y_max))),
            ]
        wps.sort(key=lambda p: distancia_chebyshev(self.base_pos, p))
        return wps

    def _decidir_explorador(
        self, unidad: UnidadVista, percepcion: Percepcion, mem: _MemoriaRavager
    ) -> Tuple[Any, ...]:
        tick = percepcion.tick
        if percepcion.fase_final:
            ev = self._evadir_tormenta_estatico(unidad.pos, mem, tick)
            if ev is not None:
                return ("mover", ev[0], ev[1])
            return ("esperar",)

        if mem.waypoints is None:
            mem.waypoints = self._generar_waypoints_blitz(unidad)

        if mem.ciclo_patrulla_completo:
            if tick >= 260 or tick % 5 != 0:
                ev = self._evadir_tormenta_estatico(unidad.pos, mem, tick)
                if ev is not None:
                    return ("mover", ev[0], ev[1])
                return ("esperar",)

        wps = mem.waypoints
        dest = wps[mem.patrulla_idx % len(wps)]
        if distancia_chebyshev(unidad.pos, dest) <= 2:
            mem.patrulla_idx = (mem.patrulla_idx + 1) % len(wps)
            if mem.patrulla_idx == 0:
                mem.ciclo_patrulla_completo = True
            dest = wps[mem.patrulla_idx]

        dx, dy = self._paso_seguro(unidad.pos, dest, mem, tick)
        return ("mover", dx, dy)

    def _decidir_recargador(
        self, unidad: UnidadVista, percepcion: Percepcion, api: Any, mem: _MemoriaRavager
    ) -> Tuple[Any, ...]:
        tick = percepcion.tick
        if unidad.carga_recarga <= 0:
            mem.rescate_actual = None
            if percepcion.base_aliada_cerca:
                return ("recargar_en_base",)
            dest = self._punto_perimetro_base(unidad.pos, percepcion.base_pos)
            dx, dy = self._paso_seguro(unidad.pos, dest, mem, tick)
            return ("mover", dx, dy)

        if percepcion.fase_final:
            ev = self._evadir_tormenta_estatico(unidad.pos, mem, tick)
            if ev is not None:
                return ("mover", ev[0], ev[1])
            return ("esperar",)

        rescate = mem.rescate_actual
        if rescate is not None:
            r_id = rescate["rover_id"]
            r_pos = tuple(rescate["pos"])
            mem.ticks_en_rescate += 1
            if distancia_chebyshev(unidad.pos, r_pos) <= 5:
                aliado = next((a for a in percepcion.aliados_visible if a["id"] == r_id), None)
                if aliado is None or aliado["estado"] != "esperando_recarga":
                    mem.rescate_actual = None
                    rescate = None
                else:
                    r_pos = tuple(aliado["pos"])
                    rescate["pos"] = r_pos

            if rescate is not None:
                if distancia_chebyshev(unidad.pos, r_pos) <= 1:
                    mem.rescate_actual = None
                    return ("recargar_aliado", r_id)
                if mem.ticks_en_rescate > 120:
                    self._enviar_acl_directo(
                        api, mem, r_id, "FAILURE", "rescue-ops", f"to_{unidad.id}_{tick}", {"motivo": "timeout"}
                    )
                    mem.rescate_actual = None
                else:
                    dx, dy = self._paso_seguro(unidad.pos, r_pos, mem, tick)
                    return ("mover", dx, dy)

        if not percepcion.base_aliada_cerca:
            dest = self._punto_perimetro_base(unidad.pos, percepcion.base_pos)
            dx, dy = self._paso_seguro(unidad.pos, dest, mem, tick)
            return ("mover", dx, dy)

        if unidad.bateria < 95.0 or unidad.carga_recarga <= 0:
            return ("recargar_en_base",)

        ev = self._evadir_tormenta_estatico(unidad.pos, mem, tick, ancla=percepcion.base_pos)
        if ev is not None:
            return ("mover", ev[0], ev[1])
        return ("esperar",)

    def _depositar_y_contabilizar(
        self, unidad: UnidadVista, mem: _MemoriaRavager
    ) -> Tuple[str]:
        if unidad.carga > 0:
            mem.hielo_depositado += unidad.carga
            self._hielo_en_base += unidad.carga
        if mem.hielo_depositado >= UMBRAL_REPOSO_ROVER or self._hielo_en_base >= UMBRAL_REPOSO_BASE:
            if self._idx_uid(unidad.id) < ELITE_ROVERS_COUNT:
                mem.modo_ultra_safe = True
            else:
                mem.en_reposo = True
        return ("depositar",)

    def _decidir_rover(
        self, unidad: UnidadVista, percepcion: Percepcion, mem: _MemoriaRavager
    ) -> Tuple[Any, ...]:
        tick = percepcion.tick

        if percepcion.fase_final:
            if unidad.carga > 0:
                if percepcion.base_aliada_cerca:
                    return self._depositar_y_contabilizar(unidad, mem)
                dest = self._punto_perimetro_base(unidad.pos, percepcion.base_pos)
                dx, dy = self._paso_seguro(unidad.pos, dest, mem, tick)
                return ("mover", dx, dy)
            ev = self._evadir_tormenta_estatico(unidad.pos, mem, tick)
            if ev is not None:
                return ("mover", ev[0], ev[1])
            return ("esperar",)

        if unidad.carga >= unidad.capacidad:
            mem.objetivo_hielo = None
            if percepcion.base_aliada_cerca:
                return self._depositar_y_contabilizar(unidad, mem)
            dest = self._punto_perimetro_base(unidad.pos, percepcion.base_pos)
            dx, dy = self._paso_seguro(unidad.pos, dest, mem, tick)
            return ("mover", dx, dy)

        # Extracción instantánea gratuita (0% batería)
        for hx, hy, cant in percepcion.hielo_visible:
            if (hx, hy) == unidad.pos and cant > 0 and unidad.carga < unidad.capacidad:
                return ("extraer",)

        # Selección de mejor objetivo por utilidad enfocada en recolección masiva 5/5
        cap_libre = max(0, unidad.capacidad - unidad.carga)
        es_encadenado = unidad.carga > 0
        mejor_u = -float("inf")
        mejor_tipo: Optional[str] = None
        mejor_target: Any = None

        for pos, cr in mem.hielo.items():
            # Anti-duplicación local en vetas pequeñas
            if cr.cantidad <= 2 and pos != mem.objetivo_hielo:
                mi_d = distancia_chebyshev(unidad.pos, pos)
                if any(
                    al["tipo"] == "rover"
                    and al["estado"] == "activo"
                    and distancia_chebyshev(tuple(al["pos"]), pos) + 1 < mi_d
                    for al in percepcion.aliados_visible
                ):
                    continue

            d_ida = distancia_chebyshev(unidad.pos, pos)
            if es_encadenado and d_ida > 12:
                continue
            d_vuelta = max(0, distancia_chebyshev(pos, percepcion.base_pos) - 2)
            r_torm = self._riesgo_tormenta_segmento(unidad.pos, pos, mem, tick)

            if mem.modo_ultra_safe:
                # Escuadrón élite ultra-safe: solo actúa si mejora estrictamente el Ratio y tiene batería holgada
                if unidad.carga == 0 and unidad.bateria < 50.0:
                    continue
                conf = cr.confianza(tick)
                if r_torm > 0.0 or conf < 0.60 or (d_ida + d_vuelta) > 28:
                    continue
                if unidad.bateria - (d_ida + d_vuelta) < UMBRAL_BATERIA_BAJA + 18.0:
                    continue
                extraible_est = min(cr.cantidad, cap_libre) * conf
                d_ret_actual = max(0, distancia_chebyshev(unidad.pos, percepcion.base_pos) - 2)
                costo_marginal = max(
                    1,
                    d_ida + (d_vuelta if unidad.carga == 0 else max(0, d_vuelta - d_ret_actual)),
                )
                if (extraible_est / costo_marginal) < 0.12:
                    continue
            else:
                if unidad.bateria - (d_ida + d_vuelta) < UMBRAL_BATERIA_BAJA + 4.0:
                    continue

            extraible = min(cr.cantidad, cap_libre)
            u_val = (
                4.8 * extraible * cr.confianza(tick) * (1.45 if es_encadenado else 1.0)
                - 0.30 * (d_ida + (0.20 if es_encadenado else 0.35) * d_vuelta)
                - 2.6 * r_torm
            )
            if u_val > mejor_u:
                mejor_u = u_val
                mejor_tipo = "hielo"
                mejor_target = pos

        if unidad.carga == 0 and unidad.bateria < 78.0 and not mem.modo_ultra_safe:
            for rid, cr in mem.restos.items():
                d_ida = distancia_chebyshev(unidad.pos, cr.pos)
                r_torm = self._riesgo_tormenta_segmento(unidad.pos, cr.pos, mem, tick)
                u_val = 2.0 * (cr.combustible / 10.0) - 0.32 * d_ida - 2.6 * r_torm
                if u_val > mejor_u:
                    mejor_u = u_val
                    mejor_tipo = "restos"
                    mejor_target = {"id": rid, "pos": cr.pos}

        if unidad.carga > 0:
            if mejor_tipo == "hielo" and mejor_u > 0.8:
                mem.objetivo_hielo = mejor_target
                dx, dy = self._paso_seguro(unidad.pos, mejor_target, mem, tick)
                return ("mover", dx, dy)
            if percepcion.base_aliada_cerca:
                return self._depositar_y_contabilizar(unidad, mem)
            dest = self._punto_perimetro_base(unidad.pos, percepcion.base_pos)
            dx, dy = self._paso_seguro(unidad.pos, dest, mem, tick)
            return ("mover", dx, dy)

        if mejor_tipo == "restos" and mejor_target is not None and mejor_u > 0.0:
            r_id = mejor_target["id"]
            r_pos = tuple(mejor_target["pos"])
            if distancia_chebyshev(unidad.pos, r_pos) <= 1:
                return ("consumir_restos", r_id, "propia")
            dx, dy = self._paso_seguro(unidad.pos, r_pos, mem, tick)
            return ("mover", dx, dy)

        if mejor_tipo == "hielo" and mejor_target is not None:
            mem.objetivo_hielo = mejor_target
            dx, dy = self._paso_seguro(unidad.pos, mejor_target, mem, tick)
            return ("mover", dx, dy)

        # Despliegue Blitz Central Agresivo (60% centro disputado, 40% cuadrante propio)
        if tick < 200 and not mem.modo_ultra_safe:
            idx = self._idx_uid(unidad.id)
            w, h = percepcion.grid_size
            bx, by = percepcion.base_pos
            if idx % 10 < 6:  # 60% Grupo Blitz al anillo central
                ang = ((idx % 12) / 12.0) * math.tau
                rad = 6 + (idx % 4) * 2
                tx = int(clamp(w * 0.5 + math.cos(ang) * rad, 5, w - 6))
                ty = int(clamp(h * 0.5 + math.sin(ang) * (rad * 0.8), 5, h - 6))
            else:  # 40% Grupo Cuadrante
                dir_x = 1 if bx < w // 2 else -1
                dir_y = 1 if by < h // 2 else -1
                rad = 10 + (idx % 4) * 4
                ang = ((idx % 8) / 7.0) * (math.pi * 0.46)
                tx = int(clamp(bx + dir_x * math.cos(ang) * rad, 3, w - 4))
                ty = int(clamp(by + dir_y * math.sin(ang) * rad, 3, h - 4))
            if distancia_chebyshev(unidad.pos, (tx, ty)) > 2:
                dx, dy = self._paso_seguro(unidad.pos, (tx, ty), mem, tick)
                return ("mover", dx, dy)

        ev = self._evadir_tormenta_estatico(unidad.pos, mem, tick)
        if ev is not None:
            return ("mover", ev[0], ev[1])
        return ("esperar",)

    # -------------------------------------------------------------------------
    # Punto de entrada principal
    # -------------------------------------------------------------------------
    def decidir_unidad(
        self, unidad: UnidadVista, percepcion: Percepcion, api: Any
    ) -> Tuple[Any, ...]:
        t0 = time.perf_counter()
        try:
            if unidad.estado == "muerto":
                return ("esperar",)

            mem = self._mem_privada(unidad)
            self._actualizar_percepcion(unidad, percepcion, mem)
            self._procesar_mensajes(unidad, percepcion, api, mem)

            accion_bat = self._manejar_bateria_baja_r(unidad, percepcion, api, mem)
            if accion_bat is not None:
                return accion_bat

            if unidad.estado == "consumiendo":
                return ("esperar",)

            # Activación de Modo Reposo tras alcanzar cuota (conservando escuadrón élite ultra-safe)
            idx_u = self._idx_uid(unidad.id)
            if percepcion.base_aliada_cerca and self._hielo_en_base >= UMBRAL_REPOSO_BASE:
                if unidad.tipo == "rover" and idx_u < ELITE_ROVERS_COUNT:
                    mem.modo_ultra_safe = True
                elif unidad.tipo == "dron_recarga" and idx_u < ELITE_RECARGADORES_COUNT:
                    mem.en_reposo = False
                else:
                    mem.en_reposo = True
            if unidad.tipo == "rover" and mem.hielo_depositado >= UMBRAL_REPOSO_ROVER and unidad.carga == 0:
                if idx_u < ELITE_ROVERS_COUNT:
                    mem.modo_ultra_safe = True
                else:
                    mem.en_reposo = True
            if unidad.tipo == "dron_explorador" and mem.ciclo_patrulla_completo and percepcion.tick >= 200:
                mem.en_reposo = True

            if mem.en_reposo:
                if unidad.tipo == "rover" and unidad.carga > 0:
                    if percepcion.base_aliada_cerca:
                        return self._depositar_y_contabilizar(unidad, mem)
                    dest = self._punto_perimetro_base(unidad.pos, percepcion.base_pos)
                    dx, dy = self._paso_seguro(unidad.pos, dest, mem, percepcion.tick)
                    return ("mover", dx, dy)
                # Evasión mínima solo si una tormenta entra exactamente encima
                en_torm = any(
                    math.hypot(unidad.pos[0] - t.fx, unidad.pos[1] - t.fy) <= t.radio + 0.3
                    for t in mem.tormentas
                    if percepcion.tick - t.tick_obs <= TTL_TORMENTA
                )
                if en_torm:
                    ev = self._evadir_tormenta_estatico(unidad.pos, mem, percepcion.tick)
                    if ev is not None:
                        return ("mover", ev[0], ev[1])
                return ("esperar",)

            if not mem.modo_ultra_safe:
                accion_restos = self._reportar_y_evaluar_restos(unidad, percepcion, api, mem)
                if accion_restos is not None:
                    return accion_restos

            if unidad.tipo == "dron_explorador":
                return self._decidir_explorador(unidad, percepcion, mem)
            if unidad.tipo == "dron_recarga":
                return self._decidir_recargador(unidad, percepcion, api, mem)
            return self._decidir_rover(unidad, percepcion, mem)
        finally:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            t_cur = percepcion.tick
            self.latencias_turno_ms[t_cur] = self.latencias_turno_ms.get(t_cur, 0.0) + dt_ms

    def obtener_resumen_profiling(self) -> Dict[str, Any]:
        turnos = sorted(self.latencias_turno_ms.values())
        if not turnos:
            return {"mean_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "max_ms": 0.0, "mensajes": self.mensajes_enviados}
        n = len(turnos)
        return {
            "mean_ms": sum(turnos) / n,
            "p95_ms": turnos[min(n - 1, int(0.95 * (n - 1)))],
            "p99_ms": turnos[min(n - 1, int(0.99 * (n - 1)))],
            "max_ms": turnos[-1],
            "mensajes": self.mensajes_enviados,
        }
