"""
equipoD.py
==========
Inteligencia autónoma de CUERVO para la Simulación Multiagente Competitiva
en Marte (60x45), heredando estrictamente de `EquipoBase` (`equipo_base.py`).

Concepto Core — Enjambre Fantasma Rival-Aware + Surfing de Estela de Tormenta:
  1. Percepción Rival-Aware y Exploración en Dos Alas (`MemoriaCuervo`):
     Además de creencias locales de hielo, restos y tormentas con velocidad
     (vx, vy), construye un Mapa de Calor de Presión Rival en macro-celdas 5x5
     con decaimiento exponencial y rastrea rivales en estado crítico
     (`bateria <= 32%` o `esperando_recarga`). Sus exploradores se dividen en
     Ala Cuadrante & Flancos (60%) y Ala Estela / Storm-Shadowers (40%).
  2. Utilidad Anti-Carrera, Transferencia de Carroña y Hook Exclusivo EquipoD:
     Evalúa cada veta con Race Check contra `percepcion.rivales_visible` para
     evitar carreras perdidas; sostiene a rovers cargados transfiriendo carroña
     en campo; y utiliza el hook nativo de `simulador.py` para `EquipoD`
     (cancelar consumo cuando `rivales_visible` revela mejor oportunidad o
     congestión, usando `paso_alejarse` de `equipo_base.py`).
  3. Comunicación Estructurada FIPA-ACL Táctica:
     Performativas FIPA-ACL sobre 4 ontologías con filtros de presión rival
     (evita gastar -1% reportando vetas rodeadas por rivales) y cooldowns.
  4. Coordinación CNP de Doble Mercado (`hielo_sigilo`, `soporte_carrona`, `rescate`):
     Subastas descentralizadas tanto para vetas despejadas como para asignar
     rovers vacíos a consumir carroña (+20 u.) y transferirla a aliados.
  5. Elección Social (Votación por Aprobación + Desempate Copeland):
     Cuatro representantes votan periódicamente entre `FLANQUEO_SIGILOSO`,
     `COSECHA_ESTELA_CARRONA`, `AHORRO_EVASIVO` y `REPLIEGUE_FINAL`.
  6. Arquitectura No Bloqueante y Profiling (< 40 ms/turno):
     Acceso O(1) a macro-celdas, race check local O(|R_vis|), A* acotado e
     instrumentación `time.perf_counter()`.

Regla de Integridad Arquitectónica:
  Cada agente accede únicamente a `self._agentes[unidad.id]`. No se lee ni
  escribe `self.hielo_conocido`, `self.restos_conocidos` ni
  `self.solicitudes_recarga`; toda información remota circula por `CommsAPI`.
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from enum import Enum
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
    paso_alejarse,
)

RADIO_VISION: Dict[str, int] = {
    "rover": 5,
    "dron_explorador": 9,
    "dron_recarga": 7,
}

POLITICAS_ORDEN: Tuple[str, ...] = (
    "FLANQUEO_SIGILOSO",
    "COSECHA_ESTELA_CARRONA",
    "AHORRO_EVASIVO",
    "REPLIEGUE_FINAL",
)

TTL_HIELO = 125
TTL_RESTOS = 70
TTL_TORMENTA = 14
TTL_CONTACTO = 95
TTL_PRESION_RIVAL = 60
TTL_RIVAL_CRITICO = 45
TTL_RESERVA_CNP = 32
MAX_EXPANSIONES_ASTAR = 24
DECAY_HIELO = 0.015
DECAY_PRESION = 0.03


# -----------------------------------------------------------------------------
# Sesión 2: Máquinas de Estados Finitos (FSM) explícitas por rol
# -----------------------------------------------------------------------------
class EstadoRover(str, Enum):
    OCIOSO = "OCIOSO"
    EXPLORANDO = "EXPLORANDO"
    RECOLECTANDO = "RECOLECTANDO"
    RETORNANDO = "RETORNANDO"
    CONSUMIENDO_CARRONA = "CONSUMIENDO_CARRONA"
    ESPERANDO_RECARGA = "ESPERANDO_RECARGA"


class EstadoExplorador(str, Enum):
    PATRULLANDO = "PATRULLANDO"
    RETORNANDO_BASE = "RETORNANDO_BASE"


class EstadoRecargador(str, Enum):
    EN_ESPERA = "EN_ESPERA"
    EN_RESCATE = "EN_RESCATE"
    RETORNANDO_BASE = "RETORNANDO_BASE"


class PoliticaTactica(str, Enum):
    FLANQUEO_SIGILOSO = "FLANQUEO_SIGILOSO"
    COSECHA_ESTELA_CARRONA = "COSECHA_ESTELA_CARRONA"
    AHORRO_EVASIVO = "AHORRO_EVASIVO"
    REPLIEGUE_FINAL = "REPLIEGUE_FINAL"


# -----------------------------------------------------------------------------
# Sesión 1: Estructuras privadas de creencias espaciales y presión rival
# -----------------------------------------------------------------------------
@dataclass
class CreenciaHielo:
    pos: Tuple[int, int]
    cantidad: int
    tick_obs: int
    origen: str

    def confianza(self, tick_actual: int) -> float:
        return math.exp(-DECAY_HIELO * max(0, tick_actual - self.tick_obs))


@dataclass
class CreenciaRestos:
    id: str
    pos: Tuple[int, int]
    combustible: int
    tick_obs: int
    equipo_origen: str


@dataclass
class TrazaTormenta:
    fx: float
    fy: float
    vx: float
    vy: float
    radio: float
    tick_obs: int


@dataclass
class SubastaCuervo:
    conversation_id: str
    tipo: str  # "hielo_sigilo" | "soporte_carrona" | "rescate"
    objetivo_key: Any
    datos: Dict[str, Any]
    tick_inicio: int
    tick_cierre: int
    candidatos_enviados: List[str] = field(default_factory=list)
    propuestas: List[Dict[str, Any]] = field(default_factory=list)
    cerrada: bool = False


@dataclass
class MemoriaCuervo:
    """Memoria espacial y táctica rival-aware privada de una unidad."""

    uid: str
    tipo: str
    estado_fsm: str

    hielo: Dict[Tuple[int, int], CreenciaHielo] = field(default_factory=dict)
    restos: Dict[str, CreenciaRestos] = field(default_factory=dict)
    tormentas: List[TrazaTormenta] = field(default_factory=list)
    celdas_visitadas: Dict[Tuple[int, int], int] = field(default_factory=dict)

    # Inteligencia sobre rivales (macro-celdas 5x5 y unidades críticas)
    presion_rival: Dict[Tuple[int, int], Tuple[float, int]] = field(default_factory=dict)
    rivales_criticos: Dict[str, Tuple[Tuple[int, int], float, int]] = field(default_factory=dict)

    contactos_rovers: Dict[str, Tuple[Tuple[int, int], float, str, int]] = field(
        default_factory=dict
    )
    contactos_recargadores: Dict[str, Tuple[Tuple[int, int], float, str, int]] = field(
        default_factory=dict
    )

    waypoints: Optional[List[Tuple[int, int]]] = None
    patrulla_idx: int = 0
    ciclo_patrulla_completo: bool = False
    objetivo_hielo: Optional[Tuple[int, int]] = None
    objetivo_restos: Optional[Dict[str, Any]] = None
    rescate_actual: Optional[Dict[str, Any]] = None
    ticks_en_rescate: int = 0
    consumo_es_transferencia: bool = False

    cooldowns: Dict[Any, int] = field(default_factory=dict)
    ultimo_ping_recarga: int = -999
    esperando_agree_rescate: bool = False

    subastas_activas: Dict[str, SubastaCuervo] = field(default_factory=dict)
    reservas_ajenas: Dict[Any, int] = field(default_factory=dict)

    politica_actual: PoliticaTactica = PoliticaTactica.FLANQUEO_SIGILOSO
    politica_epoch: int = 0
    ultimo_tick_gov_envio: int = -999
    telemetria_gov_recibida: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    bajas_conocidas: Set[str] = field(default_factory=set)


class EquipoD(EquipoBase):
    """
    Facción CUERVO: enjambre fantasma rival-aware con surfing de estela de
    tormenta, uso del hook exclusivo de EquipoD, doble mercado CNP y votación
    por aprobación con desempate Copeland.
    """

    nombre: str = "CUERVO"

    def __init__(self) -> None:
        super().__init__()
        self._agentes: Dict[str, MemoriaCuervo] = {}
        self._ids_por_rol: Dict[str, List[str]] = {
            "rover": [],
            "dron_explorador": [],
            "dron_recarga": [],
        }

        self.latencias_unidad_ms: List[float] = []
        self.latencias_turno_ms: Dict[int, float] = {}
        self.stats_comms: Dict[str, Any] = {
            "enviados": 0,
            "bloqueados_filtro": 0,
            "bloqueados_presion": 0,
            "por_performativa": {},
            "por_ontologia": {},
        }
        self.stats_cnp: Dict[str, int] = {
            "cfp_hielo": 0,
            "cfp_carrona": 0,
            "cfp_rescate": 0,
            "propose_recibidos": 0,
            "accept_enviados": 0,
            "reject_enviados": 0,
            "duplicados_evitados": 0,
        }
        self.stats_hook_d: Dict[str, int] = {
            "evaluaciones_con_rivales": 0,
            "interrupciones_veta": 0,
            "interrupciones_evasion": 0,
            "consumos_mantenidos": 0,
        }
        self.transiciones_fsm: Dict[str, int] = {}
        self.historial_votaciones: List[Dict[str, Any]] = []

    def inicializar(
        self,
        letra: str,
        base_pos: Tuple[int, int],
        grid_size: Tuple[int, int] = (GRID_W, GRID_H),
    ) -> None:
        super().inicializar(letra, base_pos, grid_size)
        self._agentes.clear()
        self._ids_por_rol = {"rover": [], "dron_explorador": [], "dron_recarga": []}
        self.latencias_unidad_ms.clear()
        self.latencias_turno_ms.clear()
        self.stats_comms = {
            "enviados": 0,
            "bloqueados_filtro": 0,
            "bloqueados_presion": 0,
            "por_performativa": {},
            "por_ontologia": {},
        }
        self.stats_cnp = {
            "cfp_hielo": 0,
            "cfp_carrona": 0,
            "cfp_rescate": 0,
            "propose_recibidos": 0,
            "accept_enviados": 0,
            "reject_enviados": 0,
            "duplicados_evitados": 0,
        }
        self.stats_hook_d = {
            "evaluaciones_con_rivales": 0,
            "interrupciones_veta": 0,
            "interrupciones_evasion": 0,
            "consumos_mantenidos": 0,
        }
        self.transiciones_fsm.clear()
        self.historial_votaciones.clear()

    # -------------------------------------------------------------------------
    # Utilidades de ciclo de vida y presión rival
    # -------------------------------------------------------------------------
    @staticmethod
    def _extraer_indice_uid(uid: str) -> int:
        try:
            return int(uid.split("_")[-1])
        except Exception:
            return abs(hash(uid)) % 100

    @staticmethod
    def _macro_celda(pos: Tuple[int, int]) -> Tuple[int, int]:
        return (pos[0] // 5, pos[1] // 5)

    def _presion_en_pos(self, pos: Tuple[int, int], mem: MemoriaCuervo, tick: int) -> float:
        mc = self._macro_celda(pos)
        entrada = mem.presion_rival.get(mc)
        if entrada is None:
            return 0.0
        val, t_obs = entrada
        dt = max(0, tick - t_obs)
        if dt > TTL_PRESION_RIVAL:
            return 0.0
        return val * math.exp(-DECAY_PRESION * dt)

    def _obtener_memoria(self, unidad: UnidadVista) -> MemoriaCuervo:
        if unidad.id not in self._agentes:
            if unidad.tipo == "rover":
                estado_ini = EstadoRover.OCIOSO.value
            elif unidad.tipo == "dron_explorador":
                estado_ini = EstadoExplorador.PATRULLANDO.value
            else:
                estado_ini = EstadoRecargador.EN_ESPERA.value

            self._agentes[unidad.id] = MemoriaCuervo(
                uid=unidad.id,
                tipo=unidad.tipo,
                estado_fsm=estado_ini,
            )
            lista = self._ids_por_rol.get(unidad.tipo)
            if lista is not None and unidad.id not in lista:
                lista.append(unidad.id)
                lista.sort(key=self._extraer_indice_uid)
        return self._agentes[unidad.id]

    def _transicionar_fsm(self, mem: MemoriaCuervo, nuevo_estado: Enum) -> None:
        val = nuevo_estado.value
        if mem.estado_fsm != val:
            clave = f"{mem.tipo}:{mem.estado_fsm}->{val}"
            self.transiciones_fsm[clave] = self.transiciones_fsm.get(clave, 0) + 1
            mem.estado_fsm = val

    # -------------------------------------------------------------------------
    # Sesión 1: Percepción rival-aware, olvido y exploración en dos alas
    # -------------------------------------------------------------------------
    def _actualizar_percepcion(
        self, unidad: UnidadVista, percepcion: Percepcion, mem: MemoriaCuervo
    ) -> None:
        tick = percepcion.tick
        rv = RADIO_VISION.get(unidad.tipo, 5)
        ux, uy = unidad.pos
        mem.celdas_visitadas[unidad.pos] = tick

        if unidad.estado != "consumiendo":
            mem.consumo_es_transferencia = False

        # 1. Hielo visible e invalidación activa
        hielo_vis: Set[Tuple[int, int]] = set()
        for hx, hy, cant in percepcion.hielo_visible:
            if cant > 0:
                pos = (hx, hy)
                hielo_vis.add(pos)
                mem.hielo[pos] = CreenciaHielo(pos=pos, cantidad=cant, tick_obs=tick, origen="local")

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

        # 2. Restos visibles e invalidación
        restos_vis: Set[str] = set()
        for r in percepcion.restos_visible:
            rid = r["id"]
            rpos = tuple(r["pos"])
            restos_vis.add(rid)
            mem.restos[rid] = CreenciaRestos(
                id=rid,
                pos=rpos,
                combustible=int(r.get("combustible", 10)),
                tick_obs=tick,
                equipo_origen=r.get("equipo", "?"),
            )
            if r.get("equipo") == self.letra:
                mem.bajas_conocidas.add(rid)

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

        # 3. Tormentas con velocidad (vx, vy)
        prev_t = [t for t in mem.tormentas if tick - t.tick_obs <= TTL_TORMENTA]
        nuevas_t: List[TrazaTormenta] = []
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
            nuevas_t.append(TrazaTormenta(fx=tx, fy=ty, vx=vx, vy=vy, radio=rad, tick_obs=tick))

        for i, pt in enumerate(prev_t):
            if i not in usadas and tick - pt.tick_obs <= 6:
                nuevas_t.append(
                    TrazaTormenta(
                        fx=clamp(pt.fx + pt.vx, 0.0, self.grid_size[0] - 1.0),
                        fy=clamp(pt.fy + pt.vy, 0.0, self.grid_size[1] - 1.0),
                        vx=pt.vx,
                        vy=pt.vy,
                        radio=pt.radio,
                        tick_obs=pt.tick_obs,
                    )
                )
        mem.tormentas = nuevas_t

        # 4. Inteligencia rival-aware (Mapa de calor de presión y rivales críticos)
        conteo_mc: Dict[Tuple[int, int], float] = {}
        for riv in percepcion.rivales_visible:
            rpos = tuple(riv["pos"])
            mc = self._macro_celda(rpos)
            peso_tipo = 1.2 if riv.get("tipo") == "rover" else 0.5
            conteo_mc[mc] = conteo_mc.get(mc, 0.0) + peso_tipo
            rbat = float(riv.get("bateria", 100.0))
            if rbat <= 32.0 or riv.get("estado") == "esperando_recarga":
                mem.rivales_criticos[riv["id"]] = (rpos, rbat, tick)

        for mc, val in conteo_mc.items():
            prev = self._presion_en_pos((mc[0] * 5, mc[1] * 5), mem, tick)
            mem.presion_rival[mc] = (min(6.0, prev * 0.5 + val), tick)

        mem.presion_rival = {
            mc: v for mc, v in mem.presion_rival.items() if tick - v[1] <= TTL_PRESION_RIVAL
        }
        mem.rivales_criticos = {
            rid: v for rid, v in mem.rivales_criticos.items() if tick - v[2] <= TTL_RIVAL_CRITICO
        }

        # 5. Contactos aliados
        for al in percepcion.aliados_visible:
            aid = al["id"]
            if aid == unidad.id:
                continue
            info = (tuple(al["pos"]), float(al["bateria"]), str(al["estado"]), tick)
            if al.get("tipo") == "rover":
                mem.contactos_rovers[aid] = info
            elif al.get("tipo") == "dron_recarga":
                mem.contactos_recargadores[aid] = info

        mem.contactos_rovers = {
            k: v for k, v in mem.contactos_rovers.items() if tick - v[3] <= TTL_CONTACTO
        }
        mem.contactos_recargadores = {
            k: v for k, v in mem.contactos_recargadores.items() if tick - v[3] <= TTL_CONTACTO
        }
        mem.reservas_ajenas = {
            k: exp for k, exp in mem.reservas_ajenas.items() if exp >= tick
        }

        if percepcion.fase_final and mem.politica_actual != PoliticaTactica.REPLIEGUE_FINAL:
            mem.politica_actual = PoliticaTactica.REPLIEGUE_FINAL
            mem.politica_epoch = max(mem.politica_epoch, 999)

    def _generar_waypoints_dos_alas(self, unidad: UnidadVista) -> List[Tuple[int, int]]:
        """
        Exploración en Dos Alas adaptada dinámicamente a cualquier esquina:
          - 60% Ala Cuadrante & Flancos (periferia y corredores laterales)
          - 40% Ala Estela (corredores medios y surf detrás de tormentas)
        """
        w, h = self.grid_size
        bx, by = self.base_pos
        dir_x = 1 if bx < w // 2 else -1
        dir_y = 1 if by < h // 2 else -1
        idx = self._extraer_indice_uid(unidad.id)

        if idx % 5 < 3:  # 60% Ala Cuadrante & Flancos
            slot = idx % 6
            # Repartir entre interior de cuadrante y los 2 flancos ortogonales
            if slot < 2:
                x1 = int(clamp(bx + dir_x * (12 + slot * 8), 4, w - 5))
                y1 = int(clamp(by + dir_y * (10 + slot * 6), 4, h - 5))
                x2 = int(clamp(bx + dir_x * 26, 4, w - 5))
                y2 = int(clamp(by + dir_y * 12, 4, h - 5))
            elif slot < 4:
                # Flanco horizontal evitando choque directo en el centro
                x1 = int(clamp(bx + dir_x * 16, 4, w - 5))
                y1 = int(clamp(by + dir_y * 6, 4, h - 5))
                x2 = int(clamp(bx + dir_x * 32, 4, w - 5))
                y2 = int(clamp(by + dir_y * (8 + (slot - 2) * 5), 4, h - 5))
            else:
                # Flanco vertical
                x1 = int(clamp(bx + dir_x * 6, 4, w - 5))
                y1 = int(clamp(by + dir_y * 14, 4, h - 5))
                x2 = int(clamp(bx + dir_x * (9 + (slot - 4) * 5), 4, w - 5))
                y2 = int(clamp(by + dir_y * 26, 4, h - 5))
            wps = [(x1, y1), (x2, y2), (int((x1 + x2) * 0.5), int((y1 + y2) * 0.5))]
        else:  # 40% Ala Estela (corredores centrales-flanco)
            cx, cy = w // 2, h // 2
            off = (idx % 4 - 1.5) * 4.5
            p1 = (int(clamp(bx + dir_x * 18, 5, w - 6)), int(clamp(by + dir_y * 14, 5, h - 6)))
            p2 = (int(clamp(cx - dir_x * 6 + off, 6, w - 7)), int(clamp(cy + off * 0.7, 6, h - 7)))
            p3 = (int(clamp(cx + off * 0.8, 6, w - 7)), int(clamp(cy - dir_y * 5, 6, h - 7)))
            wps = [p1, p2, p3]

        wps.sort(key=lambda p: distancia_chebyshev(self.base_pos, p))
        return wps

    # -------------------------------------------------------------------------
    # Sesión 3: Comunicación Estructurada FIPA-ACL Táctica
    # -------------------------------------------------------------------------
    def _empaquetar_acl(
        self,
        mem: MemoriaCuervo,
        performative: str,
        ontology: str,
        conv_id: str,
        content: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload = dict(content)
        payload["_gov"] = {
            "politica": mem.politica_actual.value,
            "epoch": mem.politica_epoch,
        }
        return {
            "performative": performative,
            "sender": mem.uid,
            "conversation_id": conv_id,
            "ontology": ontology,
            "content": payload,
            **{k: v for k, v in content.items() if k in ("pos", "cantidad", "id", "rover_id")},
        }

    def _registrar_envio(self, performative: str, ontology: str, count: int = 1) -> None:
        self.stats_comms["enviados"] += count
        self.stats_comms["por_performativa"][performative] = (
            self.stats_comms["por_performativa"].get(performative, 0) + count
        )
        self.stats_comms["por_ontologia"][ontology] = (
            self.stats_comms["por_ontologia"].get(ontology, 0) + count
        )

    def _enviar_acl(
        self,
        api: Any,
        mem: MemoriaCuervo,
        dest_id: str,
        performative: str,
        ontology: str,
        conv_id: str,
        content: Dict[str, Any],
        asunto_legado: Optional[str] = None,
    ) -> bool:
        datos = self._empaquetar_acl(mem, performative, ontology, conv_id, content)
        ok = api.enviar_mensaje(dest_id, asunto_legado or f"{ontology}:{performative}", datos)
        if ok:
            self._registrar_envio(performative, ontology, 1)
        return ok

    def _enviar_acl_helper(
        self,
        api: Any,
        mem: MemoriaCuervo,
        helper_name: str,
        performative: str,
        ontology: str,
        conv_id: str,
        content: Dict[str, Any],
        asunto_legado: str,
        pos_ref: Optional[Tuple[int, int]] = None,
    ) -> int:
        datos = self._empaquetar_acl(mem, performative, ontology, conv_id, content)
        enviados = 0
        if helper_name == "recargador":
            enviados = 1 if api.enviar_a_recargador_cercano(asunto_legado, datos) else 0
        elif helper_name == "rover":
            enviados = 1 if api.enviar_a_rover_ocioso_cercano(asunto_legado, datos, pos_ref=pos_ref) else 0
        if enviados > 0:
            self._registrar_envio(performative, ontology, enviados)
        return enviados

    def _pasa_filtro(self, mem: MemoriaCuervo, clave: Any, tick: int, cooldown: int) -> bool:
        ultimo = mem.cooldowns.get(clave, -9999)
        if tick - ultimo < cooldown:
            self.stats_comms["bloqueados_filtro"] += 1
            return False
        mem.cooldowns[clave] = tick
        return True

    def _procesar_mensajes_acl(
        self, unidad: UnidadVista, percepcion: Percepcion, api: Any, mem: MemoriaCuervo
    ) -> None:
        tick = percepcion.tick
        for msg in percepcion.mensajes:
            emisor = msg.get("emisor_id", "")
            datos = msg.get("datos", {})
            perf = datos.get("performative") or "INFORM"
            ont = datos.get("ontology") or "stealth-intel"
            conv_id = datos.get("conversation_id", f"crv_{tick}")
            content = datos.get("content", datos)

            gov = content.get("_gov") if isinstance(content, dict) else None
            if gov and int(gov.get("epoch", 0)) > mem.politica_epoch:
                try:
                    mem.politica_actual = PoliticaTactica(gov["politica"])
                    mem.politica_epoch = int(gov["epoch"])
                except Exception:
                    pass

            if ont == "cuervo-gov" and perf == "INFORM":
                if content.get("tipo") == "telemetria":
                    mem.telemetria_gov_recibida[emisor] = {**content, "tick": tick}
                elif "politica" in content:
                    ep = int(content.get("epoch", 0))
                    if ep > mem.politica_epoch:
                        try:
                            mem.politica_actual = PoliticaTactica(content["politica"])
                            mem.politica_epoch = ep
                        except Exception:
                            pass

            elif ont == "stealth-intel" and perf == "INFORM":
                if "cantidad" in content and "pos" in content:
                    pos = tuple(content["pos"])
                    cant = int(content["cantidad"])
                    mem.hielo[pos] = CreenciaHielo(pos=pos, cantidad=cant, tick_obs=tick, origen="acl")
                    if (
                        unidad.tipo == "rover"
                        and mem.objetivo_hielo is None
                        and unidad.carga < unidad.capacidad
                    ):
                        mem.objetivo_hielo = pos
                elif "id" in content and "pos" in content:
                    rid = content["id"]
                    rpos = tuple(content["pos"])
                    mem.restos[rid] = CreenciaRestos(
                        id=rid,
                        pos=rpos,
                        combustible=int(content.get("combustible", 10)),
                        tick_obs=tick,
                        equipo_origen=content.get("equipo", self.letra),
                    )
                    if content.get("equipo") == self.letra:
                        mem.bajas_conocidas.add(rid)
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
                        self._transicionar_fsm(mem, EstadoRecargador.EN_RESCATE)
                        if unidad.bateria > 12.0:
                            self._enviar_acl(
                                api,
                                mem,
                                rover_id,
                                "AGREE",
                                "rescue-ops",
                                conv_id,
                                {"rover_id": rover_id, "rescatista": unidad.id},
                                asunto_legado="rescate_agree",
                            )
                elif perf == "AGREE" and unidad.tipo == "rover":
                    mem.esperando_agree_rescate = False
                elif perf == "FAILURE" and unidad.tipo == "rover":
                    mem.esperando_agree_rescate = False
                    mem.ultimo_ping_recarga = -999

            elif ont == "cnp-dual":
                self._manejar_mensaje_cnp(
                    unidad, percepcion, api, mem, emisor, perf, conv_id, content
                )

    # -------------------------------------------------------------------------
    # Sesión 4: CNP de Doble Mercado (`hielo_sigilo`, `soporte_carrona`, `rescate`)
    # -------------------------------------------------------------------------
    def _calcular_puja_cnp(
        self, unidad: UnidadVista, percepcion: Percepcion, mem: MemoriaCuervo, cnp_data: Dict[str, Any]
    ) -> float:
        tipo_t = cnp_data.get("tipo_tarea")
        obj_pos = tuple(cnp_data["pos"])
        dist = distancia_chebyshev(unidad.pos, obj_pos)
        riesgo = self._riesgo_tormenta_segmento(unidad.pos, obj_pos, mem, percepcion.tick)
        presion = self._presion_en_pos(obj_pos, mem, percepcion.tick)

        if tipo_t == "rescate" and unidad.tipo == "dron_recarga":
            if unidad.carga_recarga <= 0 or unidad.bateria <= UMBRAL_BATERIA_BAJA + 5.0:
                return -999.0
            ocupado = 32.0 if mem.rescate_actual is not None else 0.0
            return 42.0 - 1.4 * dist + 15.0 * (unidad.bateria / 100.0) - 7.0 * riesgo - ocupado

        if unidad.tipo == "rover":
            if unidad.estado == "esperando_recarga" or unidad.bateria <= UMBRAL_BATERIA_BAJA + 6.0:
                return -999.0
            cap_libre = max(0, unidad.capacidad - unidad.carga)
            if tipo_t == "hielo_sigilo":
                if cap_libre <= 0 or mem.objetivo_hielo is not None:
                    return -999.0
                cant = min(int(cnp_data.get("cantidad", 1)), cap_libre)
                d_base = max(0, distancia_chebyshev(obj_pos, percepcion.base_pos) - 2)
                return (
                    14.5 * cant
                    - 1.25 * dist
                    - 0.35 * d_base
                    + 9.0 * (unidad.bateria / 100.0)
                    - 6.0 * riesgo
                    - 3.5 * presion
                )
            if tipo_t == "soporte_carrona":
                if unidad.carga > 0 or mem.objetivo_restos is not None:
                    return -999.0
                comb = int(cnp_data.get("combustible", 10))
                hay_aliado_critico = any(
                    distancia_chebyshev(obj_pos, info[0]) <= 8 and info[1] < 72.0
                    for info in mem.contactos_rovers.values()
                )
                if unidad.bateria >= 82.0 and not hay_aliado_critico:
                    return -999.0
                bono_soporte = 6.0 if hay_aliado_critico else 0.0
                return 9.0 * (comb / 10.0) + bono_soporte - 1.3 * dist - 5.0 * riesgo - 2.0 * presion

        return -999.0

    def _manejar_mensaje_cnp(
        self,
        unidad: UnidadVista,
        percepcion: Percepcion,
        api: Any,
        mem: MemoriaCuervo,
        emisor: str,
        performative: str,
        conv_id: str,
        content: Dict[str, Any],
    ) -> None:
        tick = percepcion.tick
        if performative == "CFP":
            if unidad.bateria <= 15.0:
                return
            puja = self._calcular_puja_cnp(unidad, percepcion, mem, content)
            if puja > 0.0:
                self._enviar_acl(
                    api,
                    mem,
                    emisor,
                    "PROPOSE",
                    "cnp-dual",
                    conv_id,
                    {
                        "bid": round(puja, 4),
                        "bateria": round(unidad.bateria, 2),
                        "pos": unidad.pos,
                        "tipo_tarea": content.get("tipo_tarea"),
                    },
                    asunto_legado="cnp_propose",
                )
        elif performative == "PROPOSE":
            sub = mem.subastas_activas.get(conv_id)
            if sub is not None and not sub.cerrada:
                self.stats_cnp["propose_recibidos"] += 1
                sub.propuestas.append(
                    {
                        "postor": emisor,
                        "bid": float(content.get("bid", 0.0)),
                        "bateria": float(content.get("bateria", 0.0)),
                    }
                )
        elif performative == "ACCEPT_PROPOSAL":
            tipo_t = content.get("tipo_tarea")
            pos = tuple(content["pos"])
            if tipo_t == "hielo_sigilo" and unidad.tipo == "rover":
                mem.objetivo_hielo = pos
                mem.hielo[pos] = CreenciaHielo(
                    pos=pos,
                    cantidad=int(content.get("cantidad", 3)),
                    tick_obs=tick,
                    origen="cnp",
                )
            elif tipo_t == "soporte_carrona" and unidad.tipo == "rover":
                rid = content["id"]
                mem.objetivo_restos = {"id": rid, "pos": pos}
                mem.restos[rid] = CreenciaRestos(
                    id=rid,
                    pos=pos,
                    combustible=int(content.get("combustible", 20)),
                    tick_obs=tick,
                    equipo_origen=content.get("equipo", self.letra),
                )
            elif tipo_t == "rescate" and unidad.tipo == "dron_recarga":
                mem.rescate_actual = {"rover_id": content["rover_id"], "pos": pos}
                mem.ticks_en_rescate = 0
                self._transicionar_fsm(mem, EstadoRecargador.EN_RESCATE)
                self._enviar_acl(
                    api,
                    mem,
                    content["rover_id"],
                    "AGREE",
                    "rescue-ops",
                    conv_id,
                    {"rover_id": content["rover_id"], "rescatista": unidad.id},
                    asunto_legado="rescate_agree",
                )
        elif performative == "REJECT_PROPOSAL":
            if content.get("tipo_tarea") == "hielo_sigilo":
                pos = tuple(content["pos"])
                mem.reservas_ajenas[("hielo", pos)] = tick + TTL_RESERVA_CNP
                if mem.objetivo_hielo == pos:
                    mem.objetivo_hielo = None
            elif content.get("tipo_tarea") == "soporte_carrona":
                rid = content.get("id")
                if rid:
                    mem.reservas_ajenas[("restos", rid)] = tick + TTL_RESERVA_CNP

    def _iniciar_subasta_cnp(
        self,
        unidad: UnidadVista,
        percepcion: Percepcion,
        api: Any,
        mem: MemoriaCuervo,
        tipo_tarea: str,
        objetivo_key: Any,
        datos_tarea: Dict[str, Any],
    ) -> bool:
        if unidad.bateria <= 18.0 or len(mem.subastas_activas) >= 2:
            return False
        tick = percepcion.tick
        if mem.reservas_ajenas.get(objetivo_key, -1) >= tick:
            self.stats_cnp["duplicados_evitados"] += 1
            return False

        pos_ref = tuple(datos_tarea["pos"])
        candidatos: List[str] = []
        if tipo_tarea in ("hielo_sigilo", "soporte_carrona"):
            elegibles = [
                (distancia_chebyshev(pos_ref, info[0]), uid)
                for uid, info in mem.contactos_rovers.items()
                if info[2] == "activo"
                and info[1] > UMBRAL_BATERIA_BAJA + 10.0
                and distancia_chebyshev(pos_ref, info[0]) <= 24
            ]
            elegibles.sort(key=lambda x: x[0])
            candidatos = [uid for _, uid in elegibles[:2]]
        elif tipo_tarea == "rescate":
            elegibles = [
                (distancia_chebyshev(pos_ref, info[0]), uid)
                for uid, info in mem.contactos_recargadores.items()
                if info[1] > UMBRAL_BATERIA_BAJA + 10.0
            ]
            elegibles.sort(key=lambda x: x[0])
            candidatos = [uid for _, uid in elegibles[:2]]

        conv_id = f"cnp_{tipo_tarea}_{unidad.id}_{tick}"
        ventana = 4 if tipo_tarea != "rescate" else 3

        if candidatos:
            enviados: List[str] = []
            for dest_id in candidatos:
                if self._enviar_acl(
                    api,
                    mem,
                    dest_id,
                    "CFP",
                    "cnp-dual",
                    conv_id,
                    {"tipo_tarea": tipo_tarea, **datos_tarea},
                    asunto_legado="cnp_cfp",
                ):
                    enviados.append(dest_id)
                    if tipo_tarea == "hielo_sigilo":
                        self.stats_cnp["cfp_hielo"] += 1
                    elif tipo_tarea == "soporte_carrona":
                        self.stats_cnp["cfp_carrona"] += 1
                    else:
                        self.stats_cnp["cfp_rescate"] += 1
            if enviados:
                mem.subastas_activas[conv_id] = SubastaCuervo(
                    conversation_id=conv_id,
                    tipo=tipo_tarea,
                    objetivo_key=objetivo_key,
                    datos=datos_tarea,
                    tick_inicio=tick,
                    tick_cierre=tick + ventana,
                    candidatos_enviados=enviados,
                )
                return True

        if tipo_tarea in ("hielo_sigilo", "soporte_carrona"):
            asunto = "hielo" if tipo_tarea == "hielo_sigilo" else "restos"
            n = self._enviar_acl_helper(
                api,
                mem,
                "rover",
                "INFORM",
                "stealth-intel",
                conv_id,
                datos_tarea,
                asunto_legado=asunto,
                pos_ref=pos_ref,
            )
            if n > 0:
                mem.reservas_ajenas[objetivo_key] = tick + TTL_RESERVA_CNP
            return n > 0

        return False

    def _adjudicar_subastas_cnp(
        self, unidad: UnidadVista, percepcion: Percepcion, api: Any, mem: MemoriaCuervo
    ) -> None:
        tick = percepcion.tick
        cerradas: List[str] = []

        for conv_id, sub in mem.subastas_activas.items():
            if sub.cerrada:
                cerradas.append(conv_id)
                continue
            if tick < sub.tick_cierre and len(sub.propuestas) < len(sub.candidatos_enviados):
                continue

            sub.cerrada = True
            cerradas.append(conv_id)

            if not sub.propuestas:
                if sub.tipo in ("hielo_sigilo", "soporte_carrona") and unidad.bateria > 18.0:
                    asunto = "hielo" if sub.tipo == "hielo_sigilo" else "restos"
                    self._enviar_acl_helper(
                        api,
                        mem,
                        "rover",
                        "INFORM",
                        "stealth-intel",
                        conv_id,
                        sub.datos,
                        asunto_legado=asunto,
                        pos_ref=tuple(sub.datos["pos"]),
                    )
                continue

            sub.propuestas.sort(key=lambda p: (-p["bid"], -p["bateria"], p["postor"]))
            ganador = sub.propuestas[0]["postor"]
            self._enviar_acl(
                api,
                mem,
                ganador,
                "ACCEPT_PROPOSAL",
                "cnp-dual",
                conv_id,
                {"tipo_tarea": sub.tipo, **sub.datos},
                asunto_legado="cnp_accept",
            )
            self.stats_cnp["accept_enviados"] += 1
            mem.reservas_ajenas[sub.objetivo_key] = tick + TTL_RESERVA_CNP

            for perdedor in sub.propuestas[1:]:
                self._enviar_acl(
                    api,
                    mem,
                    perdedor["postor"],
                    "REJECT_PROPOSAL",
                    "cnp-dual",
                    conv_id,
                    {"tipo_tarea": sub.tipo, **sub.datos},
                    asunto_legado="cnp_reject",
                )
                self.stats_cnp["reject_enviados"] += 1
                self.stats_cnp["duplicados_evitados"] += 1

        for cid in cerradas:
            mem.subastas_activas.pop(cid, None)

    # -------------------------------------------------------------------------
    # Sesión 5: Elección Social — Votación por Aprobación + Desempate Copeland
    # -------------------------------------------------------------------------
    def _enviar_telemetria_gobernanza(
        self, unidad: UnidadVista, percepcion: Percepcion, api: Any, mem: MemoriaCuervo
    ) -> None:
        tick = percepcion.tick
        if tick < 18:
            return
        estratega_id = f"{self.letra}_rover_0"
        if unidad.id == estratega_id:
            return

        rol_map = {
            f"{self.letra}_expl_0": "VigiaFrontera",
            f"{self.letra}_expl_1": "SombraEstela",
            f"{self.letra}_rover_1": "IntendenteCarrona",
        }
        rol = rol_map.get(unidad.id)
        if rol is None:
            return

        intervalo = 55 if not percepcion.fase_final else 35
        if tick - mem.ultimo_tick_gov_envio < intervalo:
            return
        mem.ultimo_tick_gov_envio = tick

        if rol == "VigiaFrontera":
            presion_media = (
                sum(v[0] for v in mem.presion_rival.values()) / max(1, len(mem.presion_rival))
            )
            datos = {
                "tipo": "telemetria",
                "rol": rol,
                "presion_rival": round(presion_media, 2),
                "vetas_libres": len(mem.hielo),
            }
        elif rol == "SombraEstela":
            datos = {
                "tipo": "telemetria",
                "rol": rol,
                "tormentas_vistas": len(mem.tormentas),
                "rivales_criticos": len(mem.rivales_criticos),
            }
        else:
            datos = {
                "tipo": "telemetria",
                "rol": rol,
                "bateria": round(unidad.bateria, 1),
                "restos_conocidos": len(mem.restos),
                "bajas_conocidas": len(mem.bajas_conocidas),
            }

        self._enviar_acl(
            api,
            mem,
            estratega_id,
            "INFORM",
            "cuervo-gov",
            f"gov_crv_{rol}_{tick}",
            datos,
            asunto_legado="gov_telemetria",
        )

    def _votar_aprobacion_copeland(
        self, unidad: UnidadVista, percepcion: Percepcion, api: Any, mem: MemoriaCuervo
    ) -> None:
        estratega_id = f"{self.letra}_rover_0"
        if unidad.id != estratega_id:
            return

        tick = percepcion.tick
        ultimo = self.historial_votaciones[-1]["tick"] if self.historial_votaciones else -999
        if not percepcion.fase_final and (tick < 28 or tick - ultimo < 45):
            return
        if percepcion.fase_final and mem.politica_actual == PoliticaTactica.REPLIEGUE_FINAL:
            return

        t_vig = mem.telemetria_gov_recibida.get(f"{self.letra}_expl_0", {})
        t_som = mem.telemetria_gov_recibida.get(f"{self.letra}_expl_1", {})
        t_int = mem.telemetria_gov_recibida.get(f"{self.letra}_rover_1", {})

        presion = float(t_vig.get("presion_rival", 0.0))
        vetas = max(len(mem.hielo), int(t_vig.get("vetas_libres", 0)))
        riv_crit = max(len(mem.rivales_criticos), int(t_som.get("rivales_criticos", 0)))
        tormentas = max(len(mem.tormentas), int(t_som.get("tormentas_vistas", 0)))
        restos = max(len(mem.restos), int(t_int.get("restos_conocidos", 0)))
        bajas = max(len(mem.bajas_conocidas), int(t_int.get("bajas_conocidas", 0)))
        bat_prom = (unidad.bateria + float(t_int.get("bateria", unidad.bateria))) * 0.5

        def evaluar_rol(rol: str) -> Tuple[Set[str], List[str]]:
            s: Dict[str, float] = {p: 0.0 for p in POLITICAS_ORDEN}
            if percepcion.fase_final:
                s["REPLIEGUE_FINAL"] += 100.0

            if rol == "VigiaFrontera":
                s["FLANQUEO_SIGILOSO"] += 3.5 + (1.2 if vetas >= 3 and presion < 1.5 else 0.0)
                s["COSECHA_ESTELA_CARRONA"] += 1.4 + riv_crit * 0.7 + restos * 0.5
                s["AHORRO_EVASIVO"] += 2.2 + presion * 1.1 + (1.2 if tick >= 260 else 0.0)
                s["REPLIEGUE_FINAL"] += 2.4 if (tick > 460 and vetas == 0) else 0.2
            elif rol == "SombraEstela":
                s["FLANQUEO_SIGILOSO"] += 2.8 if tick < 220 else 1.5
                s["COSECHA_ESTELA_CARRONA"] += 2.2 + tormentas * 0.8 + riv_crit * 0.9 + restos * 0.6
                s["AHORRO_EVASIVO"] += 2.4 + (1.2 if tormentas >= 2 else 0.0)
                s["REPLIEGUE_FINAL"] += 1.5 if tormentas >= 4 else 0.3
            elif rol == "IntendenteCarrona":
                s["FLANQUEO_SIGILOSO"] += 2.4 if bat_prom > 72.0 else 0.9
                s["COSECHA_ESTELA_CARRONA"] += 1.8 + restos * 0.9 + bajas * 1.4
                s["AHORRO_EVASIVO"] += 3.0 + (2.0 if bat_prom < 68.0 else 0.4)
                s["REPLIEGUE_FINAL"] += 2.6 if (bat_prom < 50.0 or tick > 520) else 0.4
            else:  # EstrategaCuervo
                s["FLANQUEO_SIGILOSO"] += 3.2 if tick < 200 else 1.4
                s["COSECHA_ESTELA_CARRONA"] += 1.9 + (restos + riv_crit) * 0.5
                s["AHORRO_EVASIVO"] += 3.0 if tick >= 200 else 1.8
                s["REPLIEGUE_FINAL"] += 2.5 if (tick > 480 and vetas == 0) else 0.3

            ranking = sorted(POLITICAS_ORDEN, key=lambda p: (-s[p], POLITICAS_ORDEN.index(p)))
            umbral = max(s.values()) - 0.85
            aprobadas = {p for p, val in s.items() if val >= umbral}
            return aprobadas, ranking

        roles = ("VigiaFrontera", "SombraEstela", "IntendenteCarrona", "EstrategaCuervo")
        aprob_conteo: Dict[str, int] = {p: 0 for p in POLITICAS_ORDEN}
        rankings: Dict[str, List[str]] = {}
        for r in roles:
            aprob, rnk = evaluar_rol(r)
            rankings[r] = rnk
            for p in aprob:
                aprob_conteo[p] += 1

        # Desempate Copeland sobre rankings ordinales
        duelos: Dict[str, Dict[str, int]] = {
            p1: {p2: 0 for p2 in POLITICAS_ORDEN} for p1 in POLITICAS_ORDEN
        }
        for rnk in rankings.values():
            pos_r = {p: i for i, p in enumerate(rnk)}
            for p1 in POLITICAS_ORDEN:
                for p2 in POLITICAS_ORDEN:
                    if p1 != p2 and pos_r[p1] < pos_r[p2]:
                        duelos[p1][p2] += 1

        copeland: Dict[str, float] = {p: 0.0 for p in POLITICAS_ORDEN}
        for p1 in POLITICAS_ORDEN:
            for p2 in POLITICAS_ORDEN:
                if p1 == p2:
                    continue
                if duelos[p1][p2] > duelos[p2][p1]:
                    copeland[p1] += 1.0
                elif duelos[p1][p2] == duelos[p2][p1]:
                    copeland[p1] += 0.5

        max_aprob = max(aprob_conteo.values())
        empatadas = [p for p, c in aprob_conteo.items() if c == max_aprob]
        if len(empatadas) == 1:
            ganadora = empatadas[0]
            metodo = "Votación por Aprobación"
        else:
            ganadora = max(
                empatadas,
                key=lambda p: (copeland[p], -POLITICAS_ORDEN.index(p)),
            )
            metodo = "Aprobación + Desempate Copeland"

        nueva_pol = PoliticaTactica(ganadora)
        cambio = nueva_pol != mem.politica_actual
        if cambio or not self.historial_votaciones:
            mem.politica_actual = nueva_pol
            mem.politica_epoch += 1

        self.historial_votaciones.append(
            {
                "tick": tick,
                "epoch": mem.politica_epoch,
                "ganadora": ganadora,
                "metodo": metodo,
                "aprobaciones": aprob_conteo,
                "copeland": copeland,
            }
        )

        if cambio:
            for d in (
                f"{self.letra}_expl_0",
                f"{self.letra}_expl_1",
                f"{self.letra}_rover_1",
                f"{self.letra}_rec_0",
            ):
                if d != unidad.id:
                    self._enviar_acl(
                        api,
                        mem,
                        d,
                        "INFORM",
                        "cuervo-gov",
                        f"gov_pol_{mem.politica_epoch}",
                        {"politica": ganadora, "epoch": mem.politica_epoch},
                        asunto_legado="gov_politica",
                    )

    # -------------------------------------------------------------------------
    # Sesión 2: Utilidad Anti-Carrera, Hook Exclusivo y Navegación Segura
    # -------------------------------------------------------------------------
    def _pesos_dinamicos(self, mem: MemoriaCuervo) -> Tuple[float, float, float, float, float]:
        """Retorna (w1_hielo, w2_dist, w3_carrona, w4_torm, gamma_presion)."""
        pol = mem.politica_actual
        if pol == PoliticaTactica.FLANQUEO_SIGILOSO:
            return (4.3, 0.30, 1.8, 2.7, 0.90)
        if pol == PoliticaTactica.COSECHA_ESTELA_CARRONA:
            return (3.8, 0.32, 3.2, 2.9, 0.85)
        if pol == PoliticaTactica.AHORRO_EVASIVO:
            return (4.0, 0.38, 2.0, 3.5, 1.35)
        return (2.2, 0.55, 1.0, 4.5, 1.40)

    def _prob_captura_sin_disputa(
        self, pos: Tuple[int, int], cant: int, mi_pos: Tuple[int, int], percepcion: Percepcion
    ) -> float:
        """Race Check local O(|R_vis|): descarta vetas donde un rival activo llega antes."""
        d_yo = distancia_chebyshev(mi_pos, pos)
        p_cap = 1.0
        for riv in percepcion.rivales_visible:
            if riv.get("tipo") != "rover" or riv.get("estado") != "activo":
                continue
            d_riv = distancia_chebyshev(tuple(riv["pos"]), pos)
            if d_riv + 1 < d_yo and cant <= 3:
                return 0.0
            if d_riv + 2 < d_yo:
                return 0.0
            if d_riv <= d_yo:
                p_cap = min(p_cap, 0.55)
            elif d_riv <= d_yo + 2:
                p_cap = min(p_cap, 0.78)
        return p_cap

    def _riesgo_tormenta_celda(
        self, pos: Tuple[int, int], mem: MemoriaCuervo, tick: int, pasos_fut: int = 1
    ) -> float:
        riesgo = 0.0
        px, py = pos
        for t in mem.tormentas:
            if tick - t.tick_obs > TTL_TORMENTA:
                continue
            tx = t.fx + t.vx * min(3, pasos_fut)
            ty = t.fy + t.vy * min(3, pasos_fut)
            d = math.hypot(px - tx, py - ty)
            if d <= t.radio + 0.7:
                riesgo += 1.0 + (t.radio + 0.7 - d) * 0.35
        return riesgo

    def _riesgo_tormenta_segmento(
        self, origen: Tuple[int, int], destino: Tuple[int, int], mem: MemoriaCuervo, tick: int
    ) -> float:
        if not mem.tormentas:
            return 0.0
        pasos = max(1, distancia_chebyshev(origen, destino))
        muestras = min(4, pasos)
        riesgo = 0.0
        for i in range(1, muestras + 1):
            alpha = i / muestras
            mx = int(round(origen[0] + (destino[0] - origen[0]) * alpha))
            my = int(round(origen[1] + (destino[1] - origen[1]) * alpha))
            riesgo += self._riesgo_tormenta_celda((mx, my), mem, tick, pasos_fut=i)
        return riesgo

    def _punto_perimetro_base(self, pos: Tuple[int, int], base_pos: Tuple[int, int]) -> Tuple[int, int]:
        w, h = self.grid_size
        bx, by = base_pos
        tx = int(clamp(pos[0], bx - 2, bx + 2))
        ty = int(clamp(pos[1], by - 2, by + 2))
        return (int(clamp(tx, 0, w - 1)), int(clamp(ty, 0, h - 1)))

    def _paso_seguro(
        self, origen: Tuple[int, int], destino: Tuple[int, int], mem: MemoriaCuervo, tick: int
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
        self, pos: Tuple[int, int], mem: MemoriaCuervo, tick: int, ancla: Optional[Tuple[int, int]] = None
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

    def _evaluar_hook_consumiendo(
        self, unidad: UnidadVista, percepcion: Percepcion, mem: MemoriaCuervo
    ) -> Tuple[Any, ...]:
        """
        Hook exclusivo de `EquipoD` en `simulador.py`: cuando `estado == 'consumiendo'`
        y existen `rivales_visible`, decide entre completar el consumo o interrumpirlo
        con un movimiento no nulo hacia veta ganable / evasión con `paso_alejarse`.
        """
        self._transicionar_fsm(mem, EstadoRover.CONSUMIENDO_CARRONA)
        if not percepcion.rivales_visible:
            return ("esperar",)

        self.stats_hook_d["evaluaciones_con_rivales"] += 1
        if unidad.bateria < 50.0 or mem.consumo_es_transferencia:
            self.stats_hook_d["consumos_mantenidos"] += 1
            return ("esperar",)

        if unidad.bateria >= 55.0:
            # 1. ¿Hay una veta rica cercana (>=3 u., dist <= 3) que podemos ganar antes que los rivales?
            if unidad.carga < unidad.capacidad and percepcion.hielo_visible:
                for hx, hy, cant in percepcion.hielo_visible:
                    if cant >= 3 and distancia_chebyshev(unidad.pos, (hx, hy)) <= 3:
                        p_cap = self._prob_captura_sin_disputa((hx, hy), cant, unidad.pos, percepcion)
                        if p_cap >= 0.75:
                            dx, dy = self._paso_seguro(unidad.pos, (hx, hy), mem, percepcion.tick)
                            if (dx, dy) != (0, 0):
                                self.stats_hook_d["interrupciones_veta"] += 1
                                return ("mover", dx, dy)

            # 2. Congestión rival muy cercana con carga en juego o riesgo de tormenta
            riv_cerca = [
                tuple(r["pos"])
                for r in percepcion.rivales_visible
                if r.get("tipo") == "rover"
                and r.get("estado") == "activo"
                and distancia_chebyshev(unidad.pos, tuple(r["pos"])) <= 3
            ]
            if (len(riv_cerca) >= 2 and unidad.carga >= 2) or (unidad.en_tormenta and riv_cerca):
                amenazas = [tuple(r["pos"]) for r in percepcion.rivales_visible]
                dx, dy = paso_alejarse(
                    unidad.pos,
                    amenazas,
                    percepcion.tormentas_visible,
                    percepcion.grid_size,
                    preferencia_destino=percepcion.base_pos,
                )
                if (dx, dy) != (0, 0):
                    self.stats_hook_d["interrupciones_evasion"] += 1
                    return ("mover", dx, dy)

        self.stats_hook_d["consumos_mantenidos"] += 1
        return ("esperar",)

    # -------------------------------------------------------------------------
    # Mecánicas obligatorias y lógica diferenciada por rol
    # -------------------------------------------------------------------------
    def _manejar_bateria_baja_d(
        self, unidad: UnidadVista, percepcion: Percepcion, api: Any, mem: MemoriaCuervo
    ) -> Optional[Tuple[Any, ...]]:
        tick = percepcion.tick

        if unidad.tipo == "rover":
            if unidad.estado == "esperando_recarga" or unidad.bateria <= UMBRAL_BATERIA_BAJA:
                self._transicionar_fsm(mem, EstadoRover.ESPERANDO_RECARGA)
                if not mem.esperando_agree_rescate and tick - mem.ultimo_ping_recarga >= 28:
                    mem.ultimo_ping_recarga = tick
                    mem.esperando_agree_rescate = True
                    lanzado = False
                    if unidad.carga >= 2 and len(mem.contactos_recargadores) >= 2:
                        lanzado = self._iniciar_subasta_cnp(
                            unidad,
                            percepcion,
                            api,
                            mem,
                            "rescate",
                            ("rescate", unidad.id),
                            {"rover_id": unidad.id, "pos": unidad.pos},
                        )
                    if not lanzado:
                        self._enviar_acl_helper(
                            api,
                            mem,
                            "recargador",
                            "REQUEST",
                            "rescue-ops",
                            f"resc_{unidad.id}_{tick}",
                            {"rover_id": unidad.id, "pos": unidad.pos},
                            asunto_legado="solicitud_recarga",
                        )
                if unidad.estado != "esperando_recarga":
                    return ("solicitar_recarga",)
                return ("esperar",)
            return None

        if unidad.bateria <= UMBRAL_BATERIA_BAJA:
            if unidad.tipo == "dron_explorador":
                self._transicionar_fsm(mem, EstadoExplorador.RETORNANDO_BASE)
            else:
                self._transicionar_fsm(mem, EstadoRecargador.RETORNANDO_BASE)
                if mem.rescate_actual is not None:
                    self._enviar_acl(
                        api,
                        mem,
                        mem.rescate_actual["rover_id"],
                        "FAILURE",
                        "rescue-ops",
                        f"fail_{unidad.id}_{tick}",
                        {"motivo": "bateria_baja"},
                        asunto_legado="rescate_failure",
                    )
                    mem.rescate_actual = None

            if percepcion.base_aliada_cerca:
                return ("recargar_en_base",)
            dest = self._punto_perimetro_base(unidad.pos, percepcion.base_pos)
            dx, dy = self._paso_seguro(unidad.pos, dest, mem, tick)
            return ("mover", dx, dy)
        return None

    def _evaluar_restos_y_reportar(
        self, unidad: UnidadVista, percepcion: Percepcion, api: Any, mem: MemoriaCuervo
    ) -> Optional[Tuple[Any, ...]]:
        tick = percepcion.tick

        if unidad.tipo in ("dron_explorador", "dron_recarga"):
            if percepcion.hielo_visible and not percepcion.fase_final:
                umbral = 4 if mem.politica_actual == PoliticaTactica.AHORRO_EVASIVO else 3
                candidatos = sorted(
                    [h for h in percepcion.hielo_visible if h[2] >= umbral],
                    key=lambda h: (-h[2], distancia_chebyshev(unidad.pos, (h[0], h[1]))),
                )
                for hx, hy, cant in candidatos:
                    pos = (hx, hy)
                    # Filtro de presión rival: no gastar -1% si la veta está saturada de rivales
                    riv_cerca = sum(
                        1
                        for r in percepcion.rivales_visible
                        if r.get("tipo") == "rover"
                        and r.get("estado") == "activo"
                        and distancia_chebyshev(tuple(r["pos"]), pos) <= 5
                    )
                    if riv_cerca >= 2:
                        self.stats_comms["bloqueados_presion"] += 1
                        continue
                    if self._pasa_filtro(mem, ("hielo", pos), tick, cooldown=360):
                        if cant >= 4 and distancia_chebyshev(pos, percepcion.base_pos) <= 32:
                            self._iniciar_subasta_cnp(
                                unidad,
                                percepcion,
                                api,
                                mem,
                                "hielo_sigilo",
                                ("hielo", pos),
                                {"pos": pos, "cantidad": cant},
                            )
                        elif cant >= 3:
                            self._enviar_acl_helper(
                                api,
                                mem,
                                "rover",
                                "INFORM",
                                "stealth-intel",
                                f"ice_{hx}_{hy}_{tick}",
                                {"pos": pos, "cantidad": cant},
                                asunto_legado="hielo",
                                pos_ref=pos,
                            )
                        break

            if percepcion.restos_visible and not percepcion.fase_final:
                restos = max(percepcion.restos_visible, key=lambda r: r.get("combustible", 10))
                rid = restos["id"]
                rpos = tuple(restos["pos"])
                comb = int(restos.get("combustible", 10))
                if (
                    comb >= 20
                    and distancia_chebyshev(rpos, percepcion.base_pos) <= 32
                    and self._pasa_filtro(mem, ("restos", rid), tick, cooldown=340)
                ):
                    self._iniciar_subasta_cnp(
                        unidad,
                        percepcion,
                        api,
                        mem,
                        "soporte_carrona",
                        ("restos", rid),
                        {
                            "id": rid,
                            "pos": rpos,
                            "combustible": comb,
                            "equipo": restos.get("equipo", self.letra),
                        },
                    )
            return None

        # Rover: consumo y transferencia táctica en campo
        if not percepcion.restos_visible or percepcion.fase_final:
            return None

        restos = max(percepcion.restos_visible, key=lambda r: r.get("combustible", 10))
        rid = restos["id"]
        rpos = tuple(restos["pos"])

        aliados_necesitados = [
            al
            for al in percepcion.aliados_visible
            if al["id"] != unidad.id
            and (al["estado"] == "esperando_recarga" or al["bateria"] < 68.0)
            and distancia_chebyshev(unidad.pos, tuple(al["pos"])) <= 7
        ]
        aliados_necesitados.sort(
            key=lambda al: (0 if al["estado"] == "esperando_recarga" else 1, al["bateria"])
        )
        aliado_critico = aliados_necesitados[0] if aliados_necesitados else None

        quiere_consumir = (
            unidad.bateria < 78.0
            or aliado_critico is not None
            or (unidad.carga == 0 and mem.politica_actual == PoliticaTactica.COSECHA_ESTELA_CARRONA)
        )
        if quiere_consumir and distancia_chebyshev(unidad.pos, rpos) <= 1:
            self._transicionar_fsm(mem, EstadoRover.CONSUMIENDO_CARRONA)
            if aliado_critico is not None and (
                unidad.bateria >= 65.0 or aliado_critico["estado"] == "esperando_recarga"
            ):
                mem.consumo_es_transferencia = True
                return ("consumir_restos", rid, ("transferir", aliado_critico["id"]))
            mem.consumo_es_transferencia = False
            return ("consumir_restos", rid, "propia")

        return None

    def _decidir_explorador(
        self, unidad: UnidadVista, percepcion: Percepcion, mem: MemoriaCuervo
    ) -> Tuple[Any, ...]:
        tick = percepcion.tick
        if percepcion.fase_final or mem.politica_actual == PoliticaTactica.REPLIEGUE_FINAL:
            ev = self._evadir_tormenta_estatico(unidad.pos, mem, tick)
            if ev is not None:
                return ("mover", ev[0], ev[1])
            self._transicionar_fsm(mem, EstadoExplorador.RETORNANDO_BASE)
            return ("esperar",)

        if mem.waypoints is None:
            mem.waypoints = self._generar_waypoints_dos_alas(unidad)

        idx = self._extraer_indice_uid(unidad.id)
        es_ala_estela = (idx % 5) >= 3

        # Tras completar el barrido, modular cadencia / entrar en espera segura
        if mem.ciclo_patrulla_completo:
            limite_tick = (
                170
                if (es_ala_estela and mem.politica_actual == PoliticaTactica.COSECHA_ESTELA_CARRONA and unidad.bateria > 55.0)
                else 130
            )
            if tick >= limite_tick or tick % 6 != 0:
                ev = self._evadir_tormenta_estatico(unidad.pos, mem, tick)
                if ev is not None:
                    return ("mover", ev[0], ev[1])
                return ("esperar",)

        self._transicionar_fsm(mem, EstadoExplorador.PATRULLANDO)
        wps = mem.waypoints
        dest = wps[mem.patrulla_idx % len(wps)]

        # Ala Estela: si hay una tormenta moviéndose cerca, surfear detrás de su estela segura
        if es_ala_estela and mem.tormentas:
            for t in mem.tormentas:
                if tick - t.tick_obs <= 4:
                    spd = math.hypot(t.vx, t.vy)
                    if spd > 0.15:
                        w, h = percepcion.grid_size
                        wx = int(clamp(t.fx - (t.vx / spd) * (t.radio + 2.4), 4, w - 5))
                        wy = int(clamp(t.fy - (t.vy / spd) * (t.radio + 2.4), 4, h - 5))
                        if distancia_chebyshev(unidad.pos, (wx, wy)) <= 14:
                            dest = (wx, wy)
                            break

        if distancia_chebyshev(unidad.pos, dest) <= 2:
            mem.patrulla_idx = (mem.patrulla_idx + 1) % len(wps)
            if mem.patrulla_idx == 0:
                mem.ciclo_patrulla_completo = True
            dest = wps[mem.patrulla_idx]

        dx, dy = self._paso_seguro(unidad.pos, dest, mem, tick)
        return ("mover", dx, dy)

    def _decidir_recargador(
        self, unidad: UnidadVista, percepcion: Percepcion, api: Any, mem: MemoriaCuervo
    ) -> Tuple[Any, ...]:
        tick = percepcion.tick
        if unidad.carga_recarga <= 0:
            mem.rescate_actual = None
            self._transicionar_fsm(mem, EstadoRecargador.RETORNANDO_BASE)
            if percepcion.base_aliada_cerca:
                return ("recargar_en_base",)
            dest = self._punto_perimetro_base(unidad.pos, percepcion.base_pos)
            dx, dy = self._paso_seguro(unidad.pos, dest, mem, tick)
            return ("mover", dx, dy)

        if percepcion.fase_final:
            ev = self._evadir_tormenta_estatico(unidad.pos, mem, tick)
            if ev is not None:
                return ("mover", ev[0], ev[1])
            self._transicionar_fsm(mem, EstadoRecargador.EN_ESPERA)
            return ("esperar",)

        rescate = mem.rescate_actual
        if rescate is not None:
            self._transicionar_fsm(mem, EstadoRecargador.EN_RESCATE)
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
                    self._enviar_acl(
                        api,
                        mem,
                        r_id,
                        "FAILURE",
                        "rescue-ops",
                        f"to_{unidad.id}_{tick}",
                        {"motivo": "timeout"},
                        asunto_legado="rescate_failure",
                    )
                    mem.rescate_actual = None
                else:
                    dx, dy = self._paso_seguro(unidad.pos, r_pos, mem, tick)
                    return ("mover", dx, dy)

        self._transicionar_fsm(mem, EstadoRecargador.EN_ESPERA)
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

    def _decidir_rover(
        self, unidad: UnidadVista, percepcion: Percepcion, mem: MemoriaCuervo
    ) -> Tuple[Any, ...]:
        tick = percepcion.tick

        if percepcion.fase_final:
            if unidad.carga > 0:
                self._transicionar_fsm(mem, EstadoRover.RETORNANDO)
                if percepcion.base_aliada_cerca:
                    return ("depositar",)
                dest = self._punto_perimetro_base(unidad.pos, percepcion.base_pos)
                dx, dy = self._paso_seguro(unidad.pos, dest, mem, tick)
                return ("mover", dx, dy)
            ev = self._evadir_tormenta_estatico(unidad.pos, mem, tick)
            if ev is not None:
                return ("mover", ev[0], ev[1])
            self._transicionar_fsm(mem, EstadoRover.OCIOSO)
            return ("esperar",)

        if unidad.carga >= unidad.capacidad:
            self._transicionar_fsm(mem, EstadoRover.RETORNANDO)
            mem.objetivo_hielo = None
            if percepcion.base_aliada_cerca:
                return ("depositar",)
            dest = self._punto_perimetro_base(unidad.pos, percepcion.base_pos)
            dx, dy = self._paso_seguro(unidad.pos, dest, mem, tick)
            return ("mover", dx, dy)

        for hx, hy, cant in percepcion.hielo_visible:
            if (hx, hy) == unidad.pos and cant > 0 and unidad.carga < unidad.capacidad:
                self._transicionar_fsm(mem, EstadoRover.RECOLECTANDO)
                return ("extraer",)

        w1, w2, w3, w4, gamma = self._pesos_dinamicos(mem)
        cap_libre = max(0, unidad.capacidad - unidad.carga)
        es_encadenado = unidad.carga > 0

        mejor_u = -float("inf")
        mejor_tipo: Optional[str] = None
        mejor_target: Any = None

        for pos, cr in mem.hielo.items():
            if ("hielo", pos) in mem.reservas_ajenas and pos != mem.objetivo_hielo:
                continue
            if cr.cantidad <= 2 and pos != mem.objetivo_hielo:
                mi_d = distancia_chebyshev(unidad.pos, pos)
                if any(
                    al["tipo"] == "rover"
                    and al["estado"] == "activo"
                    and distancia_chebyshev(tuple(al["pos"]), pos) + 1 < mi_d
                    for al in percepcion.aliados_visible
                ):
                    continue

            p_cap = self._prob_captura_sin_disputa(pos, cr.cantidad, unidad.pos, percepcion)
            if p_cap <= 0.0:
                continue

            d_ida = distancia_chebyshev(unidad.pos, pos)
            if es_encadenado and d_ida > 12:
                continue
            d_vuelta = max(0, distancia_chebyshev(pos, percepcion.base_pos) - 2)
            if unidad.bateria - (d_ida + d_vuelta) < UMBRAL_BATERIA_BAJA + 4.0:
                continue

            extraible = min(cr.cantidad, cap_libre)
            r_torm = self._riesgo_tormenta_segmento(unidad.pos, pos, mem, tick)
            presion = self._presion_en_pos(pos, mem, tick)

            u_val = (
                w1 * extraible * cr.confianza(tick) * p_cap * (1.45 if es_encadenado else 1.0)
                - w2 * (d_ida + (0.20 if es_encadenado else 0.35) * d_vuelta)
                - w4 * r_torm
                - gamma * presion
            )
            if u_val > mejor_u:
                mejor_u = u_val
                mejor_tipo = "hielo"
                mejor_target = pos

        if unidad.carga == 0 and unidad.bateria < 82.0:
            for rid, cr in mem.restos.items():
                if ("restos", rid) in mem.reservas_ajenas:
                    continue
                d_ida = distancia_chebyshev(unidad.pos, cr.pos)
                r_torm = self._riesgo_tormenta_segmento(unidad.pos, cr.pos, mem, tick)
                u_val = w3 * (cr.combustible / 10.0) - w2 * d_ida - w4 * r_torm
                if u_val > mejor_u:
                    mejor_u = u_val
                    mejor_tipo = "restos"
                    mejor_target = {"id": rid, "pos": cr.pos}

        if unidad.carga > 0:
            if mejor_tipo == "hielo" and mejor_u > 0.9:
                self._transicionar_fsm(mem, EstadoRover.RECOLECTANDO)
                mem.objetivo_hielo = mejor_target
                dx, dy = self._paso_seguro(unidad.pos, mejor_target, mem, tick)
                return ("mover", dx, dy)
            self._transicionar_fsm(mem, EstadoRover.RETORNANDO)
            if percepcion.base_aliada_cerca:
                return ("depositar",)
            dest = self._punto_perimetro_base(unidad.pos, percepcion.base_pos)
            dx, dy = self._paso_seguro(unidad.pos, dest, mem, tick)
            return ("mover", dx, dy)

        if mejor_tipo == "restos" and mejor_target is not None and mejor_u > 0.0:
            self._transicionar_fsm(mem, EstadoRover.CONSUMIENDO_CARRONA)
            r_id = mejor_target["id"]
            r_pos = tuple(mejor_target["pos"])
            if distancia_chebyshev(unidad.pos, r_pos) <= 1:
                aliado_crit = next(
                    (
                        al
                        for al in percepcion.aliados_visible
                        if al["id"] != unidad.id
                        and (al["estado"] == "esperando_recarga" or al["bateria"] < 68.0)
                        and distancia_chebyshev(unidad.pos, tuple(al["pos"])) <= 7
                    ),
                    None,
                )
                if aliado_crit is not None and (
                    unidad.bateria >= 65.0 or aliado_crit["estado"] == "esperando_recarga"
                ):
                    mem.consumo_es_transferencia = True
                    return ("consumir_restos", r_id, ("transferir", aliado_crit["id"]))
                mem.consumo_es_transferencia = False
                return ("consumir_restos", r_id, "propia")
            dx, dy = self._paso_seguro(unidad.pos, r_pos, mem, tick)
            return ("mover", dx, dy)

        if mejor_tipo == "hielo" and mejor_target is not None:
            self._transicionar_fsm(mem, EstadoRover.RECOLECTANDO)
            mem.objetivo_hielo = mejor_target
            dx, dy = self._paso_seguro(unidad.pos, mejor_target, mem, tick)
            return ("mover", dx, dy)

        # Despliegue inicial por flancos y cuadrante evitando choque directo central
        if tick < 105 and mem.politica_actual != PoliticaTactica.REPLIEGUE_FINAL:
            self._transicionar_fsm(mem, EstadoRover.EXPLORANDO)
            idx = self._extraer_indice_uid(unidad.id)
            w, h = percepcion.grid_size
            bx, by = percepcion.base_pos
            dir_x = 1 if bx < w // 2 else -1
            dir_y = 1 if by < h // 2 else -1
            if idx % 3 == 0:  # Flanco horizontal
                tx = int(clamp(bx + dir_x * (16 + (idx % 5) * 4), 4, w - 5))
                ty = int(clamp(by + dir_y * (6 + (idx % 3) * 4), 4, h - 5))
            elif idx % 3 == 1:  # Flanco vertical
                tx = int(clamp(bx + dir_x * (6 + (idx % 3) * 4), 4, w - 5))
                ty = int(clamp(by + dir_y * (14 + (idx % 5) * 3), 4, h - 5))
            else:  # Cuadrante interno
                tx = int(clamp(bx + dir_x * (11 + (idx % 4) * 3), 4, w - 5))
                ty = int(clamp(by + dir_y * (10 + (idx % 4) * 3), 4, h - 5))
            if distancia_chebyshev(unidad.pos, (tx, ty)) > 2:
                dx, dy = self._paso_seguro(unidad.pos, (tx, ty), mem, tick)
                return ("mover", dx, dy)

        ev = self._evadir_tormenta_estatico(unidad.pos, mem, tick)
        if ev is not None:
            return ("mover", ev[0], ev[1])

        self._transicionar_fsm(mem, EstadoRover.OCIOSO)
        return ("esperar",)

    # -------------------------------------------------------------------------
    # Sesión 6: Punto de entrada principal no bloqueante y profiling
    # -------------------------------------------------------------------------
    def decidir_unidad(
        self, unidad: UnidadVista, percepcion: Percepcion, api: Any
    ) -> Tuple[Any, ...]:
        t0 = time.perf_counter()
        try:
            if unidad.estado == "muerto":
                return ("esperar",)

            mem = self._obtener_memoria(unidad)
            self._actualizar_percepcion(unidad, percepcion, mem)
            self._procesar_mensajes_acl(unidad, percepcion, api, mem)

            if mem.subastas_activas:
                self._adjudicar_subastas_cnp(unidad, percepcion, api, mem)

            self._enviar_telemetria_gobernanza(unidad, percepcion, api, mem)
            self._votar_aprobacion_copeland(unidad, percepcion, api, mem)

            accion_bat = self._manejar_bateria_baja_d(unidad, percepcion, api, mem)
            if accion_bat is not None:
                return accion_bat

            if unidad.estado == "consumiendo":
                return self._evaluar_hook_consumiendo(unidad, percepcion, mem)

            accion_restos = self._evaluar_restos_y_reportar(unidad, percepcion, api, mem)
            if accion_restos is not None:
                return accion_restos

            if unidad.tipo == "dron_explorador":
                return self._decidir_explorador(unidad, percepcion, mem)
            if unidad.tipo == "dron_recarga":
                return self._decidir_recargador(unidad, percepcion, api, mem)
            return self._decidir_rover(unidad, percepcion, mem)
        finally:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self.latencias_unidad_ms.append(dt_ms)
            t_cur = percepcion.tick
            self.latencias_turno_ms[t_cur] = self.latencias_turno_ms.get(t_cur, 0.0) + dt_ms

    def obtener_resumen_profiling(self) -> Dict[str, Any]:
        turnos = sorted(self.latencias_turno_ms.values())
        if not turnos:
            return {"mean_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "max_ms": 0.0, "ticks": 0}
        n = len(turnos)
        return {
            "mean_ms": sum(turnos) / n,
            "p95_ms": turnos[min(n - 1, int(0.95 * (n - 1)))],
            "p99_ms": turnos[min(n - 1, int(0.99 * (n - 1)))],
            "max_ms": turnos[-1],
            "ticks": n,
            "comms": dict(self.stats_comms),
            "cnp": dict(self.stats_cnp),
            "hook_d": dict(self.stats_hook_d),
            "votaciones": len(self.historial_votaciones),
            "transiciones_fsm": len(self.transiciones_fsm),
        }
