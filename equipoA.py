"""
equipoA.py
==========
Inteligencia autónoma de LA AGENCIA para la Simulación Multiagente Competitiva
en Marte (60x45), heredando estrictamente de `EquipoBase` (`equipo_base.py`).

Arquitectura modular en 6 capas alineada con las sesiones del curso:
  1. Percepción y Memoria Espacial Individual (`MemoriaAgente`):
     Creencias locales por unidad bajo niebla de guerra, invalidación activa,
     decaimiento exponencial de confianza, predicción lineal de tormentas y
     patrulla sectorial sin redundancia.
  2. Utilidad Multi-Atributo y FSM por Rol:
     Estados explícitos para Rover, Dron Explorador y Dron Recargador;
     utilidad U(s, a); encadenamiento de vetas; consumo/transferencia de
     carroña; navegación segura mediante A* acotado ponderado por tormentas.
  3. Comunicación Estructurada FIPA-ACL:
     Mensajes estandarizados (performative, sender, conversation_id, ontology,
     content) con filtros anti-saturación (cooldowns, deduplicación, umbrales
     mínimos y direccionamiento selectivo) para proteger el costo de -0.5%.
  4. Coordinación Contract Net Protocol (CNP):
     Subastas descentralizadas (CFP -> PROPOSE -> ACCEPT/REJECT_PROPOSAL) para
     asignación de vetas de hielo, carroña y despacho de drones recargadores.
  5. Gobernanza Colectiva (Consenso Deliberativo tipo AutoGen):
     Deliberación acotada en tiempo entre roles especializados (Propuesta ->
     Crítica -> Consenso / desempate Copeland) sobre 4 políticas globales que
     ajustan dinámicamente los pesos de U(s, a), umbrales ACL y patrullaje.
  6. Arquitectura No Bloqueante y Profiling (< 40 ms/turno):
     Estructuras O(1), evaluación perezosa de A*, cero llamadas bloqueantes e
     instrumentación interna con `time.perf_counter()`.

Regla de Integridad Arquitectónica:
  Cada agente mantiene únicamente su propia instancia `MemoriaAgente` indexada
  por `unidad.id`. Ningún hallazgo remoto se comparte por variables globales;
  toda transferencia remota pasa exclusivamente por `api` (CommsAPI).
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

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
# Parámetros calibrados vía RL Policy Search (Cross-Entropy Method offline)
# -----------------------------------------------------------------------------
DEFAULT_PARAMS: Dict[str, float] = {
    "w1": 3.80,               # Peso ganancia de hielo
    "w2": 0.28,               # Peso costo de desplazamiento
    "w3": 1.65,               # Peso oportunidad energética de carroña
    "w4": 2.60,               # Peso riesgo de tormenta
    "chain_bonus": 1.50,      # Bonificación al encadenar vetas antes de depositar
    "chain_radius": 13.0,     # Radio Chebyshev máximo para encadenar vetas
    "min_ice_ahorro": 3.0,    # Umbral mínimo de reporte en política ahorro
    "cooldown_hielo": 320.0,  # Cooldown de reporte FIPA-ACL por veta
    "cnp_min_cant": 4.0,      # Cantidad mínima de hielo para abrir subasta CNP
    "expl_paso_post": 6.0,    # Cadencia de patrulla tras completar sector
    "expl_tick_standby": 220.0,  # Tick tras el cual exploradores reducen patrulla
    "rover_tick_expl": 180.0, # Tick hasta el cual rovers ociosos exploran
    "decay_lambda": 0.015,    # Tasa de decaimiento exponencial por tick
    "storm_penalty": 12.0,    # Multiplicador de costo de celda en tormenta A*
}

POLITICAS_ORDEN: Tuple[str, ...] = (
    "EXPANSION_AGRESIVA",
    "AHORRO_ENERGETICO",
    "SUPERVIVENCIA_CARRONA",
    "REPLIEGUE_FINAL",
)

RADIO_VISION: Dict[str, int] = {
    "rover": 5,
    "dron_explorador": 9,
    "dron_recarga": 7,
}

TTL_HIELO = 120
TTL_RESTOS = 70
TTL_TORMENTA = 14
TTL_CONTACTO = 90
TTL_RESERVA_CNP = 32
MAX_EXPANSIONES_ASTAR = 28


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


class PoliticaGlobal(str, Enum):
    EXPANSION_AGRESIVA = "EXPANSION_AGRESIVA"
    AHORRO_ENERGETICO = "AHORRO_ENERGETICO"
    SUPERVIVENCIA_CARRONA = "SUPERVIVENCIA_CARRONA"
    REPLIEGUE_FINAL = "REPLIEGUE_FINAL"


# -----------------------------------------------------------------------------
# Sesión 1: Estructuras de Creencias Espaciales por Agente
# -----------------------------------------------------------------------------
@dataclass
class CreenciaHielo:
    pos: Tuple[int, int]
    cantidad: int
    tick_obs: int
    origen: str  # "local" | "acl"

    def confianza(self, tick_actual: int, lam: float = 0.015) -> float:
        dt = max(0, tick_actual - self.tick_obs)
        return math.exp(-lam * dt)


@dataclass
class CreenciaRestos:
    id: str
    pos: Tuple[int, int]
    combustible: int
    tick_obs: int
    equipo_origen: str

    def confianza(self, tick_actual: int, lam: float = 0.015) -> float:
        dt = max(0, tick_actual - self.tick_obs)
        return math.exp(-lam * dt)


@dataclass
class TrazaTormenta:
    fx: float
    fy: float
    vx: float
    vy: float
    radio: float
    tick_obs: int


@dataclass
class SubastaCNP:
    conversation_id: str
    tipo: str  # "hielo" | "restos" | "rescate"
    objetivo_key: Any
    datos: Dict[str, Any]
    tick_inicio: int
    tick_cierre: int
    candidatos_enviados: List[str] = field(default_factory=list)
    propuestas: List[Dict[str, Any]] = field(default_factory=list)
    cerrada: bool = False


@dataclass
class MemoriaAgente:
    """Memoria espacial y estado de protocolo estrictamente privado de un agente."""

    uid: str
    tipo: str
    estado_fsm: str

    # Sesión 1: Creencias locales bajo niebla de guerra
    hielo: Dict[Tuple[int, int], CreenciaHielo] = field(default_factory=dict)
    restos: Dict[str, CreenciaRestos] = field(default_factory=dict)
    tormentas: List[TrazaTormenta] = field(default_factory=list)
    celdas_visitadas: Dict[Tuple[int, int], int] = field(default_factory=dict)

    # Agenda local de contactos percibidos (para direccionamiento selectivo CNP/ACL)
    contactos_rovers: Dict[str, Tuple[Tuple[int, int], float, str, int]] = field(
        default_factory=dict
    )  # uid -> (pos, bat, estado, tick)
    contactos_recargadores: Dict[str, Tuple[Tuple[int, int], float, str, int]] = field(
        default_factory=dict
    )

    # Patrulla y tareas locales
    waypoints: Optional[List[Tuple[int, int]]] = None
    patrulla_idx: int = 0
    ciclo_patrulla_completo: bool = False
    objetivo_hielo: Optional[Tuple[int, int]] = None
    objetivo_restos: Optional[Dict[str, Any]] = None
    rescate_actual: Optional[Dict[str, Any]] = None
    ticks_en_rescate: int = 0

    # Sesión 3: Filtros anti-saturación FIPA-ACL
    cooldowns: Dict[Any, int] = field(default_factory=dict)
    dedup_reportes: Dict[Any, int] = field(default_factory=dict)
    ultimo_ping_recarga: int = -999
    esperando_agree_rescate: bool = False
    rescate_confirmado_por: Optional[str] = None

    # Sesión 4: Subastas CNP iniciadas y reservas
    subastas_activas: Dict[str, SubastaCNP] = field(default_factory=dict)
    reservas_ajenas: Dict[Any, int] = field(default_factory=dict)  # target_key -> expire_tick
    contrato_actual: Optional[Dict[str, Any]] = None

    # Sesión 5: Política local conocida vía FIPA-ACL o percepción local
    politica_actual: PoliticaGlobal = PoliticaGlobal.EXPANSION_AGRESIVA
    politica_epoch: int = 0
    ultimo_tick_gov_envio: int = -999
    telemetria_gov_recibida: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    bajas_conocidas: Set[str] = field(default_factory=set)


class EquipoA(EquipoBase):
    """
    Agente multi-estrategia para la competición en Marte (LA AGENCIA).
    Hereda estrictamente de `EquipoBase` sin modificar `simulador.py` ni
    `equipo_base.py`.
    """

    nombre: str = "LA AGENCIA"

    def __init__(
        self,
        params: Optional[Dict[str, float]] = None,
        unthrottled_comms: bool = False,
        disable_cnp: bool = False,
    ) -> None:
        super().__init__()
        self.params: Dict[str, float] = dict(DEFAULT_PARAMS)
        if params:
            self.params.update(params)
        self.unthrottled_comms: bool = unthrottled_comms
        self.disable_cnp: bool = disable_cnp

        # Almacenamiento indexado por uid (cada agente solo accede a self._agentes[unidad.id])
        self._agentes: Dict[str, MemoriaAgente] = {}
        # Registro estático de IDs propios descubiertos (sin info espacial/recursos)
        self._ids_por_rol: Dict[str, List[str]] = {
            "rover": [],
            "dron_explorador": [],
            "dron_recarga": [],
        }

        # Telemetría de auditoría / informe (solo lectura para análisis posterior)
        self.latencias_unidad_ms: List[float] = []
        self.latencias_turno_ms: Dict[int, float] = {}
        self.stats_comms: Dict[str, Any] = {
            "enviados": 0,
            "bloqueados_filtro": 0,
            "por_performativa": {},
            "por_ontologia": {},
        }
        self.stats_cnp: Dict[str, int] = {
            "cfp_enviados": 0,
            "propose_recibidos": 0,
            "accept_enviados": 0,
            "reject_enviados": 0,
            "duplicados_evitados": 0,
        }
        self.transiciones_fsm: Dict[str, int] = {}
        self.historial_consenso: List[Dict[str, Any]] = []
        self.cobertura_por_tick: Dict[int, int] = {}
        self._celdas_exploradas_equipo_eval: Set[Tuple[int, int]] = set()
        self._centros_eval_vistos: Set[Tuple[int, int, int]] = set()
        self._tick_fase_final: Optional[int] = None

    def inicializar(
        self,
        letra: str,
        base_pos: Tuple[int, int],
        grid_size: Tuple[int, int] = (GRID_W, GRID_H),
    ) -> None:
        super().inicializar(letra, base_pos, grid_size)
        self._agentes.clear()
        self._ids_por_rol = {
            "rover": [],
            "dron_explorador": [],
            "dron_recarga": [],
        }
        self.latencias_unidad_ms.clear()
        self.latencias_turno_ms.clear()
        self.stats_comms = {
            "enviados": 0,
            "bloqueados_filtro": 0,
            "por_performativa": {},
            "por_ontologia": {},
        }
        self.stats_cnp = {
            "cfp_enviados": 0,
            "propose_recibidos": 0,
            "accept_enviados": 0,
            "reject_enviados": 0,
            "duplicados_evitados": 0,
        }
        self.transiciones_fsm.clear()
        self.historial_consenso.clear()
        self.cobertura_por_tick.clear()
        self._celdas_exploradas_equipo_eval.clear()
        self._centros_eval_vistos.clear()
        self._tick_fase_final = None

    # -------------------------------------------------------------------------
    # Utilidades internas de ciclo de vida del agente
    # -------------------------------------------------------------------------
    def _obtener_memoria(self, unidad: UnidadVista) -> MemoriaAgente:
        if unidad.id not in self._agentes:
            if unidad.tipo == "rover":
                estado_ini = EstadoRover.OCIOSO.value
            elif unidad.tipo == "dron_explorador":
                estado_ini = EstadoExplorador.PATRULLANDO.value
            else:
                estado_ini = EstadoRecargador.EN_ESPERA.value

            self._agentes[unidad.id] = MemoriaAgente(
                uid=unidad.id,
                tipo=unidad.tipo,
                estado_fsm=estado_ini,
            )
            lista = self._ids_por_rol.get(unidad.tipo)
            if lista is not None and unidad.id not in lista:
                lista.append(unidad.id)
                lista.sort(key=self._extraer_indice_uid)
        return self._agentes[unidad.id]

    @staticmethod
    def _extraer_indice_uid(uid: str) -> int:
        try:
            return int(uid.split("_")[-1])
        except Exception:
            return abs(hash(uid)) % 100

    def _transicionar_fsm(self, mem: MemoriaAgente, nuevo_estado: Enum) -> None:
        val = nuevo_estado.value
        if mem.estado_fsm != val:
            clave = f"{mem.tipo}:{mem.estado_fsm}->{val}"
            self.transiciones_fsm[clave] = self.transiciones_fsm.get(clave, 0) + 1
            mem.estado_fsm = val

    def _registrar_cobertura_eval(self, unidad: UnidadVista, tick: int) -> None:
        """Registra cobertura acumulada O(1) amortizado para la curva del informe."""
        rv = RADIO_VISION.get(unidad.tipo, 5)
        clave = (unidad.pos[0], unidad.pos[1], rv)
        if clave not in self._centros_eval_vistos:
            self._centros_eval_vistos.add(clave)
            ux, uy = unidad.pos
            w, h = self.grid_size
            x0 = max(0, ux - rv)
            x1 = min(w - 1, ux + rv)
            y0 = max(0, uy - rv)
            y1 = min(h - 1, uy + rv)
            for x in range(x0, x1 + 1):
                for y in range(y0, y1 + 1):
                    self._celdas_exploradas_equipo_eval.add((x, y))
        self.cobertura_por_tick[tick] = len(self._celdas_exploradas_equipo_eval)

    # -------------------------------------------------------------------------
    # Sesión 1: Percepción, actualización de creencias, olvido y sectorización
    # -------------------------------------------------------------------------
    def _actualizar_percepcion_y_olvido(
        self, unidad: UnidadVista, percepcion: Percepcion, mem: MemoriaAgente
    ) -> None:
        tick = percepcion.tick
        rv = RADIO_VISION.get(unidad.tipo, 5)
        ux, uy = unidad.pos

        # Marcar celdas dentro de visión local (para frontera/cobertura del propio agente)
        mem.celdas_visitadas[unidad.pos] = tick
        lam = self.params["decay_lambda"]

        # 1. Actualizar hielo visible y purgar vetas vaciadas dentro del radio de visión
        hielo_vis_set: Set[Tuple[int, int]] = set()
        for hx, hy, cant in percepcion.hielo_visible:
            if cant > 0:
                pos = (hx, hy)
                hielo_vis_set.add(pos)
                mem.hielo[pos] = CreenciaHielo(
                    pos=pos,
                    cantidad=cant,
                    tick_obs=tick,
                    origen="local",
                )

        a_borrar_hielo: List[Tuple[int, int]] = []
        for pos, cr in mem.hielo.items():
            # Si la celda está dentro de nuestra visión y ya no aparece, está vacía
            if max(abs(pos[0] - ux), abs(pos[1] - uy)) <= rv and pos not in hielo_vis_set:
                a_borrar_hielo.append(pos)
            elif tick - cr.tick_obs > TTL_HIELO or cr.confianza(tick, lam) < 0.14:
                a_borrar_hielo.append(pos)
        for pos in a_borrar_hielo:
            mem.hielo.pop(pos, None)
            if mem.objetivo_hielo == pos:
                mem.objetivo_hielo = None

        # 2. Actualizar restos visibles y purgar consumidos/expirados
        restos_vis_ids: Set[str] = set()
        for r in percepcion.restos_visible:
            rid = r["id"]
            rpos = tuple(r["pos"])
            restos_vis_ids.add(rid)
            mem.restos[rid] = CreenciaRestos(
                id=rid,
                pos=rpos,
                combustible=int(r.get("combustible", 10)),
                tick_obs=tick,
                equipo_origen=r.get("equipo", "?"),
            )
            if r.get("equipo") == self.letra:
                mem.bajas_conocidas.add(rid)

        a_borrar_restos: List[str] = []
        for rid, cr in mem.restos.items():
            if max(abs(cr.pos[0] - ux), abs(cr.pos[1] - uy)) <= rv and rid not in restos_vis_ids:
                a_borrar_restos.append(rid)
            elif tick - cr.tick_obs > TTL_RESTOS:
                a_borrar_restos.append(rid)
        for rid in a_borrar_restos:
            mem.restos.pop(rid, None)
            if mem.objetivo_restos and mem.objetivo_restos.get("id") == rid:
                mem.objetivo_restos = None

        # 3. Actualizar trazas de tormentas con estimación de velocidad (vx, vy)
        prev_tormentas = [t for t in mem.tormentas if tick - t.tick_obs <= TTL_TORMENTA]
        nuevas_tormentas: List[TrazaTormenta] = []
        usadas_prev: Set[int] = set()

        for tx, ty, rad in percepcion.tormentas_visible:
            mejor_idx = -1
            mejor_d = 6.5
            for idx, pt in enumerate(prev_tormentas):
                if idx in usadas_prev:
                    continue
                d = math.hypot(tx - pt.fx, ty - pt.fy)
                if d < mejor_d:
                    mejor_d = d
                    mejor_idx = idx
            vx, vy = 0.0, 0.0
            if mejor_idx >= 0:
                pt = prev_tormentas[mejor_idx]
                usadas_prev.add(mejor_idx)
                dt_ticks = max(1, tick - pt.tick_obs)
                vx = clamp((tx - pt.fx) / dt_ticks, -1.5, 1.5)
                vy = clamp((ty - pt.fy) / dt_ticks, -1.5, 1.5)
            nuevas_tormentas.append(
                TrazaTormenta(fx=tx, fy=ty, vx=vx, vy=vy, radio=rad, tick_obs=tick)
            )

        # Conservar brevemente trazas recientes no re-observadas
        for idx, pt in enumerate(prev_tormentas):
            if idx not in usadas_prev and tick - pt.tick_obs <= 6:
                nuevas_tormentas.append(
                    TrazaTormenta(
                        fx=clamp(pt.fx + pt.vx, 0.0, self.grid_size[0] - 1.0),
                        fy=clamp(pt.fy + pt.vy, 0.0, self.grid_size[1] - 1.0),
                        vx=pt.vx,
                        vy=pt.vy,
                        radio=pt.radio,
                        tick_obs=pt.tick_obs,
                    )
                )
        mem.tormentas = nuevas_tormentas

        # 4. Actualizar contactos locales visibles (para preselección CNP)
        for al in percepcion.aliados_visible:
            aid = al["id"]
            if aid == unidad.id:
                continue
            info = (tuple(al["pos"]), float(al["bateria"]), str(al["estado"]), tick)
            if al.get("tipo") == "rover":
                mem.contactos_rovers[aid] = info
            elif al.get("tipo") == "dron_recarga":
                mem.contactos_recargadores[aid] = info

        # Purgar contactos antiguos y reservas expiradas
        mem.contactos_rovers = {
            k: v for k, v in mem.contactos_rovers.items() if tick - v[3] <= TTL_CONTACTO
        }
        mem.contactos_recargadores = {
            k: v for k, v in mem.contactos_recargadores.items() if tick - v[3] <= TTL_CONTACTO
        }
        mem.reservas_ajenas = {
            k: exp for k, exp in mem.reservas_ajenas.items() if exp >= tick
        }

        # 5. Transición automática local en fase final (percepción directa del entorno)
        if percepcion.fase_final:
            if self._tick_fase_final is None:
                self._tick_fase_final = tick
            if mem.politica_actual != PoliticaGlobal.REPLIEGUE_FINAL:
                mem.politica_actual = PoliticaGlobal.REPLIEGUE_FINAL
                mem.politica_epoch = max(mem.politica_epoch, 999)

    def _generar_waypoints_explorador_p(self, unidad: UnidadVista) -> List[Tuple[int, int]]:
        """
        Particionamiento sectorial dinámico sin redundancia adaptado al tamaño real
        de la flota exploradora (soporta 15, 30 o cualquier número de exploradores).
        """
        w, h = self.grid_size
        idx = self._extraer_indice_uid(unidad.id)
        total_expl = max(15, len(self._ids_por_rol["dron_explorador"]))
        cols = 5 if total_expl <= 20 else 6
        rows = max(3, math.ceil(total_expl / cols))
        sector = idx % (cols * rows)

        # Invertir orden de sector según cuadrante de base para priorizar zona propia primero
        bx, by = self.base_pos
        sx_raw = sector % cols
        sy_raw = sector // cols
        sx = sx_raw if bx < w // 2 else (cols - 1 - sx_raw)
        sy = sy_raw if by < h // 2 else (rows - 1 - sy_raw)

        x0 = int(clamp((sx + 0.22) * (w / cols), 5, w - 6))
        x1 = int(clamp((sx + 0.78) * (w / cols), 5, w - 6))
        y0 = int(clamp((sy + 0.22) * (h / rows), 5, h - 6))
        y1 = int(clamp((sy + 0.78) * (h / rows), 5, h - 6))

        # Barrido compacto de 3 waypoints aprovechando radio de visión 9 (19x19)
        if sy % 2 == 0:
            wps = [(x0, y0), (x1, y1), (x0, y1)]
        else:
            wps = [(x1, y0), (x0, y1), (x1, y1)]

        # Empezar por el waypoint más cercano a la base para minimizar trayecto inicial
        wps.sort(key=lambda p: distancia_chebyshev(self.base_pos, p))
        return wps

    # -------------------------------------------------------------------------
    # Sesión 3: Comunicación Estructurada FIPA-ACL y Control Presupuestario
    # -------------------------------------------------------------------------
    def _enviar_acl(
        self,
        api: Any,
        mem: MemoriaAgente,
        dest_id: str,
        performative: str,
        ontology: str,
        conversation_id: str,
        content: Dict[str, Any],
        asunto_legado: Optional[str] = None,
    ) -> bool:
        """Envía un paquete FIPA-ACL estandarizado por `CommsAPI` y registra métricas."""
        payload_content = dict(content)
        payload_content["_gov"] = {
            "politica": mem.politica_actual.value,
            "epoch": mem.politica_epoch,
        }
        datos = {
            "performative": performative,
            "sender": mem.uid,
            "conversation_id": conversation_id,
            "ontology": ontology,
            "content": payload_content,
            # Campos de compatibilidad con inspectores legados
            **{k: v for k, v in content.items() if k in ("pos", "cantidad", "id", "rover_id")},
        }
        asunto = asunto_legado or f"{ontology}:{performative}"
        ok = api.enviar_mensaje(dest_id, asunto, datos)
        if ok:
            self.stats_comms["enviados"] += 1
            self.stats_comms["por_performativa"][performative] = (
                self.stats_comms["por_performativa"].get(performative, 0) + 1
            )
            self.stats_comms["por_ontologia"][ontology] = (
                self.stats_comms["por_ontologia"].get(ontology, 0) + 1
            )
        return ok

    def _enviar_acl_por_helper(
        self,
        api: Any,
        mem: MemoriaAgente,
        helper_name: str,
        performative: str,
        ontology: str,
        conversation_id: str,
        content: Dict[str, Any],
        asunto_legado: str,
        pos_ref: Optional[Tuple[int, int]] = None,
        radio: int = 8,
        max_dest: int = 1,
    ) -> int:
        payload_content = dict(content)
        payload_content["_gov"] = {
            "politica": mem.politica_actual.value,
            "epoch": mem.politica_epoch,
        }
        datos = {
            "performative": performative,
            "sender": mem.uid,
            "conversation_id": conversation_id,
            "ontology": ontology,
            "content": payload_content,
            **{k: v for k, v in content.items() if k in ("pos", "cantidad", "id", "rover_id")},
        }
        enviados = 0
        if helper_name == "recargador":
            res = api.enviar_a_recargador_cercano(asunto_legado, datos)
            enviados = 1 if res else 0
        elif helper_name == "rover":
            res = api.enviar_a_rover_ocioso_cercano(asunto_legado, datos, pos_ref=pos_ref)
            enviados = 1 if res else 0
        elif helper_name == "cercanos":
            enviados = api.enviar_a_aliados_cercanos(
                radio, asunto_legado, datos, max_dest=max_dest
            )
        if enviados > 0:
            self.stats_comms["enviados"] += enviados
            self.stats_comms["por_performativa"][performative] = (
                self.stats_comms["por_performativa"].get(performative, 0) + enviados
            )
            self.stats_comms["por_ontologia"][ontology] = (
                self.stats_comms["por_ontologia"].get(ontology, 0) + enviados
            )
        return enviados

    def _pasa_filtro_antisaturacion(
        self, mem: MemoriaAgente, clave: Any, tick: int, cooldown: int
    ) -> bool:
        if self.unthrottled_comms:
            return True
        ultimo = mem.cooldowns.get(clave, -9999)
        if tick - ultimo < cooldown:
            self.stats_comms["bloqueados_filtro"] += 1
            return False
        mem.cooldowns[clave] = tick
        return True

    def _umbral_min_hielo_reporte(self, mem: MemoriaAgente) -> int:
        if self.unthrottled_comms:
            return 1
        if mem.politica_actual == PoliticaGlobal.AHORRO_ENERGETICO:
            return int(round(self.params["min_ice_ahorro"]))
        if mem.politica_actual == PoliticaGlobal.SUPERVIVENCIA_CARRONA:
            return 3
        if mem.politica_actual == PoliticaGlobal.REPLIEGUE_FINAL:
            return 99
        return 2

    def _procesar_mensajes_acl(
        self, unidad: UnidadVista, percepcion: Percepcion, api: Any, mem: MemoriaAgente
    ) -> None:
        tick = percepcion.tick

        for msg in percepcion.mensajes:
            emisor = msg.get("emisor_id", "")
            asunto = msg.get("asunto", "")
            datos = msg.get("datos", {})

            performative = datos.get("performative")
            ontology = datos.get("ontology", "")
            conv_id = datos.get("conversation_id", f"legacy_{tick}")
            content = datos.get("content", datos)

            # Sincronizar época de política global si viene adjunta en FIPA-ACL
            gov = content.get("_gov") if isinstance(content, dict) else None
            if gov and gov.get("epoch", 0) > mem.politica_epoch:
                try:
                    mem.politica_actual = PoliticaGlobal(gov["politica"])
                    mem.politica_epoch = int(gov["epoch"])
                except Exception:
                    pass

            # Compatibilidad por si llega mensaje de formato simple
            if not performative:
                if asunto == "hielo":
                    performative, ontology = "INFORM", "resource-discovery"
                elif asunto == "restos":
                    performative, ontology = "INFORM", "resource-discovery"
                elif asunto == "solicitud_recarga":
                    performative, ontology = "REQUEST", "rescue-ops"
                else:
                    performative = "INFORM"

            # Manejo por ontología y performativa
            if ontology == "governance":
                if performative == "INFORM":
                    if content.get("tipo") == "telemetria":
                        mem.telemetria_gov_recibida[emisor] = {
                            **content,
                            "tick": tick,
                        }
                    elif "politica" in content:
                        ep = int(content.get("epoch", 0))
                        if ep > mem.politica_epoch:
                            mem.politica_actual = PoliticaGlobal(content["politica"])
                            mem.politica_epoch = ep

            elif ontology == "resource-discovery" and performative == "INFORM":
                if "pos" in content and content.get("status") == "depleted":
                    pos = tuple(content["pos"])
                    mem.hielo.pop(pos, None)
                    if mem.objetivo_hielo == pos:
                        mem.objetivo_hielo = None
                elif "cantidad" in content and "pos" in content:
                    pos = tuple(content["pos"])
                    cant = int(content["cantidad"])
                    mem.hielo[pos] = CreenciaHielo(
                        pos=pos, cantidad=cant, tick_obs=tick, origen="acl"
                    )
                    mem.dedup_reportes[("hielo", pos)] = tick
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
                    mem.dedup_reportes[("restos", rid)] = tick
                    if content.get("equipo") == self.letra:
                        mem.bajas_conocidas.add(rid)
                    if (
                        unidad.tipo == "rover"
                        and mem.objetivo_restos is None
                        and unidad.bateria < 78.0
                        and unidad.carga == 0
                    ):
                        mem.objetivo_restos = {"id": rid, "pos": rpos}

            elif ontology == "rescue-ops":
                if performative == "REQUEST" and unidad.tipo == "dron_recarga":
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
                elif performative == "AGREE" and unidad.tipo == "rover":
                    mem.esperando_agree_rescate = False
                    mem.rescate_confirmado_por = emisor
                elif performative == "FAILURE" and unidad.tipo == "rover":
                    mem.esperando_agree_rescate = False
                    mem.rescate_confirmado_por = None
                    mem.ultimo_ping_recarga = -999

            elif ontology == "cnp-task":
                self._manejar_mensaje_cnp(
                    unidad, percepcion, api, mem, emisor, performative, conv_id, content
                )

    # -------------------------------------------------------------------------
    # Sesión 4: Contract Net Protocol (CNP) — Pujas, Adjudicación y Reservas
    # -------------------------------------------------------------------------
    def _calcular_puja_cnp(
        self, unidad: UnidadVista, percepcion: Percepcion, mem: MemoriaAgente, cnp_data: Dict[str, Any]
    ) -> float:
        tipo_tarea = cnp_data.get("tipo_tarea")
        obj_pos = tuple(cnp_data["pos"])
        dist = distancia_chebyshev(unidad.pos, obj_pos)
        riesgo_torm = self._riesgo_tormenta_segmento(unidad.pos, obj_pos, mem, percepcion.tick)

        if tipo_tarea == "rescate" and unidad.tipo == "dron_recarga":
            if unidad.carga_recarga <= 0 or unidad.bateria <= UMBRAL_BATERIA_BAJA + 5.0:
                return -999.0
            ocupado_pen = 30.0 if mem.rescate_actual is not None else 0.0
            return (
                40.0
                - 1.4 * dist
                + 15.0 * (unidad.bateria / 100.0)
                - 8.0 * riesgo_torm
                - ocupado_pen
            )

        if unidad.tipo == "rover":
            if unidad.estado == "esperando_recarga" or unidad.bateria <= UMBRAL_BATERIA_BAJA + 6.0:
                return -999.0
            cap_libre = max(0, unidad.capacidad - unidad.carga)
            if tipo_tarea == "hielo":
                if cap_libre <= 0 or mem.objetivo_hielo is not None:
                    return -999.0
                cant = min(int(cnp_data.get("cantidad", 1)), cap_libre)
                dist_base = max(0, distancia_chebyshev(obj_pos, percepcion.base_pos) - 2)
                return (
                    14.0 * cant
                    - 1.25 * dist
                    - 0.35 * dist_base
                    + 10.0 * (unidad.bateria / 100.0)
                    + 6.0 * (cap_libre / max(1, unidad.capacidad))
                    - 6.0 * riesgo_torm
                )
            if tipo_tarea == "restos":
                if unidad.carga > 0 or unidad.bateria >= 88.0:
                    return -999.0
                comb = int(cnp_data.get("combustible", 10))
                necesidad = max(0.0, (90.0 - unidad.bateria) / 20.0)
                return (
                    8.0 * (comb / 10.0) * (1.0 + necesidad)
                    - 1.3 * dist
                    - 5.0 * riesgo_torm
                )
        return -999.0

    def _manejar_mensaje_cnp(
        self,
        unidad: UnidadVista,
        percepcion: Percepcion,
        api: Any,
        mem: MemoriaAgente,
        emisor: str,
        performative: str,
        conv_id: str,
        content: Dict[str, Any],
    ) -> None:
        tick = percepcion.tick
        if performative == "CFP":
            if self.disable_cnp or unidad.bateria <= 15.0:
                return
            puja = self._calcular_puja_cnp(unidad, percepcion, mem, content)
            if puja > 0.0:
                self._enviar_acl(
                    api,
                    mem,
                    emisor,
                    "PROPOSE",
                    "cnp-task",
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
            subasta = mem.subastas_activas.get(conv_id)
            if subasta is not None and not subasta.cerrada:
                self.stats_cnp["propose_recibidos"] += 1
                subasta.propuestas.append(
                    {
                        "postor": emisor,
                        "bid": float(content.get("bid", 0.0)),
                        "bateria": float(content.get("bateria", 0.0)),
                    }
                )

        elif performative == "ACCEPT_PROPOSAL":
            tipo_tarea = content.get("tipo_tarea")
            pos = tuple(content["pos"])
            if tipo_tarea == "hielo" and unidad.tipo == "rover":
                mem.objetivo_hielo = pos
                mem.contrato_actual = {"conv_id": conv_id, "tipo": "hielo", "pos": pos}
            elif tipo_tarea == "restos" and unidad.tipo == "rover":
                mem.objetivo_restos = {"id": content["id"], "pos": pos}
                mem.contrato_actual = {"conv_id": conv_id, "tipo": "restos", "id": content["id"], "pos": pos}
            elif tipo_tarea == "rescate" and unidad.tipo == "dron_recarga":
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
            if content.get("tipo_tarea") == "hielo":
                pos = tuple(content["pos"])
                mem.reservas_ajenas[("hielo", pos)] = tick + TTL_RESERVA_CNP
                if mem.objetivo_hielo == pos:
                    mem.objetivo_hielo = None
            elif content.get("tipo_tarea") == "restos":
                rid = content.get("id")
                if rid:
                    mem.reservas_ajenas[("restos", rid)] = tick + TTL_RESERVA_CNP

    def _iniciar_subasta_cnp(
        self,
        unidad: UnidadVista,
        percepcion: Percepcion,
        api: Any,
        mem: MemoriaAgente,
        tipo_tarea: str,
        objetivo_key: Any,
        datos_tarea: Dict[str, Any],
    ) -> bool:
        """Inicia una subasta CNP preseleccionando hasta 2 candidatos cercanos."""
        if self.disable_cnp or unidad.bateria <= 18.0 or len(mem.subastas_activas) >= 2:
            return False

        tick = percepcion.tick
        if objetivo_key in mem.reservas_ajenas and mem.reservas_ajenas[objetivo_key] >= tick:
            self.stats_cnp["duplicados_evitados"] += 1
            return False

        pos_ref = tuple(datos_tarea["pos"])
        candidatos: List[str] = []

        if tipo_tarea in ("hielo", "restos"):
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
        ventana = 4 if tipo_tarea in ("hielo", "restos") else 3

        if candidatos:
            enviados: List[str] = []
            for dest_id in candidatos:
                if self._enviar_acl(
                    api,
                    mem,
                    dest_id,
                    "CFP",
                    "cnp-task",
                    conv_id,
                    {"tipo_tarea": tipo_tarea, **datos_tarea},
                    asunto_legado="cnp_cfp",
                ):
                    enviados.append(dest_id)
                    self.stats_cnp["cfp_enviados"] += 1
            if enviados:
                mem.subastas_activas[conv_id] = SubastaCNP(
                    conversation_id=conv_id,
                    tipo=tipo_tarea,
                    objetivo_key=objetivo_key,
                    datos=datos_tarea,
                    tick_inicio=tick,
                    tick_cierre=tick + ventana,
                    candidatos_enviados=enviados,
                )
                return True

        # Fallback selectivo cuando aún no hay suficientes contactos en la agenda local
        if tipo_tarea in ("hielo", "restos"):
            n = self._enviar_acl_por_helper(
                api,
                mem,
                "rover",
                "INFORM",
                "resource-discovery",
                conv_id,
                datos_tarea,
                asunto_legado=tipo_tarea,
                pos_ref=pos_ref,
            )
            if n > 0:
                mem.reservas_ajenas[objetivo_key] = tick + TTL_RESERVA_CNP
            return n > 0

        return False

    def _adjudicar_subastas_cnp(
        self, unidad: UnidadVista, percepcion: Percepcion, api: Any, mem: MemoriaAgente
    ) -> None:
        """Cierra subastas CNP vencidas adjudicando al mejor postor y rechazando al resto."""
        tick = percepcion.tick
        cerradas: List[str] = []

        for conv_id, sub in mem.subastas_activas.items():
            if sub.cerrada:
                cerradas.append(conv_id)
                continue
            esperar_mas = tick < sub.tick_cierre and len(sub.propuestas) < len(sub.candidatos_enviados)
            if esperar_mas:
                continue

            sub.cerrada = True
            cerradas.append(conv_id)

            if not sub.propuestas:
                # Si ningún postor pujó y era hielo/restos valioso, fallback 1 INFORM selectivo
                if sub.tipo in ("hielo", "restos") and unidad.bateria > 20.0:
                    self._enviar_acl_por_helper(
                        api,
                        mem,
                        "rover",
                        "INFORM",
                        "resource-discovery",
                        conv_id,
                        sub.datos,
                        asunto_legado=sub.tipo,
                        pos_ref=tuple(sub.datos["pos"]),
                    )
                continue

            # Desempate determinista: mayor bid -> mayor batería -> uid alfabético
            sub.propuestas.sort(key=lambda p: (-p["bid"], -p["bateria"], p["postor"]))
            ganador = sub.propuestas[0]["postor"]
            self._enviar_acl(
                api,
                mem,
                ganador,
                "ACCEPT_PROPOSAL",
                "cnp-task",
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
                    "cnp-task",
                    conv_id,
                    {"tipo_tarea": sub.tipo, **sub.datos},
                    asunto_legado="cnp_reject",
                )
                self.stats_cnp["reject_enviados"] += 1
                self.stats_cnp["duplicados_evitados"] += 1

        for cid in cerradas:
            mem.subastas_activas.pop(cid, None)

    # -------------------------------------------------------------------------
    # Sesión 5: Gobernanza Colectiva — Consenso Deliberativo Inspirado en AutoGen
    # -------------------------------------------------------------------------
    def _enviar_telemetria_gobernanza(
        self, unidad: UnidadVista, percepcion: Percepcion, api: Any, mem: MemoriaAgente
    ) -> None:
        """Los 3 representantes envían telemetría compacta al ModeradorConsenso."""
        tick = percepcion.tick
        if tick < 18:
            return
        moderador_id = f"{self.letra}_rover_0"
        if unidad.id == moderador_id:
            return

        rol_id_map = {
            f"{self.letra}_expl_0": "ExploradorJefe",
            f"{self.letra}_rover_1": "OficialLogistica",
            f"{self.letra}_rec_0": "OficialSeguridad",
        }
        rol = rol_id_map.get(unidad.id)
        if rol is None:
            return

        intervalo = 60 if not percepcion.fase_final else 35
        if tick - mem.ultimo_tick_gov_envio < intervalo:
            return
        mem.ultimo_tick_gov_envio = tick

        if rol == "ExploradorJefe":
            vetas_ricas = sum(1 for c in mem.hielo.values() if c.cantidad >= 4)
            datos_tel = {
                "tipo": "telemetria",
                "rol": rol,
                "hielo_conocido": len(mem.hielo),
                "vetas_ricas": vetas_ricas,
                "cobertura_local": len(mem.celdas_visitadas),
            }
        elif rol == "OficialLogistica":
            datos_tel = {
                "tipo": "telemetria",
                "rol": rol,
                "bateria": round(unidad.bateria, 1),
                "carga": unidad.carga,
                "rovers_activos_vistos": len(mem.contactos_rovers),
            }
        else:
            datos_tel = {
                "tipo": "telemetria",
                "rol": rol,
                "tormentas_vistas": len(mem.tormentas),
                "restos_conocidos": len(mem.restos),
                "bajas_conocidas": len(mem.bajas_conocidas),
            }

        self._enviar_acl(
            api,
            mem,
            moderador_id,
            "INFORM",
            "governance",
            f"gov_tel_{rol}_{tick}",
            datos_tel,
            asunto_legado="gov_telemetria",
        )

    def _deliberar_autogen_y_propagar(
        self, unidad: UnidadVista, percepcion: Percepcion, api: Any, mem: MemoriaAgente
    ) -> None:
        """
        Ejecuta el protocolo de consenso deliberativo inspirado en AutoGen
        entre 4 roles especializados en < 1 ms (sin llamadas bloqueantes) y
        resuelve empates con matriz de comparación por pares Copeland.
        """
        moderador_id = f"{self.letra}_rover_0"
        if unidad.id != moderador_id:
            return

        tick = percepcion.tick
        ultimo_ciclo = self.historial_consenso[-1]["tick"] if self.historial_consenso else -999
        if not percepcion.fase_final and (tick < 30 or tick - ultimo_ciclo < 45):
            return
        if percepcion.fase_final and mem.politica_actual == PoliticaGlobal.REPLIEGUE_FINAL:
            return

        # Evidencia local + telemetría recibida vía FIPA-ACL
        tel_expl = mem.telemetria_gov_recibida.get(f"{self.letra}_expl_0", {})
        tel_log = mem.telemetria_gov_recibida.get(f"{self.letra}_rover_1", {})
        tel_seg = mem.telemetria_gov_recibida.get(f"{self.letra}_rec_0", {})

        hielo_visto = max(len(mem.hielo), int(tel_expl.get("hielo_conocido", 0)))
        vetas_ricas = int(tel_expl.get("vetas_ricas", 0))
        restos_visto = max(len(mem.restos), int(tel_seg.get("restos_conocidos", 0)))
        bajas = max(len(mem.bajas_conocidas), int(tel_seg.get("bajas_conocidas", 0)))
        tormentas = max(len(mem.tormentas), int(tel_seg.get("tormentas_vistas", 0)))
        bat_prom = (unidad.bateria + float(tel_log.get("bateria", unidad.bateria))) * 0.5

        # --- Ronda 1: Propuestas iniciales por rol ---
        def score_rol(rol: str, ronda: int, criticas: Dict[str, str]) -> List[str]:
            # Scores por política para el rol dado
            s: Dict[str, float] = {p: 0.0 for p in POLITICAS_ORDEN}

            if percepcion.fase_final:
                s["REPLIEGUE_FINAL"] += 100.0

            if rol == "ExploradorJefe":
                s["EXPANSION_AGRESIVA"] += 3.0 + vetas_ricas * 0.8 + (2.0 if tick < 350 else 0.0)
                s["AHORRO_ENERGETICO"] += 2.2 + (2.5 if tick >= 350 and hielo_visto > 3 else 0.0)
                s["SUPERVIVENCIA_CARRONA"] += 0.5 + restos_visto * 0.6
                s["REPLIEGUE_FINAL"] += 4.0 if (tick > 250 and hielo_visto == 0) else 0.0
            elif rol == "OficialLogistica":
                s["AHORRO_ENERGETICO"] += 3.5 + (2.0 if bat_prom < 75.0 else 0.5)
                s["EXPANSION_AGRESIVA"] += 2.5 if (vetas_ricas >= 2 and bat_prom > 70.0) else 1.0
                s["SUPERVIVENCIA_CARRONA"] += 1.2 + restos_visto * 0.7
                s["REPLIEGUE_FINAL"] += 1.5 if tick > 450 and hielo_visto == 0 else 0.0
            elif rol == "OficialSeguridad":
                s["SUPERVIVENCIA_CARRONA"] += 1.0 + bajas * 1.8 + min(3, restos_visto) * 0.45
                s["AHORRO_ENERGETICO"] += 3.0 + tormentas * 1.3
                s["EXPANSION_AGRESIVA"] += 2.2 if (bajas == 0 and tormentas == 0) else 0.4
                s["REPLIEGUE_FINAL"] += 1.8 if tormentas >= 3 else 0.0
            else:  # ModeradorConsenso
                s["AHORRO_ENERGETICO"] += 3.0
                s["EXPANSION_AGRESIVA"] += 2.4 if tick < 250 else 1.0
                s["SUPERVIVENCIA_CARRONA"] += 1.5 + bajas * 1.0
                s["REPLIEGUE_FINAL"] += 2.5 if (tick > 400 and hielo_visto == 0) else 0.0

            # Ronda 2: Ajuste por crítica cruzada de pares
            if ronda == 2:
                if criticas.get("OficialSeguridad") in ("SUPERVIVENCIA_CARRONA", "AHORRO_ENERGETICO"):
                    s["EXPANSION_AGRESIVA"] -= 0.9
                    s[criticas["OficialSeguridad"]] += 0.8
                if criticas.get("ExploradorJefe") == "EXPANSION_AGRESIVA" and vetas_ricas >= 3:
                    s["EXPANSION_AGRESIVA"] += 0.9

            return sorted(POLITICAS_ORDEN, key=lambda p: (-s[p], POLITICAS_ORDEN.index(p)))

        roles = ("ExploradorJefe", "OficialLogistica", "OficialSeguridad", "ModeradorConsenso")
        ronda1: Dict[str, List[str]] = {r: score_rol(r, 1, {}) for r in roles}
        propuestas_r1 = {r: ord_r[0] for r, ord_r in ronda1.items()}

        # --- Ronda 2: Crítica y revisión ---
        ronda2: Dict[str, List[str]] = {r: score_rol(r, 2, propuestas_r1) for r in roles}
        votos_primarios: Dict[str, int] = {}
        for ord_r in ronda2.values():
            votos_primarios[ord_r[0]] = votos_primarios.get(ord_r[0], 0) + 1

        # Matriz de comparación por pares (Copeland) sobre rankings revisados
        matriz_pares: Dict[str, Dict[str, int]] = {
            p1: {p2: 0 for p2 in POLITICAS_ORDEN} for p1 in POLITICAS_ORDEN
        }
        for ord_r in ronda2.values():
            pos_rank = {p: idx for idx, p in enumerate(ord_r)}
            for p1 in POLITICAS_ORDEN:
                for p2 in POLITICAS_ORDEN:
                    if p1 != p2 and pos_rank[p1] < pos_rank[p2]:
                        matriz_pares[p1][p2] += 1

        copeland: Dict[str, float] = {p: 0.0 for p in POLITICAS_ORDEN}
        for p1 in POLITICAS_ORDEN:
            for p2 in POLITICAS_ORDEN:
                if p1 == p2:
                    continue
                if matriz_pares[p1][p2] > matriz_pares[p2][p1]:
                    copeland[p1] += 1.0
                elif matriz_pares[p1][p2] == matriz_pares[p2][p1]:
                    copeland[p1] += 0.5

        mejor_primario = max(votos_primarios.items(), key=lambda kv: kv[1])
        hubo_consenso_directo = mejor_primario[1] >= 3
        if hubo_consenso_directo:
            ganadora = mejor_primario[0]
            metodo_resolucion = "Consenso AutoGen (>=3/4)"
        else:
            ganadora = max(
                POLITICAS_ORDEN,
                key=lambda p: (copeland[p], votos_primarios.get(p, 0), -POLITICAS_ORDEN.index(p)),
            )
            metodo_resolucion = "Desempate Copeland"

        nueva_pol = PoliticaGlobal(ganadora)
        cambio = nueva_pol != mem.politica_actual
        if cambio or not self.historial_consenso:
            mem.politica_actual = nueva_pol
            mem.politica_epoch += 1

        self.historial_consenso.append(
            {
                "tick": tick,
                "epoch": mem.politica_epoch,
                "ganadora": ganadora,
                "metodo": metodo_resolucion,
                "evidencia": {
                    "hielo_visto": hielo_visto,
                    "vetas_ricas": vetas_ricas,
                    "restos": restos_visto,
                    "bajas": bajas,
                    "tormentas": tormentas,
                    "bat_prom": round(bat_prom, 1),
                    "fase_final": percepcion.fase_final,
                },
                "ronda1_top": propuestas_r1,
                "ronda2_rankings": ronda2,
                "copeland": copeland,
            }
        )

        # Propagar cambio de política vía FIPA-ACL a representantes clave (fan-out acotado)
        if cambio:
            destinos = [
                f"{self.letra}_expl_0",
                f"{self.letra}_expl_1",
                f"{self.letra}_rover_1",
                f"{self.letra}_rec_0",
            ]
            for d in destinos:
                if d != unidad.id:
                    self._enviar_acl(
                        api,
                        mem,
                        d,
                        "INFORM",
                        "governance",
                        f"gov_policy_{mem.politica_epoch}",
                        {"politica": ganadora, "epoch": mem.politica_epoch},
                        asunto_legado="gov_politica",
                    )

    def _pesos_dinamicos(self, mem: MemoriaAgente) -> Tuple[float, float, float, float]:
        """Retorna (w1, w2, w3, w4) modulados por la política global vigente."""
        w1 = self.params["w1"]
        w2 = self.params["w2"]
        w3 = self.params["w3"]
        w4 = self.params["w4"]

        pol = mem.politica_actual
        if pol == PoliticaGlobal.EXPANSION_AGRESIVA:
            return (w1 * 1.30, w2 * 0.85, w3 * 0.90, w4 * 0.90)
        if pol == PoliticaGlobal.AHORRO_ENERGETICO:
            return (w1 * 0.95, w2 * 1.45, w3 * 1.05, w4 * 1.40)
        if pol == PoliticaGlobal.SUPERVIVENCIA_CARRONA:
            return (w1 * 0.85, w2 * 1.15, w3 * 1.95, w4 * 1.65)
        # REPLIEGUE_FINAL
        return (w1 * 0.40, w2 * 2.00, w3 * 0.50, w4 * 2.20)

    # -------------------------------------------------------------------------
    # Sesión 2: Utilidad Multi-Atributo U(s, a) y Navegación A* Acotada
    # -------------------------------------------------------------------------
    def _riesgo_tormenta_celda(
        self, pos: Tuple[int, int], mem: MemoriaAgente, tick: int, pasos_futuros: int = 1
    ) -> float:
        riesgo = 0.0
        px, py = pos
        for t in mem.tormentas:
            if tick - t.tick_obs > TTL_TORMENTA:
                continue
            tx = t.fx + t.vx * min(3, pasos_futuros)
            ty = t.fy + t.vy * min(3, pasos_futuros)
            d = math.hypot(px - tx, py - ty)
            if d <= t.radio + 0.7:
                riesgo += 1.0 + (t.radio + 0.7 - d) * 0.35
        return riesgo

    def _riesgo_tormenta_segmento(
        self, origen: Tuple[int, int], destino: Tuple[int, int], mem: MemoriaAgente, tick: int
    ) -> float:
        if not mem.tormentas:
            return 0.0
        pasos = max(1, distancia_chebyshev(origen, destino))
        muestras = min(5, pasos)
        riesgo = 0.0
        for i in range(1, muestras + 1):
            alpha = i / muestras
            mx = int(round(origen[0] + (destino[0] - origen[0]) * alpha))
            my = int(round(origen[1] + (destino[1] - origen[1]) * alpha))
            riesgo += self._riesgo_tormenta_celda((mx, my), mem, tick, pasos_futuros=i)
        return riesgo

    def _utilidad_hielo(
        self,
        unidad: UnidadVista,
        percepcion: Percepcion,
        mem: MemoriaAgente,
        creencia: CreenciaHielo,
        es_encadenado: bool = False,
    ) -> float:
        w1, w2, _, w4 = self._pesos_dinamicos(mem)
        cap_libre = max(0, unidad.capacidad - unidad.carga)
        if cap_libre <= 0:
            return -999.0

        conf = creencia.confianza(percepcion.tick, self.params["decay_lambda"])
        extraible = min(creencia.cantidad, cap_libre)
        g_hielo = extraible * conf * (self.params["chain_bonus"] if es_encadenado else 1.0)

        d_ida = distancia_chebyshev(unidad.pos, creencia.pos)
        d_retorno = max(0, distancia_chebyshev(creencia.pos, percepcion.base_pos) - 2)
        c_despl = d_ida + (0.35 * d_retorno if not es_encadenado else 0.15 * d_retorno)

        r_torm = self._riesgo_tormenta_segmento(unidad.pos, creencia.pos, mem, percepcion.tick)
        return w1 * g_hielo - w2 * c_despl - w4 * r_torm

    def _utilidad_carrona(
        self,
        unidad: UnidadVista,
        percepcion: Percepcion,
        mem: MemoriaAgente,
        creencia: CreenciaRestos,
    ) -> float:
        _, w2, w3, w4 = self._pesos_dinamicos(mem)
        d_ida = distancia_chebyshev(unidad.pos, creencia.pos)
        necesidad_propia = max(0.0, (82.0 - unidad.bateria) / 25.0)
        o_carrona = (creencia.combustible / 10.0) * (0.4 + necesidad_propia)
        r_torm = self._riesgo_tormenta_segmento(unidad.pos, creencia.pos, mem, percepcion.tick)
        return w3 * o_carrona - w2 * d_ida - w4 * r_torm

    def _punto_perimetro_base(self, pos: Tuple[int, int], base_pos: Tuple[int, int]) -> Tuple[int, int]:
        """Calcula el punto más cercano a `pos` con distancia Chebyshev == 2 de `base_pos`."""
        w, h = self.grid_size
        bx, by = base_pos
        tx = int(clamp(pos[0], bx - 2, bx + 2))
        ty = int(clamp(pos[1], by - 2, by + 2))
        tx = int(clamp(tx, 0, w - 1))
        ty = int(clamp(ty, 0, h - 1))
        return (tx, ty)

    def _paso_seguro(
        self,
        origen: Tuple[int, int],
        destino: Tuple[int, int],
        mem: MemoriaAgente,
        tick: int,
    ) -> Tuple[int, int]:
        """
        Navegación segura híbrida:
          - Vía rápida O(1) cuando no hay tormentas próximas al corredor.
          - A* ponderado por campo de tormentas acotado a `MAX_EXPANSIONES_ASTAR`.
        """
        if origen == destino:
            return (0, 0)

        w, h = self.grid_size
        # Vía rápida: evaluar si alguna tormenta conocida está cerca de origen o del paso directo
        dx_dir = 0 if destino[0] == origen[0] else (1 if destino[0] > origen[0] else -1)
        dy_dir = 0 if destino[1] == origen[1] else (1 if destino[1] > origen[1] else -1)
        paso_recto = (origen[0] + dx_dir, origen[1] + dy_dir)

        cerca_tormenta = False
        for t in mem.tormentas:
            if tick - t.tick_obs <= TTL_TORMENTA:
                if (
                    math.hypot(origen[0] - t.fx, origen[1] - t.fy) <= t.radio + 4.0
                    or math.hypot(paso_recto[0] - t.fx, paso_recto[1] - t.fy) <= t.radio + 3.0
                ):
                    cerca_tormenta = True
                    break

        if not cerca_tormenta:
            return (dx_dir, dy_dir)

        # A* 8-direccional acotado para rodear el radio de tormenta
        penal_torm = self.params["storm_penalty"]
        abierto: List[Tuple[float, float, Tuple[int, int]]] = []
        heapq.heappush(abierto, (float(distancia_chebyshev(origen, destino)), 0.0, origen))
        g_cost: Dict[Tuple[int, int], float] = {origen: 0.0}
        vino_de: Dict[Tuple[int, int], Tuple[int, int]] = {}
        cerrados: Set[Tuple[int, int]] = set()
        mejor_nodo = origen
        mejor_h = float(distancia_chebyshev(origen, destino))

        expansiones = 0
        while abierto and expansiones < MAX_EXPANSIONES_ASTAR:
            _, g_actual, actual = heapq.heappop(abierto)
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
                                paso_c += penal_torm + (t.radio + 1.0 - d_t) * 2.0

                    ng = g_actual + paso_c
                    if ng < g_cost.get(nxt, float("inf")):
                        g_cost[nxt] = ng
                        vino_de[nxt] = actual
                        f = ng + float(distancia_chebyshev(nxt, destino))
                        heapq.heappush(abierto, (f, ng, nxt))

        if mejor_nodo != origen and mejor_nodo in vino_de:
            curr = mejor_nodo
            while vino_de.get(curr) != origen and curr in vino_de:
                curr = vino_de[curr]
            if vino_de.get(curr) == origen:
                return (curr[0] - origen[0], curr[1] - origen[1])

        # Fallback por gradiente seguro
        mejor_paso = (dx_dir, dy_dir)
        mejor_score = float("inf")
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = origen[0] + dx, origen[1] + dy
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                sc = math.hypot(destino[0] - nx, destino[1] - ny)
                for t in mem.tormentas:
                    d_t = math.hypot(nx - t.fx, ny - t.fy)
                    if d_t <= t.radio + 0.8:
                        sc += penal_torm + (t.radio + 1.0 - d_t) * 2.0
                if sc < mejor_score:
                    mejor_score = sc
                    mejor_paso = (dx, dy)
        return mejor_paso

    def _evadir_tormenta_estatico(
        self, pos: Tuple[int, int], mem: MemoriaAgente, tick: int, ancla: Optional[Tuple[int, int]] = None
    ) -> Optional[Tuple[int, int]]:
        """Si la unidad está en riesgo de pulso de tormenta, da 1 paso hacia afuera."""
        en_peligro = any(
            math.hypot(pos[0] - t.fx, pos[1] - t.fy) <= t.radio + 0.4
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
                sc = d_min * 4.0
                if ancla is not None:
                    sc -= distancia_chebyshev((nx, ny), ancla) * 0.35
                if sc > mejor_score:
                    mejor_score = sc
                    mejor_paso = (dx, dy)
        return mejor_paso

    # -------------------------------------------------------------------------
    # Mecánicas obligatorias y lógica diferenciada por rol
    # -------------------------------------------------------------------------
    def _manejar_bateria_baja_p(
        self, unidad: UnidadVista, percepcion: Percepcion, api: Any, mem: MemoriaAgente
    ) -> Optional[Tuple[Any, ...]]:
        tick = percepcion.tick

        if unidad.tipo == "rover":
            if unidad.estado == "esperando_recarga" or unidad.bateria <= UMBRAL_BATERIA_BAJA:
                self._transicionar_fsm(mem, EstadoRover.ESPERANDO_RECARGA)
                # Reintento / solicitud estructurada FIPA-ACL
                if (
                    not mem.esperando_agree_rescate
                    and tick - mem.ultimo_ping_recarga >= 24
                ):
                    mem.ultimo_ping_recarga = tick
                    mem.esperando_agree_rescate = True
                    conv_id = f"rescue_{unidad.id}_{tick}"
                    # Intentar CNP de rescate si conocemos >=2 recargadores cercanos; sino REQUEST directo
                    recargadores_conocidos = [
                        uid for uid in mem.contactos_recargadores.keys()
                    ]
                    lanzado_cnp = False
                    if not self.disable_cnp and len(recargadores_conocidos) >= 2:
                        lanzado_cnp = self._iniciar_subasta_cnp(
                            unidad,
                            percepcion,
                            api,
                            mem,
                            "rescate",
                            ("rescate", unidad.id),
                            {"rover_id": unidad.id, "pos": unidad.pos},
                        )
                    if not lanzado_cnp:
                        self._enviar_acl_por_helper(
                            api,
                            mem,
                            "recargador",
                            "REQUEST",
                            "rescue-ops",
                            conv_id,
                            {"rover_id": unidad.id, "pos": unidad.pos},
                            asunto_legado="solicitud_recarga",
                        )
                if unidad.estado != "esperando_recarga":
                    return ("solicitar_recarga",)
                return ("esperar",)
            return None

        # Drones (explorador y recargador)
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
        self, unidad: UnidadVista, percepcion: Percepcion, api: Any, mem: MemoriaAgente
    ) -> Optional[Tuple[Any, ...]]:
        tick = percepcion.tick

        # 1. Dron reporta vetas ricas y restos nuevos respetando filtros anti-saturación
        if unidad.tipo in ("dron_explorador", "dron_recarga"):
            umbral_hielo = self._umbral_min_hielo_reporte(mem)
            if percepcion.hielo_visible and not percepcion.fase_final:
                candidatos = sorted(
                    [h for h in percepcion.hielo_visible if h[2] >= umbral_hielo],
                    key=lambda h: (-h[2], distancia_chebyshev(unidad.pos, (h[0], h[1]))),
                )
                for hx, hy, cant in candidatos:
                    pos = (hx, hy)
                    if (
                        not self.unthrottled_comms
                        and cant < 3
                        and distancia_chebyshev(pos, percepcion.base_pos) > 28
                    ):
                        continue
                    # Evitar saturar la banda de comunicación re-reportando la misma veta
                    cd_hielo = int(round(self.params["cooldown_hielo"]))
                    cnp_min = int(round(self.params["cnp_min_cant"]))
                    if self._pasa_filtro_antisaturacion(mem, ("hielo", pos), tick, cooldown=cd_hielo):
                        if not self.disable_cnp and cant >= cnp_min:
                            self._iniciar_subasta_cnp(
                                unidad,
                                percepcion,
                                api,
                                mem,
                                "hielo",
                                ("hielo", pos),
                                {"pos": pos, "cantidad": cant},
                            )
                        else:
                            self._enviar_acl_por_helper(
                                api,
                                mem,
                                "rover",
                                "INFORM",
                                "resource-discovery",
                                f"ice_{hx}_{hy}_{tick}",
                                {"pos": pos, "cantidad": cant},
                                asunto_legado="hielo",
                                pos_ref=pos,
                            )
                        break  # Máximo 1 reporte de hielo por tick

            if percepcion.restos_visible and not percepcion.fase_final:
                restos = max(percepcion.restos_visible, key=lambda r: r.get("combustible", 10))
                rid = restos["id"]
                rpos = tuple(restos["pos"])
                comb = int(restos.get("combustible", 10))
                valioso_restos = self.unthrottled_comms or (
                    comb >= 20 and distancia_chebyshev(rpos, percepcion.base_pos) <= 30
                )
                if (
                    valioso_restos
                    and self._pasa_filtro_antisaturacion(mem, ("restos", rid), tick, cooldown=300)
                ):
                    if not self.disable_cnp:
                        self._iniciar_subasta_cnp(
                            unidad,
                            percepcion,
                            api,
                            mem,
                            "restos",
                            ("restos", rid),
                            {
                                "id": rid,
                                "pos": rpos,
                                "combustible": restos.get("combustible", 10),
                                "equipo": restos.get("equipo", self.letra),
                            },
                        )
                    else:
                        self._enviar_acl_por_helper(
                            api,
                            mem,
                            "rover",
                            "INFORM",
                            "resource-discovery",
                            f"restos_{rid}_{tick}",
                            {
                                "id": rid,
                                "pos": rpos,
                                "combustible": restos.get("combustible", 10),
                                "equipo": restos.get("equipo", self.letra),
                            },
                            asunto_legado="restos",
                            pos_ref=rpos,
                        )
            return None

        # 2. Rover ve restos cercanos: decidir reporte selectivo y consumo/transferencia
        if not percepcion.restos_visible:
            return None

        restos = max(percepcion.restos_visible, key=lambda r: r.get("combustible", 10))
        rid = restos["id"]
        rpos = tuple(restos["pos"])

        # Buscar si hay aliado necesitado cerca (transferencia)
        aliado_necesitado = None
        for al in percepcion.aliados_visible:
            if (
                al["id"] != unidad.id
                and (al["estado"] == "esperando_recarga" or al["bateria"] < 60.0)
                and distancia_chebyshev(unidad.pos, tuple(al["pos"])) <= 7
            ):
                aliado_necesitado = al
                break

        consumo_util = (
            unidad.bateria < 78.0
            or aliado_necesitado is not None
            or (unidad.carga == 0 and mem.politica_actual == PoliticaGlobal.SUPERVIVENCIA_CARRONA)
        )

        if consumo_util and distancia_chebyshev(unidad.pos, rpos) <= 1 and not percepcion.fase_final:
            self._transicionar_fsm(mem, EstadoRover.CONSUMIENDO_CARRONA)
            if unidad.bateria >= 80.0 and aliado_necesitado is not None:
                return ("consumir_restos", rid, ("transferir", aliado_necesitado["id"]))
            return ("consumir_restos", rid, "propia")

        if (
            not consumo_util
            and not percepcion.fase_final
            and self._pasa_filtro_antisaturacion(mem, ("restos", rid), tick, cooldown=55)
        ):
            self._enviar_acl_por_helper(
                api,
                mem,
                "cercanos",
                "INFORM",
                "resource-discovery",
                f"restos_{rid}_{tick}",
                {
                    "id": rid,
                    "pos": rpos,
                    "combustible": restos.get("combustible", 10),
                    "equipo": restos.get("equipo", self.letra),
                },
                asunto_legado="restos",
                radio=7,
                max_dest=1,
            )

        return None

    def _decidir_dron_explorador_p(
        self, unidad: UnidadVista, percepcion: Percepcion, mem: MemoriaAgente
    ) -> Tuple[Any, ...]:
        tick = percepcion.tick

        # Fase final: evitar gasto innecesario de batería
        if percepcion.fase_final or mem.politica_actual == PoliticaGlobal.REPLIEGUE_FINAL:
            ev = self._evadir_tormenta_estatico(unidad.pos, mem, tick)
            if ev is not None:
                return ("mover", ev[0], ev[1])
            self._transicionar_fsm(mem, EstadoExplorador.RETORNANDO_BASE)
            return ("esperar",)

        if mem.waypoints is None:
            mem.waypoints = self._generar_waypoints_explorador_p(unidad)

        # Una vez completado el barrido completo del sector, modular cadencia
        # de patrulla para reducir consumo energético de exploradores
        if mem.ciclo_patrulla_completo:
            tick_sb = int(round(self.params["expl_tick_standby"]))
            base_mod = max(2, int(round(self.params["expl_paso_post"])))
            if tick >= tick_sb:
                ev = self._evadir_tormenta_estatico(unidad.pos, mem, tick)
                if ev is not None:
                    return ("mover", ev[0], ev[1])
                return ("esperar",)
            paso_mod = base_mod + 1 if mem.politica_actual == PoliticaGlobal.AHORRO_ENERGETICO else base_mod
            if tick % paso_mod != 0:
                ev = self._evadir_tormenta_estatico(unidad.pos, mem, tick)
                if ev is not None:
                    return ("mover", ev[0], ev[1])
                return ("esperar",)

        self._transicionar_fsm(mem, EstadoExplorador.PATRULLANDO)
        wps = mem.waypoints
        destino = wps[mem.patrulla_idx % len(wps)]
        if distancia_chebyshev(unidad.pos, destino) <= 2:
            mem.patrulla_idx = (mem.patrulla_idx + 1) % len(wps)
            if mem.patrulla_idx == 0:
                mem.ciclo_patrulla_completo = True
            destino = wps[mem.patrulla_idx]

        dx, dy = self._paso_seguro(unidad.pos, destino, mem, tick)
        return ("mover", dx, dy)

    def _decidir_dron_recarga_p(
        self, unidad: UnidadVista, percepcion: Percepcion, api: Any, mem: MemoriaAgente
    ) -> Tuple[Any, ...]:
        tick = percepcion.tick

        # Si gastó su recargador o está en fase final sin misión: volver a base / esperar
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

        # Si tiene rescate asignado
        rescate = mem.rescate_actual
        if rescate is not None:
            self._transicionar_fsm(mem, EstadoRecargador.EN_RESCATE)
            r_id = rescate["rover_id"]
            r_pos = tuple(rescate["pos"])
            mem.ticks_en_rescate += 1

            # Verificar al acercarse si el rover ya fue recargado o murió
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
                        f"timeout_{unidad.id}_{tick}",
                        {"motivo": "timeout"},
                        asunto_legado="rescate_failure",
                    )
                    mem.rescate_actual = None
                else:
                    dx, dy = self._paso_seguro(unidad.pos, r_pos, mem, tick)
                    return ("mover", dx, dy)

        # Sin misión activa: permanecer cerca de base recargado, evadiendo tormenta T1
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

    def _decidir_rover_p(
        self, unidad: UnidadVista, percepcion: Percepcion, api: Any, mem: MemoriaAgente
    ) -> Tuple[Any, ...]:
        tick = percepcion.tick

        # 1. En fase final: retornar de inmediato si lleva carga; si no, esperar a salvo
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

        # 2. Carga llena -> retornar a base
        if unidad.carga >= unidad.capacidad:
            self._transicionar_fsm(mem, EstadoRover.RETORNANDO)
            mem.objetivo_hielo = None
            if percepcion.base_aliada_cerca:
                return ("depositar",)
            dest = self._punto_perimetro_base(unidad.pos, percepcion.base_pos)
            dx, dy = self._paso_seguro(unidad.pos, dest, mem, tick)
            return ("mover", dx, dy)

        # 3. Si está parado sobre una celda con hielo y tiene capacidad -> extraer (0% costo)
        for hx, hy, cant in percepcion.hielo_visible:
            if (hx, hy) == unidad.pos and cant > 0 and unidad.carga < unidad.capacidad:
                self._transicionar_fsm(mem, EstadoRover.RECOLECTANDO)
                # Si al extraer queda más hielo y no cabrá todo, avisar si es veta rica
                if cant - 1 >= 3 and unidad.carga + 1 >= unidad.capacidad:
                    if self._pasa_filtro_antisaturacion(mem, ("hielo_sobrante", (hx, hy)), tick, 55):
                        self._enviar_acl_por_helper(
                            api,
                            mem,
                            "rover",
                            "INFORM",
                            "resource-discovery",
                            f"ice_rem_{hx}_{hy}_{tick}",
                            {"pos": (hx, hy), "cantidad": cant - 1},
                            asunto_legado="hielo",
                            pos_ref=(hx, hy),
                        )
                return ("extraer",)

        # 4. Evaluar mejor objetivo candidato según U(s, a)
        mejor_accion_u = -float("inf")
        mejor_tipo: Optional[str] = None
        mejor_target: Any = None

        # 4a. Candidatos de hielo (visibles + creencias válidas en memoria privada)
        hielo_candidatos: List[CreenciaHielo] = []
        for pos, cr in mem.hielo.items():
            if ("hielo", pos) in mem.reservas_ajenas:
                continue
            # Evitar duplicación: si otro rover aliado activo está más cerca de una veta pequeña
            duplicado = False
            if cr.cantidad <= 2 and pos != mem.objetivo_hielo:
                mi_d = distancia_chebyshev(unidad.pos, pos)
                for al in percepcion.aliados_visible:
                    if (
                        al["tipo"] == "rover"
                        and al["estado"] == "activo"
                        and distancia_chebyshev(tuple(al["pos"]), pos) + 1 < mi_d
                    ):
                        duplicado = True
                        break
            if not duplicado:
                hielo_candidatos.append(cr)

        es_encadenado = unidad.carga > 0
        for cr in hielo_candidatos:
            d_al_objetivo = distancia_chebyshev(unidad.pos, cr.pos)
            d_vuelta = max(0, distancia_chebyshev(cr.pos, percepcion.base_pos) - 2)
            # Asegurar que la batería alcanza para ir + volver sin morir
            if unidad.bateria - (d_al_objetivo + d_vuelta) < UMBRAL_BATERIA_BAJA + 4.0:
                continue
            if es_encadenado and d_al_objetivo > int(round(self.params["chain_radius"])):
                continue
            u_val = self._utilidad_hielo(unidad, percepcion, mem, cr, es_encadenado=es_encadenado)
            if u_val > mejor_accion_u:
                mejor_accion_u = u_val
                mejor_tipo = "hielo"
                mejor_target = cr.pos

        # 4b. Candidatos de carroña (solo si carga == 0 y batería < 78%)
        if unidad.carga == 0 and unidad.bateria < 78.0:
            for rid, cr in mem.restos.items():
                if ("restos", rid) in mem.reservas_ajenas:
                    continue
                u_val = self._utilidad_carrona(unidad, percepcion, mem, cr)
                if u_val > mejor_accion_u:
                    mejor_accion_u = u_val
                    mejor_tipo = "restos"
                    mejor_target = {"id": rid, "pos": cr.pos}

        # 4c. Si lleva carga parcial, comparar utilidad de retornar ya vs seguir buscando
        if unidad.carga > 0:
            d_ret = max(0, distancia_chebyshev(unidad.pos, percepcion.base_pos) - 2)
            # Umbral mínimo para desviarse por más hielo
            if mejor_tipo == "hielo" and mejor_accion_u > 1.0:
                self._transicionar_fsm(mem, EstadoRover.RECOLECTANDO)
                mem.objetivo_hielo = mejor_target
                dx, dy = self._paso_seguro(unidad.pos, mejor_target, mem, tick)
                return ("mover", dx, dy)
            # Retornar a base con la carga acumulada
            self._transicionar_fsm(mem, EstadoRover.RETORNANDO)
            if percepcion.base_aliada_cerca:
                return ("depositar",)
            dest = self._punto_perimetro_base(unidad.pos, percepcion.base_pos)
            dx, dy = self._paso_seguro(unidad.pos, dest, mem, tick)
            return ("mover", dx, dy)

        # 5. Si no lleva carga y tiene buen objetivo en U(s, a)
        if mejor_tipo == "restos" and mejor_target is not None and mejor_accion_u > 0.0:
            self._transicionar_fsm(mem, EstadoRover.CONSUMIENDO_CARRONA)
            r_id = mejor_target["id"]
            r_pos = tuple(mejor_target["pos"])
            if distancia_chebyshev(unidad.pos, r_pos) <= 1:
                return ("consumir_restos", r_id, "propia")
            dx, dy = self._paso_seguro(unidad.pos, r_pos, mem, tick)
            return ("mover", dx, dy)

        if mejor_tipo == "hielo" and mejor_target is not None:
            self._transicionar_fsm(mem, EstadoRover.RECOLECTANDO)
            mem.objetivo_hielo = mejor_target
            dx, dy = self._paso_seguro(unidad.pos, mejor_target, mem, tick)
            return ("mover", dx, dy)

        # 6. Sin objetivo conocido:
        # En fase inicial, distribuir rovers en abanico por el cuadrante propio
        # y zona media; luego esperar para no gastar batería caminando sin objetivo.
        if tick < int(round(self.params["rover_tick_expl"])) and mem.politica_actual != PoliticaGlobal.REPLIEGUE_FINAL:
            self._transicionar_fsm(mem, EstadoRover.EXPLORANDO)
            idx = self._extraer_indice_uid(unidad.id)
            w, h = percepcion.grid_size
            bx, by = percepcion.base_pos
            dir_x = 1 if bx < w // 2 else -1
            dir_y = 1 if by < h // 2 else -1
            radio_r = 9 + (idx % 5) * 5
            ang = ((idx % 12) / 11.0) * (math.pi * 0.48)
            tx = int(clamp(bx + dir_x * math.cos(ang) * radio_r, 3, w - 4))
            ty = int(clamp(by + dir_y * math.sin(ang) * radio_r, 3, h - 4))
            if distancia_chebyshev(unidad.pos, (tx, ty)) > 2:
                dx, dy = self._paso_seguro(unidad.pos, (tx, ty), mem, tick)
                return ("mover", dx, dy)

        ev = self._evadir_tormenta_estatico(unidad.pos, mem, tick)
        if ev is not None:
            return ("mover", ev[0], ev[1])

        self._transicionar_fsm(mem, EstadoRover.OCIOSO)
        return ("esperar",)

    # -------------------------------------------------------------------------
    # Punto de entrada principal e instrumentación de latencia (Sesión 6)
    # -------------------------------------------------------------------------
    def decidir_unidad(
        self, unidad: UnidadVista, percepcion: Percepcion, api: Any
    ) -> Tuple[Any, ...]:
        t0 = time.perf_counter()
        try:
            if unidad.estado == "muerto":
                return ("esperar",)

            mem = self._obtener_memoria(unidad)
            self._registrar_cobertura_eval(unidad, percepcion.tick)

            # Sesión 1: Percepción y actualización de creencias individuales
            self._actualizar_percepcion_y_olvido(unidad, percepcion, mem)

            # Sesión 3 y 4: Recepción de mensajes FIPA-ACL y CNP
            self._procesar_mensajes_acl(unidad, percepcion, api, mem)

            # Sesión 4: Cierre de subastas CNP activas en el iniciador
            if mem.subastas_activas:
                self._adjudicar_subastas_cnp(unidad, percepcion, api, mem)

            # Sesión 5: Telemetría y consenso deliberativo AutoGen acotado
            self._enviar_telemetria_gobernanza(unidad, percepcion, api, mem)
            self._deliberar_autogen_y_propagar(unidad, percepcion, api, mem)

            # Mecánica obligatoria de batería baja (<= 25%)
            accion_bat = self._manejar_bateria_baja_p(unidad, percepcion, api, mem)
            if accion_bat is not None:
                return accion_bat

            # Si está consumiendo carroña, terminar consumo
            if unidad.estado == "consumiendo":
                self._transicionar_fsm(mem, EstadoRover.CONSUMIENDO_CARRONA)
                return ("esperar",)

            # Evaluación de restos y reportes FIPA-ACL/CNP
            accion_restos = self._evaluar_restos_y_reportar(unidad, percepcion, api, mem)
            if accion_restos is not None:
                return accion_restos

            # Despacho por rol
            if unidad.tipo == "dron_explorador":
                return self._decidir_dron_explorador_p(unidad, percepcion, mem)
            if unidad.tipo == "dron_recarga":
                return self._decidir_dron_recarga_p(unidad, percepcion, api, mem)
            return self._decidir_rover_p(unidad, percepcion, api, mem)
        finally:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self.latencias_unidad_ms.append(dt_ms)
            t_cur = percepcion.tick
            self.latencias_turno_ms[t_cur] = self.latencias_turno_ms.get(t_cur, 0.0) + dt_ms

    # -------------------------------------------------------------------------
    # Sesión 6: Reporte de Profiling y Métricas Internas
    # -------------------------------------------------------------------------
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
            "consensos": len(self.historial_consenso),
            "transiciones_fsm": len(self.transiciones_fsm),
        }
