"""
equipoC.py
==========
Inteligencia autónoma de COLMENA para la Simulación Multiagente Competitiva
en Marte (60x45), heredando estrictamente de `EquipoBase` (`equipo_base.py`).

Concepto Core — Economía de Retorno Marginal por Anillos Concéntricos (ΔRatio):
  1. Percepción y Memoria Espacial Privada (`MemoriaColmena`):
     Creencias locales por unidad bajo niebla de guerra con decaimiento
     exponencial, invalidación activa y clasificación en anillos concéntricos
     de Chebyshev (R1: <=18, R2: 19..29, R3: 30..42) respecto a `base_pos`.
     Exploración anular por relevos: los exploradores barren solo el anillo
     vigente y entran en espera segura hasta que el consejo abre el siguiente.
  2. Utilidad Multi-Atributo ΔRatio, FSM por Rol y Navegación A*:
     Evaluación con compuerta de retorno marginal (hielo extraíble / costo de
     ruta ida+vuelta+tormenta >= tau_marg), ejecución secuencial de lotes
     encadenados (`ruta_lote`), aprovechamiento de carroña y A* 8-direccional
     acotado a 24 expansiones sobre trayectorias predichas de tormentas.
  3. Comunicación Estructurada FIPA-ACL por Lotes (Batch Digest):
     Explota el costo plano de -0.5% por mensaje agrupando de 2 a 4 vetas
     cercanas (+ trazas de tormenta) en un único boletín comprimido, con
     tablas de cooldown, deduplicación y direccionamiento selectivo.
  4. Subasta Inversa de Vickrey por Lotes (`Bundle CFP`) y Triaje de Rescates:
     Licitación descentralizada de micro-clústeres de vetas donde los rovers
     pujan verazmente el costo energético de su ruta encadenada (gana el menor
     costo y se registra el 2.º menor costo Vickrey) + triaje de recargadores
     desde base priorizando rovers con mayor carga en riesgo.
  5. Elección Social (Condorcet / Copeland con Pesos Epistémicos por Crisis):
     Cuatro representantes (`ExploradorAnular`, `EconomistaMarginal`,
     `OficialTriaje`, `ModeradorColmena`) votan periódicamente la apertura de
     anillos (`ANILLO_INTERNO`, `EXPANSION_MEDIA`, `EXPANSION_EXTERIOR`,
     `SELLADO_RATIO`), rompiendo empates/ciclos mediante Copeland ponderado.
  6. Arquitectura No Bloqueante y Profiling (< 40 ms/turno):
     Estructuras O(1), evaluación perezosa e instrumentación `perf_counter()`.

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
)

RADIO_VISION: Dict[str, int] = {
    "rover": 5,
    "dron_explorador": 9,
    "dron_recarga": 7,
}

POLITICAS_ORDEN: Tuple[str, ...] = (
    "ANILLO_INTERNO",
    "EXPANSION_MEDIA",
    "EXPANSION_EXTERIOR",
    "SELLADO_RATIO",
)

TTL_HIELO = 90
TTL_RESTOS = 70
TTL_TORMENTA = 14
TTL_CONTACTO = 95
TTL_RESERVA_VICKREY = 34
MAX_EXPANSIONES_ASTAR = 24
DECAY_LAMBDA = 0.015


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


class PoliticaAnular(str, Enum):
    ANILLO_INTERNO = "ANILLO_INTERNO"
    EXPANSION_MEDIA = "EXPANSION_MEDIA"
    EXPANSION_EXTERIOR = "EXPANSION_EXTERIOR"
    SELLADO_RATIO = "SELLADO_RATIO"


# -----------------------------------------------------------------------------
# Sesión 1: Estructuras privadas de creencias espaciales y subastas
# -----------------------------------------------------------------------------
@dataclass
class CreenciaHielo:
    pos: Tuple[int, int]
    cantidad: int
    tick_obs: int
    anillo: int
    origen: str

    def confianza(self, tick_actual: int, lam: float = DECAY_LAMBDA) -> float:
        return math.exp(-lam * max(0, tick_actual - self.tick_obs))


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
class SubastaVickrey:
    conversation_id: str
    tipo: str  # "lote_hielo" | "rescate"
    objetivo_keys: List[Any]
    datos: Dict[str, Any]
    tick_inicio: int
    tick_cierre: int
    candidatos_enviados: List[str] = field(default_factory=list)
    propuestas: List[Dict[str, Any]] = field(default_factory=list)
    cerrada: bool = False


@dataclass
class MemoriaColmena:
    """Memoria espacial y de protocolo estrictamente privada de una unidad."""

    uid: str
    tipo: str
    estado_fsm: str

    hielo: Dict[Tuple[int, int], CreenciaHielo] = field(default_factory=dict)
    restos: Dict[str, CreenciaRestos] = field(default_factory=dict)
    tormentas: List[TrazaTormenta] = field(default_factory=list)
    celdas_visitadas: Dict[Tuple[int, int], int] = field(default_factory=dict)

    contactos_rovers: Dict[str, Tuple[Tuple[int, int], float, str, int]] = field(
        default_factory=dict
    )
    contactos_recargadores: Dict[str, Tuple[Tuple[int, int], float, str, int]] = field(
        default_factory=dict
    )

    anillo_waypoints: int = 0
    waypoints: Optional[List[Tuple[int, int]]] = None
    patrulla_idx: int = 0
    ciclo_anillo_completo: bool = False

    ruta_lote: List[Tuple[int, int]] = field(default_factory=list)
    objetivo_hielo: Optional[Tuple[int, int]] = None
    objetivo_restos: Optional[Dict[str, Any]] = None
    rescate_actual: Optional[Dict[str, Any]] = None
    ticks_en_rescate: int = 0

    buffer_lote: Dict[Tuple[int, int], Tuple[int, int]] = field(default_factory=dict)
    cooldowns: Dict[Any, int] = field(default_factory=dict)
    dedup_reportes: Dict[Any, int] = field(default_factory=dict)
    ultimo_ping_recarga: int = -999
    esperando_agree_rescate: bool = False

    subastas_activas: Dict[str, SubastaVickrey] = field(default_factory=dict)
    reservas_ajenas: Dict[Any, int] = field(default_factory=dict)
    rovers_ocupados_hasta: Dict[str, int] = field(default_factory=dict)

    politica_actual: PoliticaAnular = PoliticaAnular.ANILLO_INTERNO
    politica_epoch: int = 0
    ultimo_tick_gov_envio: int = -999
    telemetria_gov_recibida: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    bajas_conocidas: Set[str] = field(default_factory=set)


class EquipoC(EquipoBase):
    """
    Facción COLMENA: limpieza en anillos concéntricos guiada por retorno
    marginal (ΔRatio), boletines FIPA-ACL por lotes, subasta inversa de
    Vickrey por lotes y elección social Condorcet/Copeland ponderada por crisis.
    """

    nombre: str = "COLMENA"

    def __init__(self) -> None:
        super().__init__()
        self._agentes: Dict[str, MemoriaColmena] = {}
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
            "lotes_emitidos": 0,
            "vetas_empaquetadas": 0,
            "por_performativa": {},
            "por_ontologia": {},
        }
        self.stats_vickrey: Dict[str, int] = {
            "cfp_lote_enviados": 0,
            "cfp_rescate_enviados": 0,
            "propose_recibidos": 0,
            "accept_enviados": 0,
            "reject_enviados": 0,
            "duplicados_evitados": 0,
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
            "lotes_emitidos": 0,
            "vetas_empaquetadas": 0,
            "por_performativa": {},
            "por_ontologia": {},
        }
        self.stats_vickrey = {
            "cfp_lote_enviados": 0,
            "cfp_rescate_enviados": 0,
            "propose_recibidos": 0,
            "accept_enviados": 0,
            "reject_enviados": 0,
            "duplicados_evitados": 0,
        }
        self.transiciones_fsm.clear()
        self.historial_votaciones.clear()

    # -------------------------------------------------------------------------
    # Utilidades de ciclo de vida y geometría anular
    # -------------------------------------------------------------------------
    @staticmethod
    def _extraer_indice_uid(uid: str) -> int:
        try:
            return int(uid.split("_")[-1])
        except Exception:
            return abs(hash(uid)) % 100

    def _clasificar_anillo(self, pos: Tuple[int, int]) -> int:
        d = distancia_chebyshev(pos, self.base_pos)
        if d <= 21:
            return 1
        if d <= 31:
            return 2
        return 3

    def _limites_politica(self, mem: MemoriaColmena) -> Tuple[int, int, float, float]:
        """Retorna (anillo_max, radio_max, tau_marginal, alpha_retorno)."""
        pol = mem.politica_actual
        if pol == PoliticaAnular.ANILLO_INTERNO:
            return (1, 24, 0.050, 0.30)
        if pol == PoliticaAnular.EXPANSION_MEDIA:
            return (2, 35, 0.042, 0.32)
        if pol == PoliticaAnular.EXPANSION_EXTERIOR:
            return (3, 46, 0.036, 0.35)
        return (1, 26, 0.070, 0.40)

    def _obtener_memoria(self, unidad: UnidadVista) -> MemoriaColmena:
        if unidad.id not in self._agentes:
            if unidad.tipo == "rover":
                estado_ini = EstadoRover.OCIOSO.value
            elif unidad.tipo == "dron_explorador":
                estado_ini = EstadoExplorador.PATRULLANDO.value
            else:
                estado_ini = EstadoRecargador.EN_ESPERA.value

            self._agentes[unidad.id] = MemoriaColmena(
                uid=unidad.id,
                tipo=unidad.tipo,
                estado_fsm=estado_ini,
            )
            lista = self._ids_por_rol.get(unidad.tipo)
            if lista is not None and unidad.id not in lista:
                lista.append(unidad.id)
                lista.sort(key=self._extraer_indice_uid)
        return self._agentes[unidad.id]

    def _transicionar_fsm(self, mem: MemoriaColmena, nuevo_estado: Enum) -> None:
        val = nuevo_estado.value
        if mem.estado_fsm != val:
            clave = f"{mem.tipo}:{mem.estado_fsm}->{val}"
            self.transiciones_fsm[clave] = self.transiciones_fsm.get(clave, 0) + 1
            mem.estado_fsm = val

    # -------------------------------------------------------------------------
    # Sesión 1: Percepción privada, olvido exponencial y barrido por anillos
    # -------------------------------------------------------------------------
    def _actualizar_percepcion_y_olvido(
        self, unidad: UnidadVista, percepcion: Percepcion, mem: MemoriaColmena
    ) -> None:
        tick = percepcion.tick
        rv = RADIO_VISION.get(unidad.tipo, 5)
        ux, uy = unidad.pos
        mem.celdas_visitadas[unidad.pos] = tick

        hielo_vis: Set[Tuple[int, int]] = set()
        for hx, hy, cant in percepcion.hielo_visible:
            if cant > 0:
                pos = (hx, hy)
                hielo_vis.add(pos)
                mem.hielo[pos] = CreenciaHielo(
                    pos=pos,
                    cantidad=cant,
                    tick_obs=tick,
                    anillo=self._clasificar_anillo(pos),
                    origen="local",
                )

        a_borrar_hielo: List[Tuple[int, int]] = []
        for pos, cr in mem.hielo.items():
            if max(abs(pos[0] - ux), abs(pos[1] - uy)) <= rv and pos not in hielo_vis:
                a_borrar_hielo.append(pos)
            elif tick - cr.tick_obs > TTL_HIELO or cr.confianza(tick) < 0.14:
                a_borrar_hielo.append(pos)
        for pos in a_borrar_hielo:
            mem.hielo.pop(pos, None)
            mem.buffer_lote.pop(pos, None)
            if mem.objetivo_hielo == pos:
                mem.objetivo_hielo = None
            if pos in mem.ruta_lote:
                mem.ruta_lote = [p for p in mem.ruta_lote if p != pos]

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

        a_borrar_restos = [
            rid
            for rid, cr in mem.restos.items()
            if (max(abs(cr.pos[0] - ux), abs(cr.pos[1] - uy)) <= rv and rid not in restos_vis)
            or (tick - cr.tick_obs > TTL_RESTOS)
        ]
        for rid in a_borrar_restos:
            mem.restos.pop(rid, None)
            if mem.objetivo_restos and mem.objetivo_restos.get("id") == rid:
                mem.objetivo_restos = None

        # Actualizar trazas de tormentas con estimación de velocidad (vx, vy)
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
        mem.rovers_ocupados_hasta = {
            k: exp for k, exp in mem.rovers_ocupados_hasta.items() if exp >= tick
        }

        if percepcion.fase_final and mem.politica_actual != PoliticaAnular.SELLADO_RATIO:
            mem.politica_actual = PoliticaAnular.SELLADO_RATIO
            mem.politica_epoch = max(mem.politica_epoch, 999)

    def _generar_waypoints_anillo(
        self, unidad: UnidadVista, anillo: int
    ) -> List[Tuple[int, int]]:
        """
        Genera 3 waypoints compactos sobre el arco del anillo concéntrico vigente
        adaptados dinámicamente a cualquier esquina `self.base_pos` y flota.
        """
        w, h = self.grid_size
        bx, by = self.base_pos
        dir_x = 1.0 if bx < w / 2.0 else -1.0
        dir_y = 1.0 if by < h / 2.0 else -1.0

        idx = self._extraer_indice_uid(unidad.id)
        total_expl = max(15, len(self._ids_por_rol["dron_explorador"]))
        slot = (idx % total_expl) / max(1, total_expl - 1)

        if anillo == 1:
            r_in, r_out = 10.0, 17.5
            ang0 = 0.06 + slot * (math.pi * 0.44)
            ang1 = clamp(ang0 + (0.14 if idx % 2 == 0 else -0.14), 0.05, math.pi * 0.47)
        elif anillo == 2:
            r_in, r_out = 23.0, 29.0
            ang0 = 0.06 + slot * (math.pi * 0.44)
            ang1 = clamp(ang0 + (0.12 if idx % 2 == 0 else -0.12), 0.05, math.pi * 0.47)
        else:
            r_in, r_out = 30.0, 36.0
            ang0 = 0.08 + slot * (math.pi * 0.40)
            ang1 = clamp(ang0 + (0.10 if idx % 2 == 0 else -0.10), 0.06, math.pi * 0.45)

        def pt(rad: float, ang: float) -> Tuple[int, int]:
            px = int(clamp(bx + dir_x * math.cos(ang) * rad, 4, w - 5))
            py = int(clamp(by + dir_y * math.sin(ang) * rad, 4, h - 5))
            return (px, py)

        p1 = pt(r_in, ang0)
        p2 = pt(r_out, ang1)
        p3 = pt((r_in + r_out) * 0.5, (ang0 + ang1) * 0.5)
        wps = [p1, p2, p3]
        wps.sort(key=lambda p: distancia_chebyshev(unidad.pos, p))
        return wps

    # -------------------------------------------------------------------------
    # Sesión 3: Comunicación Estructurada FIPA-ACL y Empaquetado por Lotes
    # -------------------------------------------------------------------------
    def _empaquetar_acl(
        self,
        mem: MemoriaColmena,
        performative: str,
        ontology: str,
        conversation_id: str,
        content: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload_content = dict(content)
        payload_content["_gov"] = {
            "politica": mem.politica_actual.value,
            "epoch": mem.politica_epoch,
        }
        return {
            "performative": performative,
            "sender": mem.uid,
            "conversation_id": conversation_id,
            "ontology": ontology,
            "content": payload_content,
            **{k: v for k, v in content.items() if k in ("pos", "cantidad", "id", "rover_id", "lote")},
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
        mem: MemoriaColmena,
        dest_id: str,
        performative: str,
        ontology: str,
        conversation_id: str,
        content: Dict[str, Any],
        asunto_legado: Optional[str] = None,
    ) -> bool:
        datos = self._empaquetar_acl(mem, performative, ontology, conversation_id, content)
        ok = api.enviar_mensaje(dest_id, asunto_legado or f"{ontology}:{performative}", datos)
        if ok:
            self._registrar_envio(performative, ontology, 1)
        return ok

    def _enviar_acl_helper(
        self,
        api: Any,
        mem: MemoriaColmena,
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
        datos = self._empaquetar_acl(mem, performative, ontology, conversation_id, content)
        enviados = 0
        if helper_name == "recargador":
            enviados = 1 if api.enviar_a_recargador_cercano(asunto_legado, datos) else 0
        elif helper_name == "rover":
            enviados = 1 if api.enviar_a_rover_ocioso_cercano(asunto_legado, datos, pos_ref=pos_ref) else 0
        elif helper_name == "cercanos":
            enviados = api.enviar_a_aliados_cercanos(radio, asunto_legado, datos, max_dest=max_dest)
        if enviados > 0:
            self._registrar_envio(performative, ontology, enviados)
        return enviados

    def _pasa_filtro(self, mem: MemoriaColmena, clave: Any, tick: int, cooldown: int) -> bool:
        ultimo = mem.cooldowns.get(clave, -9999)
        if tick - ultimo < cooldown:
            self.stats_comms["bloqueados_filtro"] += 1
            return False
        mem.cooldowns[clave] = tick
        return True

    def _procesar_mensajes_acl(
        self, unidad: UnidadVista, percepcion: Percepcion, api: Any, mem: MemoriaColmena
    ) -> None:
        tick = percepcion.tick
        for msg in percepcion.mensajes:
            emisor = msg.get("emisor_id", "")
            datos = msg.get("datos", {})

            performative = datos.get("performative") or "INFORM"
            ontology = datos.get("ontology") or "ring-discovery"
            conv_id = datos.get("conversation_id", f"col_{tick}")
            content = datos.get("content", datos)

            gov = content.get("_gov") if isinstance(content, dict) else None
            if gov and int(gov.get("epoch", 0)) > mem.politica_epoch:
                try:
                    mem.politica_actual = PoliticaAnular(gov["politica"])
                    mem.politica_epoch = int(gov["epoch"])
                except Exception:
                    pass

            if ontology == "ring-governance" and performative == "INFORM":
                if content.get("tipo") == "telemetria":
                    mem.telemetria_gov_recibida[emisor] = {**content, "tick": tick}
                elif "politica" in content:
                    ep = int(content.get("epoch", 0))
                    if ep > mem.politica_epoch:
                        try:
                            mem.politica_actual = PoliticaAnular(content["politica"])
                            mem.politica_epoch = ep
                        except Exception:
                            pass

            elif ontology == "ring-discovery" and performative == "INFORM":
                if "lote" in content:
                    lote = content["lote"]
                    ruta_nueva: List[Tuple[int, int]] = []
                    for item in lote:
                        pos = tuple(item["pos"])
                        cant = int(item.get("cantidad", 1))
                        mem.hielo[pos] = CreenciaHielo(
                            pos=pos,
                            cantidad=cant,
                            tick_obs=tick,
                            anillo=self._clasificar_anillo(pos),
                            origen="acl",
                        )
                        mem.dedup_reportes[("hielo", pos)] = tick
                        ruta_nueva.append(pos)
                    if (
                        unidad.tipo == "rover"
                        and not mem.ruta_lote
                        and mem.objetivo_hielo is None
                        and unidad.carga < unidad.capacidad
                        and ruta_nueva
                    ):
                        mem.ruta_lote = ruta_nueva
                        mem.objetivo_hielo = ruta_nueva[0]
                elif "pos" in content and "cantidad" in content:
                    pos = tuple(content["pos"])
                    cant = int(content["cantidad"])
                    mem.hielo[pos] = CreenciaHielo(
                        pos=pos,
                        cantidad=cant,
                        tick_obs=tick,
                        anillo=self._clasificar_anillo(pos),
                        origen="acl",
                    )
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

                for tr in content.get("tormentas", []):
                    if len(tr) >= 5:
                        mem.tormentas.append(
                            TrazaTormenta(
                                fx=float(tr[0]),
                                fy=float(tr[1]),
                                vx=float(tr[2]),
                                vy=float(tr[3]),
                                radio=float(tr[4]),
                                tick_obs=tick,
                            )
                        )

            elif ontology == "rescue-triage":
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
                                "rescue-triage",
                                conv_id,
                                {"rover_id": rover_id, "rescatista": unidad.id},
                                asunto_legado="rescate_agree",
                            )
                elif performative == "AGREE" and unidad.tipo == "rover":
                    mem.esperando_agree_rescate = False
                elif performative == "FAILURE" and unidad.tipo == "rover":
                    mem.esperando_agree_rescate = False
                    mem.ultimo_ping_recarga = -999

            elif ontology == "vickrey-bundle":
                self._manejar_mensaje_vickrey(
                    unidad, percepcion, api, mem, emisor, performative, conv_id, content
                )

    # -------------------------------------------------------------------------
    # Sesión 4: Subasta Inversa de Vickrey por Lotes y Triaje de Rescates
    # -------------------------------------------------------------------------
    def _ordenar_ruta_lote(
        self, origen: Tuple[int, int], lote: List[Dict[str, Any]]
    ) -> Tuple[List[Tuple[int, int]], float, int]:
        """Ordena greedy TSP un micro-lote de 1..4 vetas y retorna (ruta, dist_total, hielo_total)."""
        pendientes = [(tuple(it["pos"]), int(it.get("cantidad", 1))) for it in lote]
        actual = origen
        ruta: List[Tuple[int, int]] = []
        dist_total = 0.0
        hielo_total = 0
        while pendientes:
            pendientes.sort(key=lambda it: (distancia_chebyshev(actual, it[0]), -it[1]))
            pos_sig, cant_sig = pendientes.pop(0)
            dist_total += float(distancia_chebyshev(actual, pos_sig))
            ruta.append(pos_sig)
            hielo_total += cant_sig
            actual = pos_sig
        return ruta, dist_total, hielo_total

    def _calcular_costo_puja_vickrey(
        self, unidad: UnidadVista, percepcion: Percepcion, mem: MemoriaColmena, content: Dict[str, Any]
    ) -> Optional[float]:
        tipo_sub = content.get("tipo_sub")

        if tipo_sub == "rescate" and unidad.tipo == "dron_recarga":
            if unidad.carga_recarga <= 0 or unidad.bateria <= UMBRAL_BATERIA_BAJA + 5.0:
                return None
            rpos = tuple(content["pos"])
            carga_rov = int(content.get("carga_rover", 0))
            dist = distancia_chebyshev(unidad.pos, rpos)
            riesgo = self._riesgo_tormenta_segmento(unidad.pos, rpos, mem, percepcion.tick)
            ocupado = 35.0 if mem.rescate_actual is not None else 0.0
            return (
                dist
                + 4.0 * riesgo
                + 0.12 * (100.0 - unidad.bateria)
                - 6.0 * carga_rov
                + ocupado
            )

        if tipo_sub == "lote_hielo" and unidad.tipo == "rover":
            if (
                unidad.estado == "esperando_recarga"
                or unidad.bateria <= UMBRAL_BATERIA_BAJA + 8.0
                or mem.ruta_lote
            ):
                return None
            cap_libre = max(0, unidad.capacidad - unidad.carga)
            if cap_libre <= 0:
                return None
            lote = content.get("lote", [])
            if not lote:
                return None

            ruta, d_ruta, hielo_lote = self._ordenar_ruta_lote(unidad.pos, lote)
            # Si ya tiene objetivo, solo pujar si el lote ofrece mucho más hielo o está más cerca
            if mem.objetivo_hielo is not None and (unidad.carga > 0 or (hielo_lote < 4 and d_ruta > 10)):
                return None
            d_vuelta = max(0, distancia_chebyshev(ruta[-1], percepcion.base_pos) - 2)
            if unidad.bateria - (d_ruta + d_vuelta) < UMBRAL_BATERIA_BAJA + 5.0:
                return None

            riesgo = self._riesgo_tormenta_segmento(unidad.pos, ruta[0], mem, percepcion.tick)
            _, _, tau_marg, alpha_ret = self._limites_politica(mem)
            extraible = min(hielo_lote, cap_libre)
            eta = extraible / max(1.0, d_ruta + alpha_ret * d_vuelta + 4.0 * riesgo)
            if eta < tau_marg * 0.65:
                return None

            return (
                d_ruta
                + 0.35 * d_vuelta
                + 5.0 * riesgo
                + 0.12 * (100.0 - unidad.bateria)
                - 1.8 * extraible
            )

        return None

    def _manejar_mensaje_vickrey(
        self,
        unidad: UnidadVista,
        percepcion: Percepcion,
        api: Any,
        mem: MemoriaColmena,
        emisor: str,
        performative: str,
        conv_id: str,
        content: Dict[str, Any],
    ) -> None:
        tick = percepcion.tick
        if performative == "CFP":
            if unidad.bateria <= 15.0:
                return
            costo = self._calcular_costo_puja_vickrey(unidad, percepcion, mem, content)
            if costo is not None:
                self._enviar_acl(
                    api,
                    mem,
                    emisor,
                    "PROPOSE",
                    "vickrey-bundle",
                    conv_id,
                    {
                        "cost": round(costo, 4),
                        "bateria": round(unidad.bateria, 2),
                        "pos": unidad.pos,
                        "tipo_sub": content.get("tipo_sub"),
                    },
                    asunto_legado="vickrey_propose",
                )

        elif performative == "PROPOSE":
            sub = mem.subastas_activas.get(conv_id)
            if sub is not None and not sub.cerrada:
                self.stats_vickrey["propose_recibidos"] += 1
                sub.propuestas.append(
                    {
                        "postor": emisor,
                        "cost": float(content.get("cost", 1e6)),
                        "bateria": float(content.get("bateria", 0.0)),
                    }
                )

        elif performative == "ACCEPT_PROPOSAL":
            tipo_sub = content.get("tipo_sub")
            if tipo_sub == "lote_hielo" and unidad.tipo == "rover":
                lote = content.get("lote", [])
                if lote:
                    ruta, _, _ = self._ordenar_ruta_lote(unidad.pos, lote)
                    mem.ruta_lote = ruta
                    mem.objetivo_hielo = ruta[0]
                    for item in lote:
                        p = tuple(item["pos"])
                        mem.hielo[p] = CreenciaHielo(
                            pos=p,
                            cantidad=int(item.get("cantidad", 1)),
                            tick_obs=tick,
                            anillo=self._clasificar_anillo(p),
                            origen="vickrey",
                        )
            elif tipo_sub == "rescate" and unidad.tipo == "dron_recarga":
                rpos = tuple(content["pos"])
                mem.rescate_actual = {"rover_id": content["rover_id"], "pos": rpos}
                mem.ticks_en_rescate = 0
                self._transicionar_fsm(mem, EstadoRecargador.EN_RESCATE)
                self._enviar_acl(
                    api,
                    mem,
                    content["rover_id"],
                    "AGREE",
                    "rescue-triage",
                    conv_id,
                    {"rover_id": content["rover_id"], "rescatista": unidad.id},
                    asunto_legado="rescate_agree",
                )

        elif performative == "REJECT_PROPOSAL":
            if content.get("tipo_sub") == "lote_hielo":
                for item in content.get("lote", []):
                    p = tuple(item["pos"])
                    mem.reservas_ajenas[("hielo", p)] = tick + TTL_RESERVA_VICKREY
                    if mem.objetivo_hielo == p:
                        mem.objetivo_hielo = None
                    if p in mem.ruta_lote:
                        mem.ruta_lote = [q for q in mem.ruta_lote if q != p]

    def _iniciar_subasta_vickrey_lote(
        self,
        unidad: UnidadVista,
        percepcion: Percepcion,
        api: Any,
        mem: MemoriaColmena,
        lote: List[Dict[str, Any]],
    ) -> bool:
        """Lanza un único Bundle CFP de Vickrey (2–4 vetas) a K=2 rovers cercanos."""
        if unidad.bateria <= 18.0 or len(mem.subastas_activas) >= 2 or not lote:
            return False

        tick = percepcion.tick
        lote_libre = [
            it
            for it in lote
            if mem.reservas_ajenas.get(("hielo", tuple(it["pos"])), -1) < tick
        ]
        if not lote_libre:
            self.stats_vickrey["duplicados_evitados"] += 1
            return False

        pos_ref = tuple(lote_libre[0]["pos"])
        elegibles = [
            (distancia_chebyshev(pos_ref, info[0]), uid)
            for uid, info in mem.contactos_rovers.items()
            if info[2] == "activo"
            and info[1] > UMBRAL_BATERIA_BAJA + 10.0
            and mem.rovers_ocupados_hasta.get(uid, -1) < tick
            and distancia_chebyshev(pos_ref, info[0]) <= 26
        ]
        elegibles.sort(key=lambda x: x[0])
        candidatos = [uid for _, uid in elegibles[:2]]

        conv_id = f"vk_lote_{unidad.id}_{tick}"
        trazas_torm = [
            [round(t.fx, 1), round(t.fy, 1), round(t.vx, 2), round(t.vy, 2), t.radio]
            for t in mem.tormentas[:3]
        ]

        if candidatos:
            enviados: List[str] = []
            for dest_id in candidatos:
                if self._enviar_acl(
                    api,
                    mem,
                    dest_id,
                    "CFP",
                    "vickrey-bundle",
                    conv_id,
                    {"tipo_sub": "lote_hielo", "lote": lote_libre, "pos": pos_ref},
                    asunto_legado="vickrey_cfp",
                ):
                    enviados.append(dest_id)
                    self.stats_vickrey["cfp_lote_enviados"] += 1
            if enviados:
                keys = [("hielo", tuple(it["pos"])) for it in lote_libre]
                mem.subastas_activas[conv_id] = SubastaVickrey(
                    conversation_id=conv_id,
                    tipo="lote_hielo",
                    objetivo_keys=keys,
                    datos={"lote": lote_libre, "pos": pos_ref, "tormentas": trazas_torm},
                    tick_inicio=tick,
                    tick_cierre=tick + 4,
                    candidatos_enviados=enviados,
                )
                self.stats_comms["lotes_emitidos"] += 1
                self.stats_comms["vetas_empaquetadas"] += len(lote_libre)
                return True

        n = self._enviar_acl_helper(
            api,
            mem,
            "rover",
            "INFORM",
            "ring-discovery",
            conv_id,
            {"lote": lote_libre, "pos": pos_ref, "tormentas": trazas_torm},
            asunto_legado="lote_hielo",
            pos_ref=pos_ref,
        )
        if n > 0:
            self.stats_comms["lotes_emitidos"] += 1
            self.stats_comms["vetas_empaquetadas"] += len(lote_libre)
            for it in lote_libre:
                mem.reservas_ajenas[("hielo", tuple(it["pos"]))] = tick + TTL_RESERVA_VICKREY
        return n > 0

    def _iniciar_subasta_vickrey_rescate(
        self, unidad: UnidadVista, percepcion: Percepcion, api: Any, mem: MemoriaColmena
    ) -> bool:
        """Subasta de triaje entre recargadores cercanos ponderando carga en riesgo."""
        tick = percepcion.tick
        if len(mem.subastas_activas) >= 2:
            return False
        elegibles = [
            (distancia_chebyshev(unidad.pos, info[0]), uid)
            for uid, info in mem.contactos_recargadores.items()
            if info[1] > UMBRAL_BATERIA_BAJA + 10.0
        ]
        elegibles.sort(key=lambda x: x[0])
        candidatos = [uid for _, uid in elegibles[:2]]
        if not candidatos:
            return False

        conv_id = f"vk_resc_{unidad.id}_{tick}"
        enviados: List[str] = []
        for dest_id in candidatos:
            if self._enviar_acl(
                api,
                mem,
                dest_id,
                "CFP",
                "vickrey-bundle",
                conv_id,
                {
                    "tipo_sub": "rescate",
                    "rover_id": unidad.id,
                    "pos": unidad.pos,
                    "carga_rover": unidad.carga,
                },
                asunto_legado="vickrey_rescate_cfp",
            ):
                enviados.append(dest_id)
                self.stats_vickrey["cfp_rescate_enviados"] += 1
        if not enviados:
            return False

        mem.subastas_activas[conv_id] = SubastaVickrey(
            conversation_id=conv_id,
            tipo="rescate",
            objetivo_keys=[("rescate", unidad.id)],
            datos={"rover_id": unidad.id, "pos": unidad.pos, "carga_rover": unidad.carga},
            tick_inicio=tick,
            tick_cierre=tick + 3,
            candidatos_enviados=enviados,
        )
        return True

    def _adjudicar_subastas_vickrey(
        self, unidad: UnidadVista, percepcion: Percepcion, api: Any, mem: MemoriaColmena
    ) -> None:
        """Cierra subastas vencidas: gana menor costo y se registra 2.º menor costo Vickrey."""
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
                if sub.tipo == "lote_hielo" and unidad.bateria > 18.0:
                    self._enviar_acl_helper(
                        api,
                        mem,
                        "rover",
                        "INFORM",
                        "ring-discovery",
                        conv_id,
                        sub.datos,
                        asunto_legado="lote_hielo",
                        pos_ref=tuple(sub.datos["pos"]),
                    )
                continue

            sub.propuestas.sort(key=lambda p: (p["cost"], -p["bateria"], p["postor"]))
            ganador = sub.propuestas[0]["postor"]
            segundo_costo = (
                sub.propuestas[1]["cost"]
                if len(sub.propuestas) >= 2
                else sub.propuestas[0]["cost"] * 1.15
            )

            self._enviar_acl(
                api,
                mem,
                ganador,
                "ACCEPT_PROPOSAL",
                "vickrey-bundle",
                conv_id,
                {"tipo_sub": sub.tipo, "vickrey_price": round(segundo_costo, 4), **sub.datos},
                asunto_legado="vickrey_accept",
            )
            self.stats_vickrey["accept_enviados"] += 1
            if sub.tipo == "lote_hielo":
                mem.rovers_ocupados_hasta[ganador] = tick + 24
            for k in sub.objetivo_keys:
                mem.reservas_ajenas[k] = tick + TTL_RESERVA_VICKREY

            for perdedor in sub.propuestas[1:]:
                self._enviar_acl(
                    api,
                    mem,
                    perdedor["postor"],
                    "REJECT_PROPOSAL",
                    "vickrey-bundle",
                    conv_id,
                    {"tipo_sub": sub.tipo, **sub.datos},
                    asunto_legado="vickrey_reject",
                )
                self.stats_vickrey["reject_enviados"] += 1
                self.stats_vickrey["duplicados_evitados"] += 1

        for cid in cerradas:
            mem.subastas_activas.pop(cid, None)

    # -------------------------------------------------------------------------
    # Sesión 5: Elección Social — Condorcet / Copeland Ponderado por Crisis
    # -------------------------------------------------------------------------
    def _enviar_telemetria_gobernanza(
        self, unidad: UnidadVista, percepcion: Percepcion, api: Any, mem: MemoriaColmena
    ) -> None:
        tick = percepcion.tick
        if tick < 18:
            return
        moderador_id = f"{self.letra}_rover_0"
        if unidad.id == moderador_id:
            return

        rol_map = {
            f"{self.letra}_expl_0": "ExploradorAnular",
            f"{self.letra}_rover_1": "EconomistaMarginal",
            f"{self.letra}_rec_0": "OficialTriaje",
        }
        rol = rol_map.get(unidad.id)
        if rol is None:
            return

        intervalo = 55 if not percepcion.fase_final else 35
        if tick - mem.ultimo_tick_gov_envio < intervalo:
            return
        mem.ultimo_tick_gov_envio = tick

        if rol == "ExploradorAnular":
            r1 = sum(1 for c in mem.hielo.values() if c.anillo == 1)
            r2 = sum(1 for c in mem.hielo.values() if c.anillo == 2)
            r3 = sum(1 for c in mem.hielo.values() if c.anillo == 3)
            datos = {
                "tipo": "telemetria",
                "rol": rol,
                "hielo_r1": r1,
                "hielo_r2": r2,
                "hielo_r3": r3,
                "ciclo_completo": int(mem.ciclo_anillo_completo),
            }
        elif rol == "EconomistaMarginal":
            datos = {
                "tipo": "telemetria",
                "rol": rol,
                "bateria": round(unidad.bateria, 1),
                "hielo_local": len(mem.hielo),
                "rovers_vistos": len(mem.contactos_rovers),
            }
        else:
            datos = {
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
            "ring-governance",
            f"gov_col_{rol}_{tick}",
            datos,
            asunto_legado="gov_telemetria",
        )

    def _votar_condorcet_copeland(
        self, unidad: UnidadVista, percepcion: Percepcion, api: Any, mem: MemoriaColmena
    ) -> None:
        moderador_id = f"{self.letra}_rover_0"
        if unidad.id != moderador_id:
            return

        tick = percepcion.tick
        ultimo = self.historial_votaciones[-1]["tick"] if self.historial_votaciones else -999
        if not percepcion.fase_final and (tick < 20 or tick - ultimo < 26):
            return
        if percepcion.fase_final and mem.politica_actual == PoliticaAnular.SELLADO_RATIO:
            return

        tel_e = mem.telemetria_gov_recibida.get(f"{self.letra}_expl_0", {})
        tel_l = mem.telemetria_gov_recibida.get(f"{self.letra}_rover_1", {})
        tel_t = mem.telemetria_gov_recibida.get(f"{self.letra}_rec_0", {})

        h_r1 = int(tel_e.get("hielo_r1", sum(1 for c in mem.hielo.values() if c.anillo == 1)))
        h_r2 = int(tel_e.get("hielo_r2", sum(1 for c in mem.hielo.values() if c.anillo == 2)))
        h_r3 = int(tel_e.get("hielo_r3", sum(1 for c in mem.hielo.values() if c.anillo == 3)))
        ciclo_e = int(tel_e.get("ciclo_completo", 0))
        bat_prom = (unidad.bateria + float(tel_l.get("bateria", unidad.bateria))) * 0.5
        tormentas = max(len(mem.tormentas), int(tel_t.get("tormentas_vistas", 0)))
        bajas = max(len(mem.bajas_conocidas), int(tel_t.get("bajas_conocidas", 0)))

        crisis_torm = tormentas >= 2 or bajas >= 3
        crisis_escasez = tick >= 44 and h_r1 <= 3
        crisis_energia = bat_prom < 62.0

        pesos: Dict[str, float] = {
            "ExploradorAnular": 2.0 if crisis_escasez else 1.0,
            "EconomistaMarginal": 2.0 if crisis_energia else 1.0,
            "OficialTriaje": 2.0 if crisis_torm else 1.0,
            "ModeradorColmena": 1.2,
        }

        def boleta_rol(rol: str) -> List[str]:
            s: Dict[str, float] = {p: 0.0 for p in POLITICAS_ORDEN}
            if percepcion.fase_final:
                s["SELLADO_RATIO"] += 100.0

            if rol == "ExploradorAnular":
                s["ANILLO_INTERNO"] += 3.5 if (tick < 44 and h_r1 >= 2) else 0.5
                s["EXPANSION_MEDIA"] += 2.9 + (2.7 if (44 <= tick < 180) else 0.0) + h_r2 * 0.25
                s["EXPANSION_EXTERIOR"] += 0.9 + (3.0 if (100 <= tick < 460 and h_r2 <= 2) else 0.0) + h_r3 * 0.2
                s["SELLADO_RATIO"] += 3.8 if (tick >= 500 or (tick >= 360 and (h_r1 + h_r2 + h_r3) == 0)) else 0.2
            elif rol == "EconomistaMarginal":
                s["ANILLO_INTERNO"] += 3.4 if (h_r1 >= 3 and tick < 44) else 0.8
                s["EXPANSION_MEDIA"] += 3.2 if (44 <= tick < 300 and bat_prom >= 58.0) else 1.4
                s["EXPANSION_EXTERIOR"] += 2.5 if (bat_prom >= 64.0 and 100 <= tick < 460) else 0.3
                s["SELLADO_RATIO"] += 3.8 if (bat_prom < 52.0 or tick >= 500) else 0.6
            elif rol == "OficialTriaje":
                s["ANILLO_INTERNO"] += 3.0 + tormentas * 0.4
                s["EXPANSION_MEDIA"] += 2.8 if bajas < 3 else 1.3
                s["EXPANSION_EXTERIOR"] += 2.0 if (tormentas <= 2 and bajas <= 3 and 100 <= tick < 450) else 0.2
                s["SELLADO_RATIO"] += 2.5 + bajas * 1.0 + (2.4 if tick >= 500 else 0.0)
            else:
                s["ANILLO_INTERNO"] += 3.2 if tick < 44 else 0.9
                s["EXPANSION_MEDIA"] += 3.3 if 44 <= tick < 200 else 1.4
                s["EXPANSION_EXTERIOR"] += 2.7 if (100 <= tick < 480 and h_r1 + h_r2 <= 5) else 0.7
                s["SELLADO_RATIO"] += 3.8 if tick >= 500 else 0.4

            return sorted(POLITICAS_ORDEN, key=lambda p: (-s[p], POLITICAS_ORDEN.index(p)))

        boletas = {rol: boleta_rol(rol) for rol in pesos}

        duelos: Dict[str, Dict[str, float]] = {
            p1: {p2: 0.0 for p2 in POLITICAS_ORDEN} for p1 in POLITICAS_ORDEN
        }
        for rol, ord_r in boletas.items():
            w = pesos[rol]
            rank_pos = {p: i for i, p in enumerate(ord_r)}
            for p1 in POLITICAS_ORDEN:
                for p2 in POLITICAS_ORDEN:
                    if p1 != p2 and rank_pos[p1] < rank_pos[p2]:
                        duelos[p1][p2] += w

        copeland: Dict[str, float] = {p: 0.0 for p in POLITICAS_ORDEN}
        ganador_condorcet: Optional[str] = None
        for p1 in POLITICAS_ORDEN:
            victorias = 0
            for p2 in POLITICAS_ORDEN:
                if p1 == p2:
                    continue
                if duelos[p1][p2] > duelos[p2][p1]:
                    copeland[p1] += 1.0
                    victorias += 1
                elif duelos[p1][p2] == duelos[p2][p1]:
                    copeland[p1] += 0.5
            if victorias == len(POLITICAS_ORDEN) - 1:
                ganador_condorcet = p1

        if ganador_condorcet is not None:
            ganadora = ganador_condorcet
            metodo = "Condorcet Ponderado"
        else:
            ganadora = max(
                POLITICAS_ORDEN,
                key=lambda p: (copeland[p], -POLITICAS_ORDEN.index(p)),
            )
            metodo = "Desempate Copeland Ponderado"

        nueva_pol = PoliticaAnular(ganadora)
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
                "pesos": pesos,
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
                        "ring-governance",
                        f"gov_pol_{mem.politica_epoch}",
                        {"politica": ganadora, "epoch": mem.politica_epoch},
                        asunto_legado="gov_politica",
                    )

    # -------------------------------------------------------------------------
    # Sesión 2: Navegación segura A* Acotada y riesgo de tormenta
    # -------------------------------------------------------------------------
    def _riesgo_tormenta_celda(
        self, pos: Tuple[int, int], mem: MemoriaColmena, tick: int, pasos_fut: int = 1
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
        self, origen: Tuple[int, int], destino: Tuple[int, int], mem: MemoriaColmena, tick: int
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
        self, origen: Tuple[int, int], destino: Tuple[int, int], mem: MemoriaColmena, tick: int
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
        self, pos: Tuple[int, int], mem: MemoriaColmena, tick: int, ancla: Optional[Tuple[int, int]] = None
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
    # Mecánicas obligatorias y lógica diferenciada por rol
    # -------------------------------------------------------------------------
    def _manejar_bateria_baja_c(
        self, unidad: UnidadVista, percepcion: Percepcion, api: Any, mem: MemoriaColmena
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
                        lanzado = self._iniciar_subasta_vickrey_rescate(unidad, percepcion, api, mem)
                    if not lanzado:
                        self._enviar_acl_helper(
                            api,
                            mem,
                            "recargador",
                            "REQUEST",
                            "rescue-triage",
                            f"resc_{unidad.id}_{tick}",
                            {"rover_id": unidad.id, "pos": unidad.pos, "carga_rover": unidad.carga},
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
                        "rescue-triage",
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

    def _acumular_y_despachar_lotes_dron(
        self, unidad: UnidadVista, percepcion: Percepcion, api: Any, mem: MemoriaColmena
    ) -> None:
        """Agrupa vetas en buffer_lote y emite un único Bundle CFP / Batch INFORM."""
        if percepcion.fase_final:
            return
        tick = percepcion.tick
        _, radio_max, _, _ = self._limites_politica(mem)

        for hx, hy, cant in percepcion.hielo_visible:
            pos = (hx, hy)
            d_base = distancia_chebyshev(pos, percepcion.base_pos)
            if d_base > radio_max + 6 and cant < 3:
                continue
            if cant < 2 and d_base > 22:
                continue
            if mem.cooldowns.get(("hielo", pos), -9999) + 320 > tick:
                continue
            mem.buffer_lote[pos] = (cant, tick)

        mem.buffer_lote = {
            p: (c, t0) for p, (c, t0) in mem.buffer_lote.items() if tick - t0 <= 16
        }

        if mem.buffer_lote:
            ordenadas = sorted(
                mem.buffer_lote.items(),
                key=lambda kv: (-kv[1][0], distancia_chebyshev(unidad.pos, kv[0])),
            )
            semilla_pos, (semilla_cant, semilla_t) = ordenadas[0]
            cluster = [
                {"pos": p, "cantidad": c}
                for p, (c, _) in ordenadas
                if distancia_chebyshev(semilla_pos, p) <= 10
            ][:4]
            total_lote = sum(it["cantidad"] for it in cluster)

            listo = (
                len(cluster) >= 2
                or semilla_cant >= 4
                or (tick - semilla_t >= 3 and total_lote >= 3)
            )
            if listo:
                if self._iniciar_subasta_vickrey_lote(unidad, percepcion, api, mem, cluster):
                    for it in cluster:
                        p = tuple(it["pos"])
                        mem.cooldowns[("hielo", p)] = tick
                        mem.buffer_lote.pop(p, None)

        if percepcion.restos_visible:
            restos = max(percepcion.restos_visible, key=lambda r: r.get("combustible", 10))
            rid = restos["id"]
            rpos = tuple(restos["pos"])
            comb = int(restos.get("combustible", 10))
            if (
                comb >= 20
                and distancia_chebyshev(rpos, percepcion.base_pos) <= 32
                and self._pasa_filtro(mem, ("restos", rid), tick, cooldown=280)
            ):
                self._enviar_acl_helper(
                    api,
                    mem,
                    "rover",
                    "INFORM",
                    "ring-discovery",
                    f"restos_{rid}_{tick}",
                    {"id": rid, "pos": rpos, "combustible": comb, "equipo": restos.get("equipo", self.letra)},
                    asunto_legado="restos",
                    pos_ref=rpos,
                )

    def _evaluar_restos_y_reportar(
        self, unidad: UnidadVista, percepcion: Percepcion, api: Any, mem: MemoriaColmena
    ) -> Optional[Tuple[Any, ...]]:
        if unidad.tipo in ("dron_explorador", "dron_recarga"):
            self._acumular_y_despachar_lotes_dron(unidad, percepcion, api, mem)
            return None

        if not percepcion.restos_visible or percepcion.fase_final:
            return None

        restos = max(percepcion.restos_visible, key=lambda r: r.get("combustible", 10))
        rid = restos["id"]
        rpos = tuple(restos["pos"])

        aliado_critico = None
        for al in percepcion.aliados_visible:
            if (
                al["id"] != unidad.id
                and (al["estado"] == "esperando_recarga" or al["bateria"] < 62.0)
                and distancia_chebyshev(unidad.pos, tuple(al["pos"])) <= 7
            ):
                aliado_critico = al
                break

        if (unidad.bateria < 78.0 or aliado_critico is not None) and distancia_chebyshev(unidad.pos, rpos) <= 1:
            self._transicionar_fsm(mem, EstadoRover.CONSUMIENDO_CARRONA)
            if unidad.bateria >= 80.0 and aliado_critico is not None:
                return ("consumir_restos", rid, ("transferir", aliado_critico["id"]))
            return ("consumir_restos", rid, "propia")

        return None

    def _decidir_explorador(
        self, unidad: UnidadVista, percepcion: Percepcion, mem: MemoriaColmena
    ) -> Tuple[Any, ...]:
        tick = percepcion.tick
        anillo_pol, _, _, _ = self._limites_politica(mem)

        if percepcion.fase_final or mem.politica_actual == PoliticaAnular.SELLADO_RATIO:
            ev = self._evadir_tormenta_estatico(unidad.pos, mem, tick)
            if ev is not None:
                return ("mover", ev[0], ev[1])
            self._transicionar_fsm(mem, EstadoExplorador.RETORNANDO_BASE)
            return ("esperar",)

        if mem.waypoints is None or mem.anillo_waypoints != anillo_pol:
            idx_e = self._extraer_indice_uid(unidad.id)
            if anillo_pol == 2 and (unidad.bateria < 58.0 or idx_e % 3 != 0):
                mem.ciclo_anillo_completo = True
            elif anillo_pol == 3 and (unidad.bateria < 65.0 or idx_e % 5 != 0):
                mem.ciclo_anillo_completo = True
            else:
                mem.anillo_waypoints = anillo_pol
                mem.waypoints = self._generar_waypoints_anillo(unidad, anillo_pol)
                mem.patrulla_idx = 0
                mem.ciclo_anillo_completo = False

        if mem.ciclo_anillo_completo or mem.waypoints is None:
            ev = self._evadir_tormenta_estatico(unidad.pos, mem, tick)
            if ev is not None:
                return ("mover", ev[0], ev[1])
            return ("esperar",)

        self._transicionar_fsm(mem, EstadoExplorador.PATRULLANDO)
        wps = mem.waypoints
        dest = wps[mem.patrulla_idx % len(wps)]
        if distancia_chebyshev(unidad.pos, dest) <= 2:
            mem.patrulla_idx += 1
            if mem.patrulla_idx >= len(wps):
                mem.ciclo_anillo_completo = True
                ev = self._evadir_tormenta_estatico(unidad.pos, mem, tick)
                if ev is not None:
                    return ("mover", ev[0], ev[1])
                return ("esperar",)
            dest = wps[mem.patrulla_idx]

        dx, dy = self._paso_seguro(unidad.pos, dest, mem, tick)
        return ("mover", dx, dy)

    def _decidir_recargador(
        self, unidad: UnidadVista, percepcion: Percepcion, api: Any, mem: MemoriaColmena
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
                        "rescue-triage",
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
        self, unidad: UnidadVista, percepcion: Percepcion, mem: MemoriaColmena
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
            mem.ruta_lote.clear()
            if percepcion.base_aliada_cerca:
                return ("depositar",)
            dest = self._punto_perimetro_base(unidad.pos, percepcion.base_pos)
            dx, dy = self._paso_seguro(unidad.pos, dest, mem, tick)
            return ("mover", dx, dy)

        for hx, hy, cant in percepcion.hielo_visible:
            if (hx, hy) == unidad.pos and cant > 0 and unidad.carga < unidad.capacidad:
                self._transicionar_fsm(mem, EstadoRover.RECOLECTANDO)
                return ("extraer",)

        while mem.ruta_lote and mem.ruta_lote[0] not in mem.hielo:
            mem.ruta_lote.pop(0)
        if mem.ruta_lote:
            sig = mem.ruta_lote[0]
            d_ida = distancia_chebyshev(unidad.pos, sig)
            d_vuelta = max(0, distancia_chebyshev(sig, percepcion.base_pos) - 2)
            # Solo seguir lote si no hay una veta visible inmediata a distancia <= 2 cuando sig está lejos
            veta_inmediata = any(
                distancia_chebyshev(unidad.pos, (hx, hy)) <= 2 and cant >= 2
                for hx, hy, cant in percepcion.hielo_visible
            )
            if (d_ida <= 5 or not veta_inmediata) and unidad.bateria - (d_ida + d_vuelta) >= UMBRAL_BATERIA_BAJA + 4.0:
                self._transicionar_fsm(mem, EstadoRover.RECOLECTANDO)
                mem.objetivo_hielo = sig
                dx, dy = self._paso_seguro(unidad.pos, sig, mem, tick)
                return ("mover", dx, dy)
            if d_ida > 5 and veta_inmediata:
                mem.ruta_lote.clear()

        _, radio_max, tau_marg, alpha_ret = self._limites_politica(mem)
        cap_libre = max(0, unidad.capacidad - unidad.carga)
        es_encadenado = unidad.carga > 0
        hielo_vis_coords = {(hx, hy) for hx, hy, _ in percepcion.hielo_visible}

        mejor_u = -float("inf")
        mejor_tipo: Optional[str] = None
        mejor_target: Any = None

        for pos, cr in mem.hielo.items():
            if ("hielo", pos) in mem.reservas_ajenas and pos != mem.objetivo_hielo:
                continue
            d_base = distancia_chebyshev(pos, percepcion.base_pos)
            if d_base > radio_max and cr.cantidad < 3 and not es_encadenado:
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

            d_ida = distancia_chebyshev(unidad.pos, pos)
            if es_encadenado and d_ida > 13:
                continue
            d_vuelta = max(0, d_base - 2)
            if unidad.bateria - (d_ida + d_vuelta) < UMBRAL_BATERIA_BAJA + 4.0:
                continue

            extraible = min(cr.cantidad, cap_libre)
            conf = cr.confianza(tick)
            r_torm = self._riesgo_tormenta_segmento(unidad.pos, pos, mem, tick)

            eta = (extraible * conf) / max(1.0, d_ida + alpha_ret * d_vuelta + 4.0 * r_torm)
            umbral_efectivo = 0.025 if es_encadenado else tau_marg
            if eta < umbral_efectivo and pos not in hielo_vis_coords:
                continue

            u_val = (
                4.5 * extraible * conf * (1.50 if es_encadenado else 1.0)
                - 0.30 * (d_ida + (0.18 if es_encadenado else alpha_ret) * d_vuelta)
                - 2.6 * r_torm
            )
            if u_val > mejor_u:
                mejor_u = u_val
                mejor_tipo = "hielo"
                mejor_target = pos

        if unidad.carga == 0 and unidad.bateria < 78.0:
            for rid, cr in mem.restos.items():
                d_ida = distancia_chebyshev(unidad.pos, cr.pos)
                r_torm = self._riesgo_tormenta_segmento(unidad.pos, cr.pos, mem, tick)
                u_val = 2.1 * (cr.combustible / 10.0) - 0.32 * d_ida - 2.6 * r_torm
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
                return ("consumir_restos", r_id, "propia")
            dx, dy = self._paso_seguro(unidad.pos, r_pos, mem, tick)
            return ("mover", dx, dy)

        if mejor_tipo == "hielo" and mejor_target is not None:
            self._transicionar_fsm(mem, EstadoRover.RECOLECTANDO)
            mem.objetivo_hielo = mejor_target
            dx, dy = self._paso_seguro(unidad.pos, mejor_target, mem, tick)
            return ("mover", dx, dy)

        if tick < 75 and mem.politica_actual != PoliticaAnular.SELLADO_RATIO:
            self._transicionar_fsm(mem, EstadoRover.EXPLORANDO)
            idx = self._extraer_indice_uid(unidad.id)
            w, h = percepcion.grid_size
            bx, by = percepcion.base_pos
            dir_x = 1 if bx < w // 2 else -1
            dir_y = 1 if by < h // 2 else -1
            rad = 8 + (idx % 4) * 3
            ang = ((idx % 12) / 11.0) * (math.pi * 0.46)
            tx = int(clamp(bx + dir_x * math.cos(ang) * rad, 3, w - 4))
            ty = int(clamp(by + dir_y * math.sin(ang) * rad, 3, h - 4))
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
            self._actualizar_percepcion_y_olvido(unidad, percepcion, mem)
            self._procesar_mensajes_acl(unidad, percepcion, api, mem)

            if mem.subastas_activas:
                self._adjudicar_subastas_vickrey(unidad, percepcion, api, mem)

            self._enviar_telemetria_gobernanza(unidad, percepcion, api, mem)
            self._votar_condorcet_copeland(unidad, percepcion, api, mem)

            accion_bat = self._manejar_bateria_baja_c(unidad, percepcion, api, mem)
            if accion_bat is not None:
                return accion_bat

            if unidad.estado == "consumiendo":
                self._transicionar_fsm(mem, EstadoRover.CONSUMIENDO_CARRONA)
                return ("esperar",)

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
            "vickrey": dict(self.stats_vickrey),
            "votaciones": len(self.historial_votaciones),
            "transiciones_fsm": len(self.transiciones_fsm),
        }
