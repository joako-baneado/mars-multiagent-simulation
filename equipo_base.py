"""
equipo_base.py
==============
Plantilla documentada para crear equipos compatibles con `simulador.py`.

Para crear un nuevo equipo (ej. equipoX.py):
1. Importa `EquipoBase` y las utilidades desde `equipo_base`.
2. Define una subclase de `EquipoBase` (el simulador detectará automáticamente la subclase).
3. Personaliza los métodos estratégicos (`decidir_unidad`, `al_ver_restos`,
   `decidir_rover`, `decidir_dron_explorador`, `decidir_dron_recarga`, etc.)
   respetando las mecánicas obligatorias del entorno.

Reglas de energía y acciones válidas que puedes retornar en `decidir_unidad`:
  - ("mover", dx, dy)                        -> dx, dy in {-1, 0, 1}. Costo: -1% normal, -5% en tormenta.
  - ("extraer",)                             -> Extrae 1 u. de hielo de la celda actual (Rovers).
  - ("depositar",)                           -> Deposita carga en base (si distancia <= 2 a tu base).
  - ("solicitar_recarga",)                   -> Rover se detiene, activa alerta roja y espera recarga.
  - ("recargar_aliado", rover_id)            -> Dron de recarga recarga a un rover cercano (dist <= 1).
  - ("recargar_en_base",)                    -> Dron en base (dist <= 2) restaura batería y recargador.
  - ("consumir_restos", restos_id, modo)     -> Rover junto a restos (dist <= 1) consume carroña.
                                                modo = "propia" o ("transferir", aliado_id).
  - ("esperar",)                             -> Permanece inmóvil (0% costo de movimiento).

Comunicaciones (`api`):
  - Cada mensaje enviado o recibido cuesta -0.5% de batería al emisor y -0.5% al receptor.
  - `api.enviar_mensaje(dest_id, asunto, datos)`
  - `api.enviar_a_recargador_cercano(asunto, datos)`
  - `api.enviar_a_rover_ocioso_cercano(asunto, datos, pos_ref=None)`
  - `api.enviar_a_aliados_cercanos(radio, asunto, datos, max_dest=2)`
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

GRID_W = 60
GRID_H = 45
UMBRAL_BATERIA_BAJA = 25.0
CAPACIDAD_ROVER = 5


@dataclass
class UnidadVista:
    """Estado de la unidad propia que está tomando una decisión."""
    id: str
    equipo: str
    tipo: str  # "rover" | "dron_explorador" | "dron_recarga"
    pos: Tuple[int, int]
    bateria: float
    carga: int
    capacidad: int
    estado: str  # "activo" | "esperando_recarga" | "consumiendo" | "muerto"
    en_tormenta: bool
    carga_recarga: int = 0


@dataclass
class Percepcion:
    """Información percibida localmente por la unidad en el tick actual."""
    tick: int
    base_pos: Tuple[int, int]
    grid_size: Tuple[int, int]
    fase_final: bool
    tiempo_final_restante: float
    base_aliada_cerca: bool
    hielo_visible: List[Tuple[int, int, int]] = field(default_factory=list)  # (x, y, cantidad)
    restos_visible: List[Dict[str, Any]] = field(default_factory=list)
    aliados_visible: List[Dict[str, Any]] = field(default_factory=list)
    rivales_visible: List[Dict[str, Any]] = field(default_factory=list)
    tormentas_visible: List[Tuple[float, float, float]] = field(default_factory=list)  # (tx, ty, radio)
    mensajes: List[Dict[str, Any]] = field(default_factory=list)


def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def distancia_manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def distancia_chebyshev(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def distancia_euclidea(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def paso_hacia(
    origen: Tuple[int, int],
    destino: Tuple[int, int],
    tormentas: Optional[List[Tuple[float, float, float]]] = None,
    grid_size: Tuple[int, int] = (GRID_W, GRID_H),
    penalizar_tormenta: float = 6.0,
) -> Tuple[int, int]:
    """Calcula el mejor paso (dx, dy) hacia `destino`, evitando tormentas si es posible."""
    if origen == destino:
        return (0, 0)

    w, h = grid_size
    mejor_paso = (0, 0)
    mejor_score = float("inf")

    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx, ny = origen[0] + dx, origen[1] + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue

            score = math.hypot(destino[0] - nx, destino[1] - ny)
            if tormentas and penalizar_tormenta > 0:
                for tx, ty, radio in tormentas:
                    d_storm = math.hypot(nx - tx, ny - ty)
                    if d_storm <= radio + 0.5:
                        score += penalizar_tormenta + (radio + 1.0 - d_storm)

            if score < mejor_score:
                mejor_score = score
                mejor_paso = (dx, dy)

    return mejor_paso


def paso_alejarse(
    origen: Tuple[int, int],
    amenazas: List[Tuple[int, int]],
    tormentas: Optional[List[Tuple[float, float, float]]] = None,
    grid_size: Tuple[int, int] = (GRID_W, GRID_H),
    preferencia_destino: Optional[Tuple[int, int]] = None,
) -> Tuple[int, int]:
    """Calcula el paso (dx, dy) que maximiza la distancia respecto a una lista de amenazas."""
    if not amenazas:
        return (0, 0)

    w, h = grid_size
    mejor_paso = (0, 0)
    mejor_score = -float("inf")

    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx, ny = origen[0] + dx, origen[1] + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue

            # Distancia ponderada respecto a amenazas (más peso a las más cercanas)
            dist_min = min(math.hypot(nx - ax, ny - ay) for ax, ay in amenazas)
            dist_sum = sum(1.0 / max(0.5, math.hypot(nx - ax, ny - ay)) for ax, ay in amenazas)
            score = dist_min * 3.0 - dist_sum * 4.0

            if preferencia_destino is not None:
                score -= math.hypot(nx - preferencia_destino[0], ny - preferencia_destino[1]) * 0.25

            if tormentas:
                for tx, ty, radio in tormentas:
                    if math.hypot(nx - tx, ny - ty) <= radio:
                        score -= 4.5

            if score > mejor_score:
                mejor_score = score
                mejor_paso = (dx, dy)

    return mejor_paso


class EquipoBase:
    """
    Clase base documentada que implementa todas las mecánicas obligatorias:
      - Gestión de batería baja (drones vuelven a base; rovers se detienen y piden recarga).
      - Exploración continua de drones exploradores con aviso de hielo/restos.
      - Drones de recarga en espera en base hasta atender rovers con batería baja y volver a base.
      - Rovers ociosos recolectando hielo o consumiendo carroña.
      - Lógica de reacción ante restos vistos (aviso, interrupción/consumo).
    """

    nombre: str = "Equipo Base"

    def __init__(self) -> None:
        self.letra: str = "?"
        self.base_pos: Tuple[int, int] = (0, 0)
        self.grid_size: Tuple[int, int] = (GRID_W, GRID_H)
        self.memoria: Dict[str, Dict[str, Any]] = {}
        self.hielo_conocido: Dict[Tuple[int, int], int] = {}
        self.restos_conocidos: Dict[str, Dict[str, Any]] = {}
        self.solicitudes_recarga: Dict[str, Tuple[int, int]] = {}

    def inicializar(
        self,
        letra: str,
        base_pos: Tuple[int, int],
        grid_size: Tuple[int, int] = (GRID_W, GRID_H),
    ) -> None:
        self.letra = letra
        self.base_pos = base_pos
        self.grid_size = grid_size
        self.memoria.clear()
        self.hielo_conocido.clear()
        self.restos_conocidos.clear()
        self.solicitudes_recarga.clear()

    def _mem(self, uid: str) -> Dict[str, Any]:
        if uid not in self.memoria:
            self.memoria[uid] = {
                "reportados": {},         # (tipo, clave) -> ultimo_tick
                "patrulla_idx": 0,
                "waypoints": None,
                "objetivo_hielo": None,
                "objetivo_restos": None,  # {"id": ..., "pos": ...}
                "rescate_actual": None,   # {"rover_id": ..., "pos": ...}
                "ultimo_ping_recarga": -999,
            }
        return self.memoria[uid]

    def _puede_reportar(self, mem: Dict[str, Any], clave: Any, tick: int, cooldown: int = 35) -> bool:
        ultimo = mem["reportados"].get(clave, -9999)
        if tick - ultimo >= cooldown:
            mem["reportados"][clave] = tick
            return True
        return False

    def _generar_waypoints_explorador(self, idx_explorador: int) -> List[Tuple[int, int]]:
        """Genera una ruta de patrulla distribuida en los 15 drones exploradores."""
        w, h = self.grid_size
        cols = 5
        rows = 3
        sector = idx_explorador % (cols * rows)
        sx = sector % cols
        sy = sector // cols

        x0 = max(3, int((sx + 0.2) * (w / cols)))
        x1 = min(w - 4, int((sx + 0.85) * (w / cols)))
        y0 = max(3, int((sy + 0.2) * (h / rows)))
        y1 = min(h - 4, int((sy + 0.85) * (h / rows)))

        cx = (w // 2) + ((idx_explorador % 3) - 1) * 6
        cy = (h // 2) + (((idx_explorador // 3) % 3) - 1) * 5
        cx = int(clamp(cx, 4, w - 5))
        cy = int(clamp(cy, 4, h - 5))

        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (cx, cy)]

    def procesar_mensajes(self, unidad: UnidadVista, percepcion: Percepcion) -> None:
        """Procesa mensajes recibidos por la unidad en este tick."""
        mem = self._mem(unidad.id)
        for msg in percepcion.mensajes:
            asunto = msg.get("asunto")
            datos = msg.get("datos", {})

            if asunto == "hielo":
                pos = tuple(datos["pos"])
                cant = int(datos.get("cantidad", 1))
                self.hielo_conocido[pos] = cant
                if unidad.tipo == "rover" and mem["objetivo_hielo"] is None and unidad.carga == 0:
                    mem["objetivo_hielo"] = pos

            elif asunto == "restos":
                rid = datos.get("id")
                pos = tuple(datos["pos"])
                if rid:
                    self.restos_conocidos[rid] = datos
                    if (
                        unidad.tipo == "rover"
                        and mem["objetivo_restos"] is None
                        and unidad.bateria < 85.0
                        and unidad.carga == 0
                    ):
                        mem["objetivo_restos"] = {"id": rid, "pos": pos}

            elif asunto == "solicitud_recarga":
                rover_id = datos.get("rover_id")
                pos = tuple(datos["pos"])
                if rover_id:
                    self.solicitudes_recarga[rover_id] = pos
                    if unidad.tipo == "dron_recarga" and mem["rescate_actual"] is None:
                        mem["rescate_actual"] = {"rover_id": rover_id, "pos": pos}

    def manejar_bateria_baja(
        self, unidad: UnidadVista, percepcion: Percepcion, api: Any
    ) -> Optional[Tuple[Any, ...]]:
        """
        Mecánica obligatoria de batería baja:
          - Rovers: se detienen, activan alerta y solicitan dron recargador cercano.
          - Drones: abortan tarea y vuelven a base para recargar.
        """
        mem = self._mem(unidad.id)

        if unidad.tipo == "rover":
            if unidad.estado == "esperando_recarga" or unidad.bateria <= UMBRAL_BATERIA_BAJA:
                if percepcion.tick - mem["ultimo_ping_recarga"] >= 20:
                    mem["ultimo_ping_recarga"] = percepcion.tick
                    api.enviar_a_recargador_cercano(
                        "solicitud_recarga",
                        {"rover_id": unidad.id, "pos": unidad.pos},
                    )
                if unidad.estado != "esperando_recarga":
                    return ("solicitar_recarga",)
                return ("esperar",)
            return None

        # Drones (exploración o recarga)
        if unidad.bateria <= UMBRAL_BATERIA_BAJA:
            if percepcion.base_aliada_cerca:
                return ("recargar_en_base",)
            dx, dy = paso_hacia(unidad.pos, percepcion.base_pos, percepcion.tormentas_visible, percepcion.grid_size)
            return ("mover", dx, dy)

        return None

    def al_ver_restos(
        self, unidad: UnidadVista, percepcion: Percepcion, api: Any
    ) -> Optional[Tuple[Any, ...]]:
        """
        Mecánica al detectar restos (carroña):
          - Avisa a aliados cercanos/rovers sobre los restos (costo -0.5% por mensaje).
          - Si es un rover que necesita energía (o está junto a los restos), interrumpe
            su tarea si conviene y los consume (propia batería o transferir a aliado).
        """
        if not percepcion.restos_visible:
            return None

        mem = self._mem(unidad.id)
        restos = percepcion.restos_visible[0]
        rid = restos["id"]
        rpos = tuple(restos["pos"])
        self.restos_conocidos[rid] = restos

        # Alertar a aliados/rover ocioso con control de spam
        if self._puede_reportar(mem, ("restos", rid), percepcion.tick, cooldown=40):
            if unidad.tipo in ("dron_explorador", "dron_recarga"):
                api.enviar_a_rover_ocioso_cercano("restos", restos, pos_ref=rpos)
            else:
                api.enviar_a_aliados_cercanos(7, "restos", restos, max_dest=1)

        # Si es rover y está al lado de los restos, evaluar consumirlos
        if unidad.tipo == "rover":
            aliado_necesitado = None
            for al in percepcion.aliados_visible:
                if (
                    al["id"] != unidad.id
                    and al["bateria"] < 65.0
                    and distancia_chebyshev(unidad.pos, tuple(al["pos"])) <= 6
                ):
                    aliado_necesitado = al
                    break

            debe_consumir = (
                unidad.bateria < 82.0
                or aliado_necesitado is not None
                or (unidad.carga == 0 and mem["objetivo_hielo"] is None)
            )

            if debe_consumir:
                if distancia_chebyshev(unidad.pos, rpos) <= 1:
                    if unidad.bateria >= 85.0 and aliado_necesitado is not None:
                        return ("consumir_restos", rid, ("transferir", aliado_necesitado["id"]))
                    return ("consumir_restos", rid, "propia")
                elif unidad.carga == 0:
                    mem["objetivo_restos"] = {"id": rid, "pos": rpos}

        return None

    def _reportar_hallazgos_dron(self, unidad: UnidadVista, percepcion: Percepcion, api: Any) -> None:
        """Drones reportan hielo/restos nuevos encontrados en su campo de visión."""
        mem = self._mem(unidad.id)
        if percepcion.hielo_visible:
            # Reportar el depósito de mayor cantidad visible aún no reportado recientemente
            candidatos = sorted(
                percepcion.hielo_visible,
                key=lambda h: (-h[2], distancia_manhattan(unidad.pos, (h[0], h[1]))),
            )
            for hx, hy, cant in candidatos:
                self.hielo_conocido[(hx, hy)] = cant
                if self._puede_reportar(mem, ("hielo", (hx, hy)), percepcion.tick, cooldown=45):
                    api.enviar_a_rover_ocioso_cercano(
                        "hielo",
                        {"pos": (hx, hy), "cantidad": cant},
                        pos_ref=(hx, hy),
                    )
                    break  # Máximo 1 mensaje de hielo por tick para ahorrar batería

    def decidir_dron_explorador(
        self, unidad: UnidadVista, percepcion: Percepcion, api: Any
    ) -> Tuple[Any, ...]:
        """Patrulla continua por sectores y aviso de hielo/restos."""
        mem = self._mem(unidad.id)
        self._reportar_hallazgos_dron(unidad, percepcion, api)

        if mem["waypoints"] is None:
            try:
                num = int(unidad.id.split("_")[-1])
            except Exception:
                num = hash(unidad.id) % 15
            mem["waypoints"] = self._generar_waypoints_explorador(num)

        wps = mem["waypoints"]
        destino = wps[mem["patrulla_idx"] % len(wps)]
        if distancia_chebyshev(unidad.pos, destino) <= 2:
            mem["patrulla_idx"] = (mem["patrulla_idx"] + 1) % len(wps)
            destino = wps[mem["patrulla_idx"]]

        dx, dy = paso_hacia(unidad.pos, destino, percepcion.tormentas_visible, percepcion.grid_size)
        return ("mover", dx, dy)

    def decidir_dron_recarga(
        self, unidad: UnidadVista, percepcion: Percepcion, api: Any
    ) -> Tuple[Any, ...]:
        """
        Solo sale a recargar rovers con batería baja y vuelve a base al terminar.
        También avisa hielo/restos que observe en el trayecto.
        """
        mem = self._mem(unidad.id)
        self._reportar_hallazgos_dron(unidad, percepcion, api)

        # Si gastó su recarga o ya no tiene misión, volver a base
        if unidad.carga_recarga <= 0:
            mem["rescate_actual"] = None
            if percepcion.base_aliada_cerca:
                return ("recargar_en_base",)
            dx, dy = paso_hacia(unidad.pos, percepcion.base_pos, percepcion.tormentas_visible, percepcion.grid_size)
            return ("mover", dx, dy)

        # Si tiene rescate asignado, ir hacia ese rover
        rescate = mem["rescate_actual"]
        if rescate is not None:
            r_id = rescate["rover_id"]
            r_pos = tuple(rescate["pos"])

            # Verificar si ya murió o si ya fue recargado cuando estemos cerca
            if distancia_chebyshev(unidad.pos, r_pos) <= 4:
                aliado = next((a for a in percepcion.aliados_visible if a["id"] == r_id), None)
                if aliado is None or aliado["estado"] != "esperando_recarga":
                    self.solicitudes_recarga.pop(r_id, None)
                    mem["rescate_actual"] = None
                    rescate = None
                else:
                    r_pos = tuple(aliado["pos"])

            if rescate is not None:
                if distancia_chebyshev(unidad.pos, r_pos) <= 1:
                    self.solicitudes_recarga.pop(r_id, None)
                    mem["rescate_actual"] = None
                    return ("recargar_aliado", r_id)
                dx, dy = paso_hacia(unidad.pos, r_pos, percepcion.tormentas_visible, percepcion.grid_size)
                return ("mover", dx, dy)

        # Si no tiene misión activa, revisar si hay solicitudes pendientes en el equipo
        if self.solicitudes_recarga:
            r_id, r_pos = min(
                self.solicitudes_recarga.items(),
                key=lambda item: distancia_manhattan(unidad.pos, item[1]),
            )
            mem["rescate_actual"] = {"rover_id": r_id, "pos": r_pos}
            dx, dy = paso_hacia(unidad.pos, r_pos, percepcion.tormentas_visible, percepcion.grid_size)
            return ("mover", dx, dy)

        # Esperar en base y mantenerse al 100%
        if not percepcion.base_aliada_cerca:
            dx, dy = paso_hacia(unidad.pos, percepcion.base_pos, percepcion.tormentas_visible, percepcion.grid_size)
            return ("mover", dx, dy)
        if unidad.bateria < 99.0 or unidad.carga_recarga <= 0:
            return ("recargar_en_base",)
        return ("esperar",)

    def decidir_rover(
        self, unidad: UnidadVista, percepcion: Percepcion, api: Any
    ) -> Tuple[Any, ...]:
        """Lógica de rover: extraer hielo, devolver a base o consumir restos cuando está ocioso."""
        mem = self._mem(unidad.id)

        # En fase final (30s) o con carga llena, regresar de inmediato si lleva hielo
        if unidad.carga >= unidad.capacidad or (percepcion.fase_final and unidad.carga > 0):
            if percepcion.base_aliada_cerca:
                return ("depositar",)
            dx, dy = paso_hacia(unidad.pos, percepcion.base_pos, percepcion.tormentas_visible, percepcion.grid_size)
            return ("mover", dx, dy)

        # Si está sobre una celda con hielo y tiene espacio, extraer
        for hx, hy, cant in percepcion.hielo_visible:
            if (hx, hy) == unidad.pos and cant > 0 and unidad.carga < unidad.capacidad:
                return ("extraer",)

        # Si lleva algo de carga pero no hay hielo visible cerca, volver a base a depositar
        if unidad.carga > 0 and not percepcion.hielo_visible:
            if percepcion.base_aliada_cerca:
                return ("depositar",)
            dx, dy = paso_hacia(unidad.pos, percepcion.base_pos, percepcion.tormentas_visible, percepcion.grid_size)
            return ("mover", dx, dy)

        # Si tiene un objetivo de restos pendiente (y no está lleno de batería)
        if mem["objetivo_restos"] is not None and unidad.carga == 0:
            r_id = mem["objetivo_restos"]["id"]
            r_pos = tuple(mem["objetivo_restos"]["pos"])
            if distancia_chebyshev(unidad.pos, r_pos) <= 1:
                visible = any(r["id"] == r_id for r in percepcion.restos_visible)
                mem["objetivo_restos"] = None
                if visible:
                    return ("consumir_restos", r_id, "propia")
            elif distancia_chebyshev(unidad.pos, r_pos) <= 5 and not any(
                r["id"] == r_id for r in percepcion.restos_visible
            ):
                mem["objetivo_restos"] = None
            else:
                dx, dy = paso_hacia(unidad.pos, r_pos, percepcion.tormentas_visible, percepcion.grid_size)
                return ("mover", dx, dy)

        # Buscar el hielo más cercano visible
        if percepcion.hielo_visible:
            hx, hy, _ = min(
                percepcion.hielo_visible,
                key=lambda h: distancia_manhattan(unidad.pos, (h[0], h[1])) - h[2] * 0.6,
            )
            mem["objetivo_hielo"] = (hx, hy)
            dx, dy = paso_hacia(unidad.pos, (hx, hy), percepcion.tormentas_visible, percepcion.grid_size)
            return ("mover", dx, dy)

        # Si tiene un hielo conocido reportado por un dron
        if mem["objetivo_hielo"] is not None:
            obj = tuple(mem["objetivo_hielo"])
            if distancia_chebyshev(unidad.pos, obj) <= 2:
                mem["objetivo_hielo"] = None
            else:
                dx, dy = paso_hacia(unidad.pos, obj, percepcion.tormentas_visible, percepcion.grid_size)
                return ("mover", dx, dy)

        # Usar hielo conocido globalmente reportado por mensajes previos
        if self.hielo_conocido:
            obj = min(
                self.hielo_conocido.keys(),
                key=lambda p: distancia_manhattan(unidad.pos, p),
            )
            mem["objetivo_hielo"] = obj
            dx, dy = paso_hacia(unidad.pos, obj, percepcion.tormentas_visible, percepcion.grid_size)
            return ("mover", dx, dy)

        # Ocioso sin objetivo conocido: avanzar hacia zona de exploración según su ID
        try:
            num = int(unidad.id.split("_")[-1])
        except Exception:
            num = hash(unidad.id) % 30
        w, h = percepcion.grid_size
        angle = (num / 30.0) * math.pi * 2.0
        tx = int(clamp(w * 0.5 + math.cos(angle) * (w * 0.28), 3, w - 4))
        ty = int(clamp(h * 0.5 + math.sin(angle) * (h * 0.28), 3, h - 4))
        dx, dy = paso_hacia(unidad.pos, (tx, ty), percepcion.tormentas_visible, percepcion.grid_size)
        return ("mover", dx, dy)

    def decidir_unidad(
        self, unidad: UnidadVista, percepcion: Percepcion, api: Any
    ) -> Tuple[Any, ...]:
        """Punto de entrada principal llamado por `simulador.py` para cada unidad."""
        if unidad.estado == "muerto":
            return ("esperar",)

        self.procesar_mensajes(unidad, percepcion)

        # 1. Mecánica obligatoria de batería baja
        accion_bat = self.manejar_bateria_baja(unidad, percepcion, api)
        if accion_bat is not None:
            return accion_bat

        # 2. Si está actualmente consumiendo carroña, terminar el consumo
        if unidad.estado == "consumiendo":
            return ("esperar",)

        # 3. Mecánica de reacción al ver restos
        accion_restos = self.al_ver_restos(unidad, percepcion, api)
        if accion_restos is not None:
            return accion_restos

        # 4. Comportamiento según rol
        if unidad.tipo == "dron_explorador":
            return self.decidir_dron_explorador(unidad, percepcion, api)
        if unidad.tipo == "dron_recarga":
            return self.decidir_dron_recarga(unidad, percepcion, api)
        return self.decidir_rover(unidad, percepcion, api)