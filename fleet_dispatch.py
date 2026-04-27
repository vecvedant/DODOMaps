"""
Fleet Dispatch — multi-shipment cascade simulation
===================================================
Models a real fleet day: each vehicle runs its shipments sequentially, and a
delay on shipment N pushes shipment N+1's start time forward. That's the
cascade we're trying to prevent.

Two modes per fleet:
  - baseline_dispatch: every shipment uses raw shortest path
  - smart_dispatch:    every shipment uses the predictive smart-weight A*

Same conditions pipeline (calendar + learning + per-shipment time-of-day
traffic + live weather) feeds both modes — the only difference is whether the
router is allowed to detour around what it sees.

Design notes:
  - Conditions are recomputed per shipment based on the shipment's own
    start_time, so a delayed S1 can push S2 into rush-hour traffic.
  - Weather is fetched once per fleet day (it doesn't shift meaningfully at
    our resolution) and held constant across shipments.
  - Calendar events come from the existing IntegratedConditionsProvider by
    passing query_date — same code path as scenario and manual modes.
"""

from dataclasses import dataclass, field, asdict
from datetime import date
from typing import List, Dict, Optional, Tuple

from cities import CityConfig, get_city
from learning_loop import LearningStore
from multicity_router import (
    Conditions, RoutingEngine, IntegratedConditionsProvider
)


# =============================================================================
# CONSTANTS
# =============================================================================

LOADING_BUFFER_MIN = 10
"""Minutes between the end of shipment N and the start of shipment N+1
(unload + reload at the destination + crew handover)."""

FLEET_DAY_START_HOUR = 9
"""The fleet day starts at 9:00 AM local time. Per-shipment traffic baseline
is derived by adding shipment.start_time/60 to this hour."""

ON_TIME_MARGIN_MIN = 15
"""A shipment is ON_TIME only if its margin (deadline - end_time) is at least
this many minutes. Below this it's AT_RISK; below zero it's LATE."""


# =============================================================================
# TIME-OF-DAY → TRAFFIC (matches manual mode buckets)
# =============================================================================

def _traffic_for_hour(hour: int) -> Tuple[str, float]:
    if 8 <= hour < 11:    return ("morning rush", 0.65)
    if 11 <= hour < 17:   return ("midday",       0.35)
    if 17 <= hour < 21:   return ("evening rush", 0.70)
    return ("off-peak", 0.15)


def _dow_adjust(weekday_idx: int) -> float:
    if weekday_idx == 5: return -0.10   # Saturday
    if weekday_idx == 6: return -0.20   # Sunday
    return 0.0


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class Shipment:
    id: str
    vehicle_id: str
    origin_node: str                # graph node id (hand-built OR OSM after resolution)
    destination_node: str
    deadline_min_from_now: int
    priority: str = 'standard'      # 'standard' | 'express'
    load_kg: int = 100
    # Human-readable place names — preserved through resolution so the UI can
    # always say "Hinjewadi → Viman Nagar" regardless of which graph is active.
    origin_name: str = ''
    destination_name: str = ''


@dataclass
class ShipmentResult:
    shipment: Shipment
    sequence: int                   # 0-indexed within the vehicle's run
    start_time: float               # min from t=0
    end_time: float                 # min from t=0
    route_time: float
    path_ids: List[str]
    path_names: List[str]
    coords: List[List[float]]
    margin: float                   # deadline - end_time (negative if LATE)
    status: str                     # ON_TIME | AT_RISK | LATE
    conditions: Dict                # snapshot for the assumptions UI
    error: Optional[str] = None


@dataclass
class FleetResult:
    mode: str                       # 'baseline' | 'smart'
    shipments: List[ShipmentResult]
    total_late: int
    total_at_risk: int
    total_on_time: int
    total_minutes_late: float       # sum over LATE of (end_time - deadline)
    fleet_day_start: str            # "9:00 AM"


# =============================================================================
# DEMO FLEET — tuned so IPL day produces baseline 5 LATE, smart 0 LATE,
# while "Today" (no event) keeps both modes mostly ON_TIME.
# =============================================================================

DEMO_FLEET: List[Shipment] = [
    # ----- V1: northern corridor, S1 hits IPL highway, cascades -----------
    Shipment('S1', 'V1', 'WH',  'DEL', deadline_min_from_now=85,  priority='express', load_kg=520),
    Shipment('S2', 'V1', 'DEL', 'SWG', deadline_min_from_now=130, priority='standard', load_kg=310),
    Shipment('S3', 'V1', 'SWG', 'MGP', deadline_min_from_now=165, priority='standard', load_kg=280),

    # ----- V2: Wakad → highway → cascades through 4 shipments -------------
    Shipment('S1', 'V2', 'A',   'DEL', deadline_min_from_now=70,  priority='express', load_kg=480),
    Shipment('S2', 'V2', 'DEL', 'CMP', deadline_min_from_now=110, priority='standard', load_kg=350),
    Shipment('S3', 'V2', 'CMP', 'HAD', deadline_min_from_now=145, priority='standard', load_kg=290),
    Shipment('S4', 'V2', 'HAD', 'MGP', deadline_min_from_now=175, priority='standard', load_kg=240),

    # ----- V3: central/southern, never touches IPL corridor ---------------
    Shipment('S1', 'V3', 'SWG', 'KTH', deadline_min_from_now=50,  priority='standard', load_kg=180),
    Shipment('S2', 'V3', 'KTH', 'KRV', deadline_min_from_now=70,  priority='standard', load_kg=160),
    Shipment('S3', 'V3', 'KRV', 'SNH', deadline_min_from_now=90,  priority='standard', load_kg=140),

    # ----- V4: mixed, S1 touches IPL but deadline keeps it AT_RISK only ---
    Shipment('S1', 'V4', 'PIM', 'DEL', deadline_min_from_now=90,  priority='express', load_kg=410),
    Shipment('S2', 'V4', 'DEL', 'YWD', deadline_min_from_now=130, priority='standard', load_kg=220),
]


def _populate_demo_names_from(city_key: str = 'pune') -> None:
    """One-time: stamp human-readable origin/destination names onto DEMO_FLEET
    by reading them out of the hand-built city. Called at module import."""
    city = get_city(city_key)
    for s in DEMO_FLEET:
        if not s.origin_name:
            s.origin_name = city.nodes[s.origin_node].name
        if not s.destination_name:
            s.destination_name = city.nodes[s.destination_node].name


def resolve_fleet_for(real_city: CityConfig, handbuilt_city: CityConfig,
                      fleet=None) -> List[Shipment]:
    """Snap each shipment's origin/destination from the hand-built city to the
    nearest node in the real (OSM) city. Reuses lat/lon from the hand-built
    nodes — no Nominatim needed.

    Returns a fresh list of Shipment objects with origin_node/destination_node
    set to OSM node IDs. origin_name/destination_name are preserved.
    """
    from pune_real import nearest_node     # local import to avoid cycles
    fleet = fleet if fleet is not None else DEMO_FLEET

    # Cache the resolution per hand-built node id so we only snap each unique
    # location once (a place can serve as origin for one shipment and dest for
    # another).
    snap_cache: Dict[str, str] = {}
    def snap(handbuilt_id: str) -> str:
        cached = snap_cache.get(handbuilt_id)
        if cached is not None:
            return cached
        n = handbuilt_city.nodes[handbuilt_id]
        rid, _dist = nearest_node(real_city, n.lat, n.lon)
        snap_cache[handbuilt_id] = rid
        return rid

    return [
        Shipment(
            id=s.id, vehicle_id=s.vehicle_id,
            origin_node=snap(s.origin_node),
            destination_node=snap(s.destination_node),
            deadline_min_from_now=s.deadline_min_from_now,
            priority=s.priority, load_kg=s.load_kg,
            origin_name=s.origin_name or handbuilt_city.nodes[s.origin_node].name,
            destination_name=s.destination_name or handbuilt_city.nodes[s.destination_node].name,
        )
        for s in fleet
    ]


def fleet_metadata(city_key: str = 'pune', fleet=None) -> dict:
    """Return a JSON-safe view of the (possibly already resolved) demo fleet."""
    fleet = fleet if fleet is not None else DEMO_FLEET
    return {
        'city_key': city_key,
        'fleet_day_start': f"{FLEET_DAY_START_HOUR}:00 AM",
        'loading_buffer_min': LOADING_BUFFER_MIN,
        'on_time_margin_min': ON_TIME_MARGIN_MIN,
        'shipments': [asdict(s) for s in fleet],
    }


# Stamp names onto the module-level DEMO_FLEET on import so anything that
# references DEMO_FLEET sees full place names without needing a second pass.
_populate_demo_names_from('pune')


# =============================================================================
# SIMULATION
# =============================================================================

def simulate_fleet(
    city_key: str,
    fleet: List[Shipment],
    query_date: date,
    rain_intensity: float,
    use_smart: bool,
    learning_store: Optional[LearningStore] = None,
) -> FleetResult:
    """
    Run all shipments through RoutingEngine, computing cascade times.

    Conditions are recomputed per shipment based on its computed start_time:
      - traffic_baseline derives from the time-of-day at start_time
      - dow_adjust is fixed across the fleet day (depends on weekday)
      - rain_intensity is fixed across the fleet day (passed in)
      - calendar events are date-bound and applied by IntegratedConditionsProvider
      - learning store is consulted per-edge as usual

    Vehicles run in parallel (each has its own clock starting at t=0).
    """
    city = get_city(city_key)
    engine = RoutingEngine(city)

    weekday_idx = query_date.weekday()
    dow_adj = _dow_adjust(weekday_idx)

    # Group shipments by vehicle, preserving the order they appear in `fleet`.
    by_vehicle: Dict[str, List[Shipment]] = {}
    for s in fleet:
        by_vehicle.setdefault(s.vehicle_id, []).append(s)

    results: List[ShipmentResult] = []

    for vehicle_id, vshipments in by_vehicle.items():
        prev_end = 0.0
        for seq, s in enumerate(vshipments):
            start_time = prev_end + (LOADING_BUFFER_MIN if seq > 0 else 0.0)

            # Per-shipment time-of-day → traffic baseline.
            shipment_hour_real = (FLEET_DAY_START_HOUR + start_time / 60.0) % 24
            hour_bucket, traffic_baseline = _traffic_for_hour(int(shipment_hour_real))
            traffic_level = max(0.0, min(1.0, traffic_baseline + dow_adj))

            cond = Conditions(
                traffic_level=traffic_level,
                rain_intensity=rain_intensity,
                has_event=False,    # let IntegratedConditionsProvider's calendar overlay fire
            )
            realtime = {(e.from_node, e.to_node): cond for e in city.edges}

            provider = IntegratedConditionsProvider(
                city=city,
                realtime_overrides=realtime,
                query_date=query_date,
                learning_store=learning_store,
            )

            path, _ = engine.find_route(
                s.origin_node, s.destination_node,
                provider.lookup,
                use_smart_weights=use_smart,
            )

            if not path:
                results.append(ShipmentResult(
                    shipment=s, sequence=seq,
                    start_time=start_time, end_time=start_time,
                    route_time=0.0, path_ids=[], path_names=[], coords=[],
                    margin=s.deadline_min_from_now - start_time,
                    status='LATE',
                    conditions={},
                    error=f"No path from {s.origin_node} to {s.destination_node}",
                ))
                # Cannot cascade without an end_time; treat as zero-duration so the
                # remaining shipments still run for visibility.
                prev_end = start_time
                continue

            route_time = engine.actual_travel_time(path, provider.lookup)
            end_time = start_time + route_time
            margin = s.deadline_min_from_now - end_time

            if end_time > s.deadline_min_from_now:
                status = 'LATE'
            elif margin < ON_TIME_MARGIN_MIN:
                status = 'AT_RISK'
            else:
                status = 'ON_TIME'

            results.append(ShipmentResult(
                shipment=s,
                sequence=seq,
                start_time=round(start_time, 1),
                end_time=round(end_time, 1),
                route_time=round(route_time, 1),
                path_ids=path,
                path_names=[city.nodes[n].name for n in path],
                coords=[[city.nodes[n].lat, city.nodes[n].lon] for n in path],
                margin=round(margin, 1),
                status=status,
                conditions={
                    'shipment_hour': round(shipment_hour_real, 2),
                    'hour_bucket': hour_bucket,
                    'traffic_level': round(traffic_level, 3),
                    'rain_intensity': round(rain_intensity, 3),
                },
            ))
            prev_end = end_time

    total_late = sum(1 for r in results if r.status == 'LATE')
    total_at_risk = sum(1 for r in results if r.status == 'AT_RISK')
    total_on_time = sum(1 for r in results if r.status == 'ON_TIME')
    total_minutes_late = sum(
        max(0.0, r.end_time - r.shipment.deadline_min_from_now) for r in results
    )

    return FleetResult(
        mode='smart' if use_smart else 'baseline',
        shipments=results,
        total_late=total_late,
        total_at_risk=total_at_risk,
        total_on_time=total_on_time,
        total_minutes_late=round(total_minutes_late, 1),
        fleet_day_start=f"{FLEET_DAY_START_HOUR}:00 AM",
    )


# =============================================================================
# JSON-SAFE SERIALIZATION (Flask jsonify can't handle dataclasses with nested
# dataclass fields cleanly, so we hand-build the dict)
# =============================================================================

def shipment_result_to_dict(r: ShipmentResult) -> dict:
    s = r.shipment
    return {
        'shipment_id': s.id,
        'vehicle_id': s.vehicle_id,
        'origin_node': s.origin_node,
        'destination_node': s.destination_node,
        'origin_name': s.origin_name or (r.path_names[0] if r.path_names else None),
        'destination_name': s.destination_name or (r.path_names[-1] if r.path_names else None),
        'deadline_min': s.deadline_min_from_now,
        'priority': s.priority,
        'load_kg': s.load_kg,
        'sequence': r.sequence,
        'start_time': r.start_time,
        'end_time': r.end_time,
        'route_time': r.route_time,
        'path_node_count': len(r.path_ids),     # easy to display for OSM-scale paths
        'path_distance_km': round(sum(
            # rough segment-by-segment haversine using coords (km)
            _hav_km(a, b) for a, b in zip(r.coords, r.coords[1:])
        ), 2) if len(r.coords) >= 2 else 0.0,
        'path_ids': r.path_ids,
        'path_names': r.path_names,
        'coords': r.coords,
        'margin': r.margin,
        'status': r.status,
        'conditions': r.conditions,
        'error': r.error,
    }


def _hav_km(a, b):
    import math
    R = 6371.0
    la1, lo1, la2, lo2 = math.radians(a[0]), math.radians(a[1]), math.radians(b[0]), math.radians(b[1])
    dlat = la2 - la1
    dlon = lo2 - lo1
    h = math.sin(dlat/2)**2 + math.cos(la1)*math.cos(la2)*math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(h))


def fleet_result_to_dict(fr: FleetResult) -> dict:
    return {
        'mode': fr.mode,
        'shipments': [shipment_result_to_dict(r) for r in fr.shipments],
        'total_late': fr.total_late,
        'total_at_risk': fr.total_at_risk,
        'total_on_time': fr.total_on_time,
        'total_minutes_late': fr.total_minutes_late,
        'fleet_day_start': fr.fleet_day_start,
    }


# =============================================================================
# CASCADE CALLOUTS — only LATE → ON_TIME, grouped by vehicle
# =============================================================================

def build_cascade_callouts(
    baseline: FleetResult,
    smart: FleetResult,
    city: CityConfig,
) -> List[dict]:
    """
    For each vehicle whose smart run rescues at least one LATE-in-baseline
    shipment, emit a single callout naming the trigger and listing the
    downstream recoveries. Only LATE→ON_TIME counts (AT_RISK→ON_TIME is real
    progress but not headline-worthy).
    """
    by_vehicle_b: Dict[str, List[ShipmentResult]] = {}
    by_vehicle_s: Dict[str, List[ShipmentResult]] = {}
    for r in baseline.shipments: by_vehicle_b.setdefault(r.shipment.vehicle_id, []).append(r)
    for r in smart.shipments:    by_vehicle_s.setdefault(r.shipment.vehicle_id, []).append(r)

    SEQ_WORD = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth']

    callouts = []
    for vid, b_ships in by_vehicle_b.items():
        s_ships = by_vehicle_s.get(vid, [])
        if not s_ships:
            continue

        recovered_idx = [
            i for i in range(min(len(b_ships), len(s_ships)))
            if b_ships[i].status == 'LATE' and s_ships[i].status == 'ON_TIME'
        ]
        if not recovered_idx:
            continue

        # Trigger = first shipment in the vehicle whose path differs between
        # baseline and smart AND whose baseline status is LATE. That's the
        # actual reroute the smart engine made.
        trigger_idx = None
        for i in range(min(len(b_ships), len(s_ships))):
            if b_ships[i].path_ids != s_ships[i].path_ids and b_ships[i].status == 'LATE':
                trigger_idx = i
                break
        if trigger_idx is None:
            # Cascade-only recovery: downstream shipment recovered because an
            # earlier (already-ON_TIME) shipment got faster. Pick the earliest
            # shipment whose path differs.
            for i in range(min(len(b_ships), len(s_ships))):
                if b_ships[i].path_ids != s_ships[i].path_ids:
                    trigger_idx = i
                    break
        if trigger_idx is None:
            continue   # no path difference at all; can't write a useful callout

        b_trigger = b_ships[trigger_idx]
        s_trigger = s_ships[trigger_idx]

        # Identify the most "interesting" leg of the smart reroute (the first
        # node-pair that differs from baseline).
        smart_corridor = _summarise_reroute(b_trigger, s_trigger)

        downstream_recovered = [i for i in recovered_idx if i > trigger_idx]
        own_recovered = trigger_idx in recovered_idx

        ship_word = SEQ_WORD[trigger_idx] if trigger_idx < len(SEQ_WORD) else f"#{trigger_idx + 1}"
        late_amount = round(-b_trigger.margin, 1) if b_trigger.margin < 0 else 0
        recovered_count = len(recovered_idx)
        recovered_count_word = (
            "1 shipment" if recovered_count == 1 else f"{recovered_count} shipments"
        )

        sentence = (
            f"Vehicle {vid} — {recovered_count_word} recovered. "
            f"{vid}'s {ship_word} shipment hit IPL highway congestion in baseline "
            f"(late by {late_amount} minutes), cascading the delay forward. "
            f"Smart routed the {ship_word} shipment via {smart_corridor} — "
            f"{vid}'s full day stays on schedule."
        )

        callouts.append({
            'vehicle_id': vid,
            'trigger_shipment': b_trigger.shipment.id,
            'trigger_late_min': late_amount,
            'recovered_shipment_ids': [b_ships[i].shipment.id for i in recovered_idx],
            'reroute_summary': smart_corridor,
            'sentence': sentence,
        })

    return callouts


def _summarise_reroute(baseline_r: ShipmentResult, smart_r: ShipmentResult) -> str:
    """Pick a short corridor name from the smart path that's NOT in the baseline.
    Falls back to "the alternate corridor" when the underlying graph has no
    readable node names (e.g. OSM intersections)."""
    base_set = set(baseline_r.path_ids)
    smart_unique_named = [
        smart_r.path_names[i]
        for i, nid in enumerate(smart_r.path_ids)
        if nid not in base_set and (smart_r.path_names[i] or '').strip()
    ]
    if len(smart_unique_named) >= 2:
        return f"{smart_unique_named[0]}\u2013{smart_unique_named[-1]}"
    if len(smart_unique_named) == 1:
        return smart_unique_named[0]
    return "the alternate corridor"
