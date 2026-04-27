"""
Pune (Real Graph) — OpenStreetMap-derived CityConfig
======================================================
Builds a CityConfig from osmnx for the central Pune bbox, then caches the
*converted* CityConfig (not the raw MultiDiGraph). First run downloads from
OSM (~30-90s), subsequent runs unpickle in <2s.

Used by manual mode and fleet dispatch only — scenario mode and the
validation harness keep using the hand-built PUNE in cities.py.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import time
from typing import Tuple, Optional

from cities import CityConfig, Node, Edge, CalendarEvent, CITIES


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# bbox = (west, south, east, north). Extended slightly from the tight central
# bbox so all hand-built node coords (SNH south, PIM/CHW north) are inside —
# matters because fleet shipments and manual queries reference these places.
BBOX = (73.72, 18.46, 73.96, 18.64)

# Bump SPEED_VERSION when the speed_for_highway() / road_type mapping changes,
# so the cache hash invalidates automatically.
SPEED_VERSION = 1

NETWORK_TYPE = "drive"

CACHE_DIR = os.path.dirname(os.path.abspath(__file__))


# -----------------------------------------------------------------------------
# Tag → speed and road_type
# -----------------------------------------------------------------------------

# km/h baselines per spec. Default (unknown tag) is 18 km/h.
_SPEED_KPH = {
    'motorway': 60, 'motorway_link': 60,
    'trunk':    60, 'trunk_link':    60,
    'primary':  50, 'primary_link':  50,
    'secondary': 40, 'secondary_link': 40,
    'tertiary': 30, 'tertiary_link': 30,
    'residential': 20, 'unclassified': 20, 'service': 20,
    'living_street': 15, 'pedestrian': 10,
}

_ROAD_TYPE = {
    'motorway': 'highway', 'motorway_link': 'highway',
    'trunk':    'highway', 'trunk_link':    'highway',
    'primary':  'arterial', 'primary_link':  'arterial',
    'secondary': 'arterial', 'secondary_link': 'arterial',
}


def _normalise_tag(tag) -> str:
    """OSM `highway` can be str or list (osmnx merges parallel ways).
    Pick the first element of the list, lowercase the whole thing."""
    if isinstance(tag, list):
        tag = tag[0] if tag else ''
    return (tag or '').lower()


def _speed_for_tag(tag: str) -> int:
    return _SPEED_KPH.get(tag, 18)


def _road_type_for_tag(tag: str) -> str:
    return _ROAD_TYPE.get(tag, 'local')


# -----------------------------------------------------------------------------
# Cache filename — content-hashed so bbox / version changes auto-invalidate
# -----------------------------------------------------------------------------

def _cache_path(bbox: Tuple[float, float, float, float]) -> str:
    key = f"bbox={bbox}|speed_v={SPEED_VERSION}|network={NETWORK_TYPE}"
    h = hashlib.sha1(key.encode()).hexdigest()[:8]
    return os.path.join(CACHE_DIR, f"pune_real_{h}.pkl")


# -----------------------------------------------------------------------------
# OSM → CityConfig
# -----------------------------------------------------------------------------

def _build_from_osm(bbox: Tuple[float, float, float, float]) -> CityConfig:
    """Download from OSM and convert to a CityConfig. Heavy — only call when
    the cache is missing."""
    import osmnx as ox  # lazy import; avoids osmnx cost when cache is hot

    print(f"  [pune_real] downloading OSM bbox={bbox} (network_type={NETWORK_TYPE})...")
    t0 = time.time()
    # osmnx 2.x: graph_from_bbox(bbox=(left, bottom, right, top), ...)
    G = ox.graph_from_bbox(bbox=bbox, network_type=NETWORK_TYPE)
    print(f"  [pune_real] download done in {time.time()-t0:.1f}s "
          f"({len(G.nodes):,} nodes, {len(G.edges):,} OSM edges)")

    # ---- Nodes ----
    nodes_dict = {}
    for osmid, data in G.nodes(data=True):
        nid = str(osmid)
        nodes_dict[nid] = Node(
            id=nid,
            name='',     # OSM nodes are mostly unnamed intersections
            lat=float(data['y']),
            lon=float(data['x']),
        )

    # ---- Edges ----
    # Walk all (u, v, k) edges. Respect oneway. Flatten parallel ways by
    # keeping the shortest length per (u, v) pair (saves time in A*'s inner
    # adjacency loop).
    best_for_pair = {}   # (u, v) -> (length_m, tag, oneway)
    for u, v, _k, data in G.edges(keys=True, data=True):
        length = float(data.get('length') or 0.0)
        if length <= 0:
            continue
        tag = _normalise_tag(data.get('highway'))
        oneway = bool(data.get('oneway'))
        key = (str(u), str(v))
        prev = best_for_pair.get(key)
        if prev is None or length < prev[0]:
            best_for_pair[key] = (length, tag, oneway)

    # Convert to Edge dataclass instances. For two-way roads, add the reverse
    # if it isn't already present from another OSM edge.
    edges = []
    seen_pairs = set()
    for (u, v), (length, tag, oneway) in best_for_pair.items():
        if (u, v) in seen_pairs:
            continue
        speed_kph = _speed_for_tag(tag)
        base_time = (length / 1000.0) / speed_kph * 60.0
        rt = _road_type_for_tag(tag)
        edges.append(Edge(
            from_node=u, to_node=v,
            base_time_min=base_time,
            distance_km=length / 1000.0,
            road_type=rt,
        ))
        seen_pairs.add((u, v))

        # Two-way: ensure the reverse exists too. If OSM already had an
        # explicit (v, u) edge it will appear in best_for_pair and get added
        # in its own iteration, so only synthesise one if it's missing.
        if not oneway and (v, u) not in best_for_pair and (v, u) not in seen_pairs:
            edges.append(Edge(
                from_node=v, to_node=u,
                base_time_min=base_time,
                distance_km=length / 1000.0,
                road_type=rt,
            ))
            seen_pairs.add((v, u))

    # Pune (Real) has no calendar events for now — events reference hand-built
    # node IDs that don't exist in this graph, and adding them would require a
    # separate effort to map old IDs → OSM nodes.
    city = CityConfig(
        key='pune_real',
        display_name='Pune (Real Graph)',
        default_origin='',          # set after we resolve a place name
        default_destination='',
        nodes=nodes_dict,
        edges=edges,
        events=[],
    )
    return city


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------

def get_pune_real(bbox: Optional[Tuple[float, float, float, float]] = None,
                  force_rebuild: bool = False) -> CityConfig:
    """Load Pune (Real) from cache; build + cache if missing.

    The default bbox is module-level BBOX. Override only for testing — the
    cache hash changes with bbox, so cache files don't collide.
    """
    bbox = bbox or BBOX
    path = _cache_path(bbox)

    if force_rebuild and os.path.exists(path):
        os.remove(path)

    if os.path.exists(path):
        t0 = time.time()
        with open(path, 'rb') as f:
            city = pickle.load(f)
        print(f"  [pune_real] loaded cache in {time.time()-t0:.1f}s "
              f"({len(city.nodes):,} nodes, {len(city.edges):,} edges)")
        CITIES[city.key] = city
        return city

    city = _build_from_osm(bbox)
    with open(path, 'wb') as f:
        pickle.dump(city, f)
    sz = os.path.getsize(path) / (1024 * 1024)
    print(f"  [pune_real] cached to {os.path.basename(path)} ({sz:.1f} MB) — "
          f"{len(city.nodes):,} nodes, {len(city.edges):,} edges")
    CITIES[city.key] = city
    return city


# -----------------------------------------------------------------------------
# Snap a (lat, lon) to the nearest graph node — for manual mode geocoding and
# fleet shipment resolution. Implemented directly (haversine over all nodes)
# so we don't have to keep the raw OSM graph in memory after the conversion.
# -----------------------------------------------------------------------------

def nearest_node(city: CityConfig, lat: float, lon: float) -> Tuple[str, float]:
    """Return (node_id, distance_m) for the nearest node to (lat, lon)."""
    import math
    R = 6371000.0
    lat_r = math.radians(lat)
    cos_lat = math.cos(lat_r)
    best_id = None
    best_d2 = float('inf')
    # Cheap squared-equirectangular approx — fast and accurate enough for
    # snapping queries to the nearest intersection. Final distance is computed
    # with full haversine on the winner only.
    for nid, n in city.nodes.items():
        dx = (n.lon - lon) * cos_lat
        dy = (n.lat - lat)
        d2 = dx*dx + dy*dy
        if d2 < best_d2:
            best_d2 = d2
            best_id = nid
    # Haversine for the winner.
    n = city.nodes[best_id]
    lat1, lat2 = math.radians(n.lat), lat_r
    dlat = lat2 - lat1
    dlon = math.radians(lon - n.lon)
    h = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    dist_m = 2 * R * math.asin(math.sqrt(h))
    return best_id, dist_m


# -----------------------------------------------------------------------------
# Calendar event translation: hand-built (WH→A→C→F …) → real-graph corridors
# -----------------------------------------------------------------------------

def translate_handbuilt_events(handbuilt_city: CityConfig,
                                real_city: CityConfig) -> list:
    """For each event on the hand-built city, find the corresponding corridor
    in the real graph by routing baseline shortest paths between the snapped
    endpoints and marking every OSM edge along the way as affected.

    Without this step, calendar events would never fire on pune_real (the
    affected_edges in PUNE.events reference hand-built node IDs that don't
    exist in OSM), and the fleet's IPL-day cascade demo would be dead.
    """
    from multicity_router import RoutingEngine, Conditions   # local; avoids cycle
    engine = RoutingEngine(real_city)
    _no_cond = lambda a, b: Conditions()

    snap_cache: dict = {}
    def snap(hid: str) -> str:
        s = snap_cache.get(hid)
        if s is None:
            n = handbuilt_city.nodes[hid]
            s, _ = nearest_node(real_city, n.lat, n.lon)
            snap_cache[hid] = s
        return s

    out = []
    for ev in handbuilt_city.events:
        translated_edges: list = []
        for hb_from, hb_to in ev.affected_edges:
            osm_from, osm_to = snap(hb_from), snap(hb_to)
            path, _ = engine.find_route(osm_from, osm_to, _no_cond, use_smart_weights=False)
            for i in range(len(path) - 1):
                translated_edges.append((path[i], path[i+1]))
        out.append(CalendarEvent(
            name=ev.name,
            start_date=ev.start_date,
            end_date=ev.end_date,
            affected_edges=translated_edges,
            severity=ev.severity,
            notes=ev.notes,
        ))
    return out


if __name__ == '__main__':
    # Smoke-test entry point: python pune_real.py
    city = get_pune_real()
    print(f"\nPune (Real) summary:")
    print(f"  bbox: {BBOX}")
    print(f"  nodes: {len(city.nodes):,}")
    print(f"  edges: {len(city.edges):,}")
    sample_lat, sample_lon = 18.5008, 73.8567   # Swargate
    nid, dm = nearest_node(city, sample_lat, sample_lon)
    print(f"  nearest to Swargate ({sample_lat},{sample_lon}): {nid} @ {dm:.0f}m")
