"""
Cities Module - Multi-city Configuration
==========================================
Each city is a config bundle:
  - Road graph (nodes + edges)
  - Calendar events (city-specific)

Adding a new city = adding a new entry here. No code changes needed.
This is the architecture's generalization story.

Currently configured:
  - Pune (validated headline city, 8 nodes, 8 calendar events)
  - Mumbai (5 nodes, 3 calendar events)
  - Bangalore (5 nodes, 3 calendar events)

Roadmap (not built): Delhi, Kolkata, Vizag — same config pattern.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Tuple


# =============================================================================
# Shared data classes (imported by other modules too)
# =============================================================================

@dataclass
class Node:
    id: str
    name: str
    lat: float
    lon: float


@dataclass
class Edge:
    from_node: str
    to_node: str
    base_time_min: float
    distance_km: float
    road_type: str   # 'highway' | 'arterial' | 'local'


@dataclass
class CalendarEvent:
    name: str
    start_date: date
    end_date: date
    affected_edges: List[Tuple[str, str]]
    severity: float
    notes: str = ""


@dataclass
class CityConfig:
    """Everything the routing engine needs about a city."""
    key: str                                # 'pune', 'mumbai', 'bangalore'
    display_name: str
    default_origin: str                     # default WH node id
    default_destination: str                # default DEL node id
    nodes: Dict[str, Node]
    edges: List[Edge]
    events: List[CalendarEvent]


# =============================================================================
# PUNE - the validated headline city
# =============================================================================

PUNE_NODES = {
    # Original 8 (kept as-is — existing scenarios depend on these IDs)
    'WH':  Node('WH',  'Warehouse (Hinjewadi)', 18.5908, 73.7380),
    'A':   Node('A',   'Wakad Junction',         18.5970, 73.7660),
    'B':   Node('B',   'Aundh',                  18.5590, 73.8070),
    'C':   Node('C',   'Baner',                  18.5640, 73.7860),
    'D':   Node('D',   'Shivajinagar',           18.5300, 73.8470),
    'E':   Node('E',   'Koregaon Park',          18.5360, 73.8940),
    'F':   Node('F',   'Kalyani Nagar',          18.5490, 73.9070),
    'DEL': Node('DEL', 'Delivery (Viman Nagar)', 18.5680, 73.9140),
    # 11 well-known additions for manual-mode dropdown
    'SWG': Node('SWG', 'Swargate',               18.5008, 73.8567),
    'KTH': Node('KTH', 'Kothrud',                18.5074, 73.8077),
    'KRV': Node('KRV', 'Karve Nagar',            18.4868, 73.8175),
    'SNH': Node('SNH', 'Sinhagad Road',          18.4677, 73.8217),
    'DCN': Node('DCN', 'Deccan',                 18.5139, 73.8412),
    'CMP': Node('CMP', 'Camp',                   18.5189, 73.8780),
    'HAD': Node('HAD', 'Hadapsar',               18.5089, 73.9260),
    'MGP': Node('MGP', 'Magarpatta',             18.5141, 73.9279),
    'YWD': Node('YWD', 'Yerwada',                18.5530, 73.8745),
    'PIM': Node('PIM', 'Pimpri',                 18.6298, 73.7997),
    'CHW': Node('CHW', 'Chinchwad',              18.6279, 73.7805),
}

# Forward edges only — _bidirectional() below adds the reverses so manual mode
# can route between any pair. The forward set is what existing scenarios + the
# validation harness rely on, so it is preserved verbatim.
_PUNE_EDGES_FORWARD = [
    # --- original 11 (unchanged) -----------------------------------------
    Edge('WH', 'A',  base_time_min=8,  distance_km=4.5, road_type='highway'),
    Edge('A',  'C',  base_time_min=10, distance_km=5.0, road_type='arterial'),
    Edge('C',  'F',  base_time_min=18, distance_km=10.0, road_type='highway'),
    Edge('F',  'DEL',base_time_min=6,  distance_km=2.5, road_type='arterial'),
    Edge('A',  'B',  base_time_min=10, distance_km=4.0, road_type='arterial'),
    Edge('B',  'D',  base_time_min=11, distance_km=5.5, road_type='arterial'),
    Edge('D',  'E',  base_time_min=10, distance_km=5.0, road_type='arterial'),
    Edge('E',  'DEL',base_time_min=7,  distance_km=3.0, road_type='local'),
    Edge('C',  'D',  base_time_min=12, distance_km=6.0, road_type='arterial'),
    Edge('B',  'C',  base_time_min=6,  distance_km=2.5, road_type='local'),
    Edge('E',  'F',  base_time_min=5,  distance_km=2.0, road_type='local'),
    # --- NW corridor (Hinjewadi <-> Pimpri-Chinchwad <-> Wakad) ----------
    Edge('WH', 'PIM', base_time_min=12, distance_km=6.0, road_type='highway'),
    Edge('PIM','CHW', base_time_min=5,  distance_km=2.0, road_type='arterial'),
    Edge('CHW','A',   base_time_min=8,  distance_km=4.0, road_type='highway'),
    Edge('PIM','A',   base_time_min=10, distance_km=5.0, road_type='arterial'),
    # --- West-central (Baner <-> Kothrud <-> Karve Nagar <-> Sinhagad) ---
    Edge('C',  'KTH', base_time_min=12, distance_km=6.0, road_type='arterial'),
    Edge('KTH','KRV', base_time_min=6,  distance_km=3.0, road_type='arterial'),
    Edge('KTH','DCN', base_time_min=8,  distance_km=4.0, road_type='arterial'),
    Edge('KRV','SNH', base_time_min=8,  distance_km=4.0, road_type='arterial'),
    # --- Central spine (Deccan <-> Shivajinagar <-> Camp <-> Swargate) ---
    Edge('DCN','D',   base_time_min=5,  distance_km=2.5, road_type='arterial'),
    Edge('D',  'CMP', base_time_min=7,  distance_km=3.0, road_type='arterial'),
    Edge('CMP','SWG', base_time_min=6,  distance_km=2.5, road_type='arterial'),
    Edge('SWG','SNH', base_time_min=10, distance_km=5.0, road_type='arterial'),
    Edge('SWG','D',   base_time_min=8,  distance_km=3.5, road_type='arterial'),
    Edge('KRV','CMP', base_time_min=12, distance_km=6.0, road_type='arterial'),
    # --- Yerwada link ---------------------------------------------------
    Edge('E',  'YWD', base_time_min=5,  distance_km=2.5, road_type='local'),
    Edge('YWD','F',   base_time_min=6,  distance_km=3.0, road_type='arterial'),
    # --- SE quadrant (Hadapsar / Magarpatta) ----------------------------
    Edge('CMP','HAD', base_time_min=12, distance_km=6.0, road_type='arterial'),
    Edge('HAD','MGP', base_time_min=5,  distance_km=2.0, road_type='local'),
    Edge('MGP','E',   base_time_min=10, distance_km=5.0, road_type='arterial'),
    Edge('HAD','DEL', base_time_min=14, distance_km=7.0, road_type='arterial'),
    Edge('MGP','DEL', base_time_min=10, distance_km=4.5, road_type='arterial'),
    Edge('D',  'HAD', base_time_min=14, distance_km=7.0, road_type='arterial'),
]


def _bidirectional(edges: List[Edge]) -> List[Edge]:
    """Return edges + their reverses. Reverse keeps base_time/distance/type."""
    return edges + [
        Edge(e.to_node, e.from_node, e.base_time_min, e.distance_km, e.road_type)
        for e in edges
    ]


PUNE_EDGES = _bidirectional(_PUNE_EDGES_FORWARD)

PUNE_EVENTS = [
    CalendarEvent("Ganesh Chaturthi (Visarjan period)",
        date(2026, 9, 14), date(2026, 9, 24),
        [('B', 'D'), ('D', 'E'), ('A', 'B')], 0.85,
        "Mandals + processions block central arterials"),
    CalendarEvent("Diwali shopping rush",
        date(2026, 10, 28), date(2026, 11, 4),
        [('D', 'E'), ('E', 'DEL')], 0.6,
        "Tulsi Baug + Laxmi Road shopping crowds"),
    CalendarEvent("Monsoon peak (waterlogging risk)",
        date(2026, 7, 1), date(2026, 8, 31),
        [('C', 'F'), ('A', 'C')], 0.5,
        "Highway sections prone to flooding"),
    CalendarEvent("Pune International Marathon",
        date(2026, 12, 6), date(2026, 12, 6),
        [('A', 'B'), ('B', 'D'), ('D', 'E')], 0.95,
        "Multiple central roads closed 5 AM - 11 AM"),
    CalendarEvent("Board exam morning rush",
        date(2026, 3, 1), date(2026, 3, 25),
        [('A', 'C'), ('A', 'B')], 0.4,
        "Extended morning peak"),
    CalendarEvent("Republic Day parade",
        date(2026, 1, 26), date(2026, 1, 26),
        [('B', 'D'), ('D', 'E')], 0.7,
        "Parade closures until ~noon"),
    CalendarEvent("IPL match day at MCA Stadium",
        date(2026, 5, 2), date(2026, 5, 2),
        [('WH', 'A'), ('A', 'C'), ('C', 'F')], 0.85,
        "Stadium-bound traffic chokes northern highway"),
    CalendarEvent("Wedding season (evening peaks)",
        date(2026, 11, 15), date(2026, 12, 15),
        [('E', 'DEL'), ('F', 'DEL')], 0.45,
        "Banquet hall clusters in Viman Nagar"),
]

PUNE = CityConfig(
    key='pune', display_name='Pune',
    default_origin='WH', default_destination='DEL',
    nodes=PUNE_NODES, edges=PUNE_EDGES, events=PUNE_EVENTS,
)


# =============================================================================
# MUMBAI - generalization proof city #1
# =============================================================================

MUMBAI_NODES = {
    'WH':  Node('WH',  'Warehouse (Bhiwandi)',     19.2967, 73.0631),
    'M1':  Node('M1',  'Thane',                    19.2183, 72.9781),
    'M2':  Node('M2',  'Andheri East',             19.1136, 72.8697),
    'M3':  Node('M3',  'Bandra-Kurla Complex',     19.0680, 72.8678),
    'DEL': Node('DEL', 'Delivery (Worli)',         19.0176, 72.8170),
}

MUMBAI_EDGES = [
    # Eastern Express Highway route
    Edge('WH', 'M1', base_time_min=20, distance_km=18.0, road_type='highway'),
    Edge('M1', 'M2', base_time_min=25, distance_km=14.0, road_type='highway'),
    Edge('M2', 'M3', base_time_min=15, distance_km=8.0,  road_type='arterial'),
    Edge('M3', 'DEL',base_time_min=12, distance_km=6.0,  road_type='arterial'),
    # Alternative via Sion-Panvel (longer but bypasses central)
    Edge('M1', 'M3', base_time_min=35, distance_km=22.0, road_type='highway'),
    Edge('M2', 'DEL',base_time_min=22, distance_km=12.0, road_type='arterial'),
]

MUMBAI_EVENTS = [
    CalendarEvent("Monsoon (severe waterlogging)",
        date(2026, 6, 15), date(2026, 9, 30),
        [('M2', 'M3'), ('M3', 'DEL')], 0.7,
        "Hindmata + central Mumbai flooding common"),
    CalendarEvent("Ganpati Visarjan in Mumbai",
        date(2026, 9, 22), date(2026, 9, 24),
        [('M2', 'M3'), ('M3', 'DEL')], 0.9,
        "Lalbaug processions, central Mumbai gridlock"),
    CalendarEvent("Mumbai Marathon",
        date(2026, 1, 18), date(2026, 1, 18),
        [('M3', 'DEL'), ('M2', 'DEL')], 0.95,
        "Marine Drive + central roads closed early morning"),
]

MUMBAI = CityConfig(
    key='mumbai', display_name='Mumbai',
    default_origin='WH', default_destination='DEL',
    nodes=MUMBAI_NODES, edges=MUMBAI_EDGES, events=MUMBAI_EVENTS,
)


# =============================================================================
# BANGALORE - generalization proof city #2
# =============================================================================

BANGALORE_NODES = {
    'WH':  Node('WH',  'Warehouse (Hoskote)',         13.0710, 77.7949),
    'B1':  Node('B1',  'Whitefield',                  12.9698, 77.7500),
    'B2':  Node('B2',  'Marathahalli',                12.9591, 77.6974),
    'B3':  Node('B3',  'Silk Board Junction',         12.9166, 77.6224),
    'DEL': Node('DEL', 'Delivery (Electronic City)',  12.8456, 77.6603),
}

BANGALORE_EDGES = [
    # ORR (Outer Ring Road) corridor - the famous tech traffic route
    Edge('WH', 'B1', base_time_min=18, distance_km=12.0, road_type='highway'),
    Edge('B1', 'B2', base_time_min=20, distance_km=7.0,  road_type='arterial'),
    Edge('B2', 'B3', base_time_min=30, distance_km=12.0, road_type='arterial'),
    Edge('B3', 'DEL',base_time_min=20, distance_km=14.0, road_type='highway'),
    # Alternative via Sarjapur (longer, often faster during ORR rush)
    Edge('B1', 'B3', base_time_min=35, distance_km=18.0, road_type='arterial'),
    Edge('B2', 'DEL',base_time_min=30, distance_km=22.0, road_type='arterial'),
]

BANGALORE_EVENTS = [
    CalendarEvent("Tech park rush hour (chronic ORR congestion)",
        date(2026, 1, 1), date(2026, 12, 31),
        [('B1', 'B2'), ('B2', 'B3')], 0.55,
        "Daily 6-10 PM ORR jam, chronic year-round"),
    CalendarEvent("Karnataka Bandh (general strike)",
        date(2026, 2, 14), date(2026, 2, 14),
        [('B2', 'B3'), ('B3', 'DEL')], 0.9,
        "Citywide bandh, central roads avoided"),
    CalendarEvent("Bangalore Tech Summit",
        date(2026, 11, 18), date(2026, 11, 20),
        [('B1', 'B2'), ('B2', 'B3')], 0.65,
        "Whitefield + ORR congestion from event traffic"),
]

BANGALORE = CityConfig(
    key='bangalore', display_name='Bangalore',
    default_origin='WH', default_destination='DEL',
    nodes=BANGALORE_NODES, edges=BANGALORE_EDGES, events=BANGALORE_EVENTS,
)


# =============================================================================
# CITY REGISTRY
# =============================================================================

CITIES: Dict[str, CityConfig] = {
    'pune':      PUNE,
    'mumbai':    MUMBAI,
    'bangalore': BANGALORE,
}


def get_city(key: str) -> CityConfig:
    if key not in CITIES:
        raise KeyError(f"Unknown city '{key}'. Available: {list(CITIES.keys())}")
    return CITIES[key]


def list_cities() -> List[str]:
    return list(CITIES.keys())


# =============================================================================
# EVENT LOOKUPS (operate on a city's events)
# =============================================================================

def get_active_events(city: CityConfig, query_date: date) -> List[CalendarEvent]:
    return [e for e in city.events if e.start_date <= query_date <= e.end_date]


def get_disrupted_edges(city: CityConfig, query_date: date) -> dict:
    """edge_pair -> (severity, event_name) for the worst event affecting each edge.

    Events are applied bidirectionally — a real road closure or major event
    affects both directions of travel on the listed edges, so we mirror each
    (a, b) onto (b, a). Without this, manual-mode routes that approach an
    affected node from the opposite direction would silently bypass the event."""
    disrupted = {}
    for event in get_active_events(city, query_date):
        for a, b in event.affected_edges:
            for ep in ((a, b), (b, a)):
                existing_sev = disrupted.get(ep, (0.0, ""))[0]
                if event.severity > existing_sev:
                    disrupted[ep] = (event.severity, event.name)
    return disrupted


if __name__ == "__main__":
    # Demo: show available cities and their config sizes
    print("Configured cities:")
    print(f"{'City':<12} {'Nodes':>6} {'Edges':>6} {'Events':>7}")
    print("-" * 35)
    for key in list_cities():
        c = get_city(key)
        print(f"{c.display_name:<12} {len(c.nodes):>6} {len(c.edges):>6} {len(c.events):>7}")

    print("\nMumbai active events on 2026-09-23 (Ganpati Visarjan):")
    for e in get_active_events(get_city('mumbai'), date(2026, 9, 23)):
        print(f"  - {e.name} (severity {e.severity:.0%})")

    print("\nBangalore active events on 2026-02-14 (Bandh):")
    for e in get_active_events(get_city('bangalore'), date(2026, 2, 14)):
        print(f"  - {e.name} (severity {e.severity:.0%})")
