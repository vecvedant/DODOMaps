"""
Multi-City Smart Routing System
================================
Takes any configured Indian city, runs validation + demos.
Same architecture as v2 - just parameterized by city.

Currently configured cities: pune, mumbai, bangalore
Adding a new city = adding a CityConfig in cities.py. No code changes here.

Usage:
  python multicity_router.py            # runs all 3 cities + comparison
  python multicity_router.py pune       # runs validation for one city
"""

import heapq
import math
import random
import statistics
import sys
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Tuple, Optional, Callable

from cities import (
    Node, Edge, CalendarEvent, CityConfig,
    get_city, list_cities,
    get_active_events, get_disrupted_edges,
)
from learning_loop import LearningStore


# =============================================================================
# 1. CONDITIONS
# =============================================================================

@dataclass
class Conditions:
    traffic_level: float = 0.0
    rain_intensity: float = 0.0
    has_event: bool = False
    event_severity: float = 0.0
    event_name: str = ""
    historical_delay_factor: float = 0.0


# =============================================================================
# 2. SMART WEIGHT (unchanged - same coefficients work across cities)
# =============================================================================

def compute_smart_weight(edge: Edge, cond: Conditions) -> float:
    base = edge.base_time_min
    traffic_mult = 0.8 if edge.road_type == 'highway' else 0.5
    traffic_impact = base * cond.traffic_level * traffic_mult
    weather_impact = base * (cond.rain_intensity ** 1.5) * 0.35
    if cond.has_event:
        severity = cond.event_severity if cond.event_severity > 0 else 0.30
        event_impact = base * severity
    else:
        event_impact = 0.0
    historical_impact = base * cond.historical_delay_factor
    return base + traffic_impact + weather_impact + event_impact + historical_impact


# =============================================================================
# 3. ROUTING ENGINE (city-agnostic)
# =============================================================================

class RoutingEngine:
    def __init__(self, city: CityConfig):
        self.city = city
        self.nodes = city.nodes
        self.adj: Dict[str, List[Edge]] = {n: [] for n in city.nodes}
        for e in city.edges:
            self.adj[e.from_node].append(e)

    def haversine_km(self, a: Node, b: Node) -> float:
        R = 6371.0
        lat1, lat2 = math.radians(a.lat), math.radians(b.lat)
        dlat = lat2 - lat1
        dlon = math.radians(b.lon - a.lon)
        h = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
        return 2 * R * math.asin(math.sqrt(h))

    def find_route(self, start, goal, conditions_lookup, use_smart_weights=True):
        # Optimized A*:
        #   - heuristic-to-goal cached per node (was recomputed every relax)
        #   - parent pointers for O(N) path reconstruction (was O(N^2) via path-in-heap)
        #   - heap entries are scalar tuples, not list-carrying — much smaller pushes
        # Algorithm itself unchanged: same admissible heuristic, same relax rule,
        # so validation numbers stay identical.
        goal_node = self.nodes[goal]
        nodes = self.nodes
        adj = self.adj
        haversine = self.haversine_km

        h_cache: Dict[str, float] = {}
        def h_to_goal(nid: str) -> float:
            v = h_cache.get(nid)
            if v is None:
                v = haversine(nodes[nid], goal_node) / 40.0 * 60.0
                h_cache[nid] = v
            return v

        counter = 0
        open_set = [(h_to_goal(start), counter, start)]
        best_g: Dict[str, float] = {start: 0.0}
        parent: Dict[str, Optional[str]] = {start: None}

        while open_set:
            _, _, current = heapq.heappop(open_set)
            if current == goal:
                path = []
                n = current
                while n is not None:
                    path.append(n)
                    n = parent[n]
                path.reverse()
                return path, best_g[goal]
            g = best_g[current]
            for edge in adj.get(current, ()):
                if use_smart_weights:
                    weight = compute_smart_weight(edge, conditions_lookup(edge.from_node, edge.to_node))
                else:
                    weight = edge.base_time_min
                tentative_g = g + weight
                to_node = edge.to_node
                if tentative_g < best_g.get(to_node, float('inf')):
                    best_g[to_node] = tentative_g
                    parent[to_node] = current
                    counter += 1
                    heapq.heappush(open_set, (tentative_g + h_to_goal(to_node), counter, to_node))
        return [], float('inf')

    def actual_travel_time(self, path, conditions_lookup):
        total = 0.0
        for a, b in zip(path, path[1:]):
            edge = next(e for e in self.adj[a] if e.to_node == b)
            total += compute_smart_weight(edge, conditions_lookup(a, b))
        return total


# =============================================================================
# 4. INTEGRATED CONDITIONS PROVIDER (city-aware)
# =============================================================================

class IntegratedConditionsProvider:
    def __init__(
        self,
        city: CityConfig,
        realtime_overrides: Dict[Tuple[str, str], Conditions] = None,
        query_date: date = None,
        learning_store: LearningStore = None,
        learning_namespace: str = "",
    ):
        self.city = city
        self.realtime = realtime_overrides or {}
        self.query_date = query_date or date(2026, 4, 26)
        self.learning_store = learning_store
        # Namespace per city so Pune learning doesn't pollute Mumbai
        self.ns = learning_namespace or city.key
        self.calendar_disruptions = get_disrupted_edges(city, self.query_date)

    def lookup(self, from_node, to_node):
        edge = (from_node, to_node)
        c = self.realtime.get(edge, Conditions())

        # Calendar overlay
        if edge in self.calendar_disruptions and not c.has_event:
            sev, name = self.calendar_disruptions[edge]
            c = Conditions(
                traffic_level=c.traffic_level,
                rain_intensity=c.rain_intensity,
                has_event=True,
                event_severity=sev,
                event_name=name,
                historical_delay_factor=c.historical_delay_factor,
            )

        # Learning overlay (city-namespaced)
        if self.learning_store and c.historical_delay_factor == 0:
            ns_edge = (f"{self.ns}:{from_node}", f"{self.ns}:{to_node}")
            learned = self.learning_store.get_delay_factor(*ns_edge)
            if learned > 0:
                c = Conditions(
                    traffic_level=c.traffic_level,
                    rain_intensity=c.rain_intensity,
                    has_event=c.has_event,
                    event_severity=c.event_severity,
                    event_name=c.event_name,
                    historical_delay_factor=learned,
                )
        return c


# =============================================================================
# 5. PER-CITY LEARNING WARM-UP
# =============================================================================

CITY_CHRONIC_DELAYS = {
    'pune': {
        ('B', 'D'): 0.25,
        ('C', 'F'): 0.15,
        ('D', 'E'): 0.10,
    },
    'mumbai': {
        ('M2', 'M3'): 0.30,    # central Mumbai chronic
        ('M1', 'M2'): 0.20,    # Eastern Express known slow
    },
    'bangalore': {
        ('B1', 'B2'): 0.35,    # ORR famously chronic
        ('B2', 'B3'): 0.40,    # Silk Board approach
    },
}


def warm_up_learning_for_city(store: LearningStore, city: CityConfig, num_days: int = 90):
    """Simulate trip history for one city, namespaced so cities don't interfere."""
    random.seed(123 + hash(city.key) % 1000)
    chronic = CITY_CHRONIC_DELAYS.get(city.key, {})
    base_times = {(e.from_node, e.to_node): e.base_time_min for e in city.edges}
    edge_keys = list(base_times.keys())

    for day in range(num_days):
        for _ in range(5):
            edge = random.choice(edge_keys)
            base = base_times[edge]
            chronic_factor = chronic.get(edge, 0.0)
            actual = base * (1 + chronic_factor + random.uniform(-0.05, 0.05))
            ns_edge = (f"{city.key}:{edge[0]}", f"{city.key}:{edge[1]}")
            store.update_from_trip(ns_edge, actual, base)


# =============================================================================
# 6. SCENARIOS PER CITY
# =============================================================================

@dataclass
class Scenario:
    name: str
    query_date: date
    realtime: Dict[Tuple[str, str], Conditions] = field(default_factory=dict)


def generate_pune_scenarios(seed=42) -> List[Scenario]:
    """Pune has the full 20-scenario validation - this is our headline city."""
    random.seed(seed)
    s = []
    normal = date(2026, 4, 26)
    s.append(Scenario("Clear weekday morning", normal))
    s.append(Scenario("Heavy rain on highway (north)", normal,
        {('C','F'): Conditions(rain_intensity=0.9, traffic_level=0.5)}))
    s.append(Scenario("Ganpati Visarjan (calendar)", date(2026,9,18)))
    s.append(Scenario("Rush hour on highway", normal,
        {('C','F'): Conditions(traffic_level=0.85),
         ('A','C'): Conditions(traffic_level=0.5)}))
    s.append(Scenario("Accident on C-F highway", normal,
        {('C','F'): Conditions(has_event=True, traffic_level=0.95)}))
    sc = Scenario("Light rain citywide", normal)
    for p in [('WH','A'),('A','C'),('A','B'),('C','F'),('B','D'),('D','E'),('E','DEL'),('F','DEL')]:
        sc.realtime[p] = Conditions(rain_intensity=0.3, traffic_level=0.2)
    s.append(sc)
    s.append(Scenario("Monsoon day + rain spike", date(2026,7,20),
        {('C','F'): Conditions(rain_intensity=0.6, traffic_level=0.4)}))
    s.append(Scenario("Weekend (light)", normal))
    s.append(Scenario("Storm on north corridor", normal,
        {('A','C'): Conditions(rain_intensity=0.85, traffic_level=0.4),
         ('C','F'): Conditions(rain_intensity=0.9, traffic_level=0.6)}))
    s.append(Scenario("Roadblock near delivery", normal,
        {('F','DEL'): Conditions(has_event=True, traffic_level=0.9)}))
    s.append(Scenario("Marathon day (calendar)", date(2026,12,6)))
    s.append(Scenario("IPL match day + light rain", date(2026,5,2),
        {('A','C'): Conditions(rain_intensity=0.3)}))
    s.append(Scenario("Storm + accident on highway", normal,
        {('C','F'): Conditions(rain_intensity=0.8, has_event=True, traffic_level=0.85)}))
    s.append(Scenario("Festival on central only", normal,
        {('A','B'): Conditions(has_event=True, traffic_level=0.8),
         ('B','D'): Conditions(has_event=True)}))
    s.append(Scenario("Random midday + highway impact", normal,
        {('C','F'): Conditions(traffic_level=random.uniform(0.5,0.8)),
         ('A','C'): Conditions(traffic_level=random.uniform(0.3,0.6))}))
    base_e = [('WH','A'),('A','C'),('C','F'),('F','DEL')]
    other_e = [('A','B'),('B','D'),('D','E'),('E','DEL'),('C','D'),('B','C'),('E','F')]
    for i in range(16, 21):
        sc = Scenario(f"Mixed conditions #{i-15}", normal)
        for p in random.sample(base_e, random.randint(1,2)):
            sc.realtime[p] = Conditions(
                traffic_level=random.uniform(0.3,0.8),
                rain_intensity=random.uniform(0,0.6),
                has_event=random.random() < 0.3,
                historical_delay_factor=random.uniform(0,0.4))
        for p in random.sample(other_e, random.randint(1,2)):
            sc.realtime[p] = Conditions(
                traffic_level=random.uniform(0,0.5),
                rain_intensity=random.uniform(0,0.4))
        s.append(sc)
    return s


def generate_mumbai_scenarios() -> List[Scenario]:
    """Mumbai has fewer scenarios - this is a generalization proof, not a deep validation."""
    normal = date(2026, 4, 26)
    return [
        Scenario("Clear weekday", normal),
        Scenario("Monsoon waterlogging (calendar)", date(2026, 7, 15)),
        Scenario("Ganpati Visarjan (calendar)", date(2026, 9, 23)),
        Scenario("Mumbai Marathon (calendar)", date(2026, 1, 18)),
        Scenario("Heavy rain + central traffic", normal,
            {('M2','M3'): Conditions(rain_intensity=0.7, traffic_level=0.6)}),
        Scenario("Accident on Eastern Express", normal,
            {('M1','M2'): Conditions(has_event=True, traffic_level=0.9)}),
    ]


def generate_bangalore_scenarios() -> List[Scenario]:
    """Bangalore: ORR-focused scenarios."""
    normal = date(2026, 4, 26)
    return [
        Scenario("Clear weekday morning", normal),
        Scenario("Tech park evening rush", normal,
            {('B1','B2'): Conditions(traffic_level=0.85),
             ('B2','B3'): Conditions(traffic_level=0.9)}),
        Scenario("Karnataka Bandh (calendar)", date(2026, 2, 14)),
        Scenario("Tech Summit days (calendar)", date(2026, 11, 19)),
        Scenario("Silk Board jam + light rain", normal,
            {('B2','B3'): Conditions(traffic_level=0.85, rain_intensity=0.3)}),
        Scenario("Whitefield protest + ORR jam", normal,
            {('B1','B2'): Conditions(has_event=True, traffic_level=0.85)}),
    ]


SCENARIO_GENERATORS = {
    'pune':      generate_pune_scenarios,
    'mumbai':    generate_mumbai_scenarios,
    'bangalore': generate_bangalore_scenarios,
}


# =============================================================================
# 7. VALIDATION FOR ANY CITY
# =============================================================================

def validate_city(city_key: str, store: LearningStore, verbose: bool = True):
    city = get_city(city_key)
    if verbose:
        print(f"\n{'=' * 82}")
        print(f"  {city.display_name.upper()} - VALIDATION ({len(city.nodes)} nodes, {len(city.edges)} edges)")
        print(f"{'=' * 82}")

    engine = RoutingEngine(city)
    scenarios = SCENARIO_GENERATORS[city_key]()

    if verbose:
        print(f"{'Scenario':<42} {'Baseline':>10} {'Smart':>10} {'Saved':>10}")
        print("-" * 82)

    base_times, smart_times = [], []
    for s in scenarios:
        provider = IntegratedConditionsProvider(
            city=city, realtime_overrides=s.realtime,
            query_date=s.query_date, learning_store=store,
        )
        base_path, _ = engine.find_route(city.default_origin, city.default_destination,
                                         provider.lookup, use_smart_weights=False)
        base_actual = engine.actual_travel_time(base_path, provider.lookup)
        smart_path, _ = engine.find_route(city.default_origin, city.default_destination,
                                          provider.lookup, use_smart_weights=True)
        smart_actual = engine.actual_travel_time(smart_path, provider.lookup)
        saved = (base_actual - smart_actual) / base_actual * 100 if base_actual > 0 else 0
        if verbose:
            print(f"{s.name:<42} {base_actual:>8.1f}m {smart_actual:>8.1f}m {saved:>8.1f}%")
        base_times.append(base_actual)
        smart_times.append(smart_actual)

    avg_b = statistics.mean(base_times)
    avg_s = statistics.mean(smart_times)
    avg_saved = (avg_b - avg_s) / avg_b * 100
    best = max((b-s)/b*100 for b,s in zip(base_times, smart_times))

    if verbose:
        print("-" * 82)
        print(f"{'AVERAGE (' + str(len(scenarios)) + ' scenarios)':<42} {avg_b:>8.1f}m {avg_s:>8.1f}m {avg_saved:>8.1f}%")
        print(f"\n>>> {city.display_name}: {avg_saved:.1f}% avg saved, {best:.1f}% best case")

    return {'city': city.display_name, 'scenarios': len(scenarios),
            'avg_saved': avg_saved, 'best_case': best, 'avg_baseline': avg_b, 'avg_smart': avg_s}


def explain_route_for_city(city_key: str, scenario: Scenario, store: LearningStore):
    city = get_city(city_key)
    engine = RoutingEngine(city)
    provider = IntegratedConditionsProvider(
        city=city, realtime_overrides=scenario.realtime,
        query_date=scenario.query_date, learning_store=store,
    )
    print(f"\n>>> {city.display_name}: {scenario.name}  (date: {scenario.query_date})")
    print("-" * 70)

    cal_events = get_active_events(city, scenario.query_date)
    if cal_events:
        print("Calendar events active today:")
        for e in cal_events:
            print(f"  - {e.name} (severity {e.severity:.0%})")

    base_path, _ = engine.find_route(city.default_origin, city.default_destination,
                                     provider.lookup, use_smart_weights=False)
    smart_path, _ = engine.find_route(city.default_origin, city.default_destination,
                                      provider.lookup, use_smart_weights=True)
    base_actual = engine.actual_travel_time(base_path, provider.lookup)
    smart_actual = engine.actual_travel_time(smart_path, provider.lookup)

    def names(p): return ' -> '.join(city.nodes[x].name for x in p)
    print(f"\nBaseline: {names(base_path)}")
    print(f"  Actual: {base_actual:.1f} min")
    print(f"\nSmart:    {names(smart_path)}")
    print(f"  Actual: {smart_actual:.1f} min")

    if smart_path != base_path:
        print(f"\n>>> REROUTED to save {base_actual - smart_actual:.1f} minutes.")
    else:
        print(f"\n>>> Kept baseline route ({base_actual - smart_actual:+.1f} min difference).")


# =============================================================================
# 8. MAIN
# =============================================================================

def run_all_cities():
    print("Warming up learning loop with simulated trip data for 3 cities...")
    store = LearningStore()
    store.reset()
    for key in list_cities():
        warm_up_learning_for_city(store, get_city(key), num_days=90)
    print(f"Warm-up complete: {sum(s.trips_observed for s in store.stats.values())} trips across all cities.\n")

    # Validate each city
    results = []
    for key in list_cities():
        results.append(validate_city(key, store, verbose=True))

    # Multi-city comparison summary
    print(f"\n{'=' * 82}")
    print(f"  MULTI-CITY COMPARISON")
    print(f"{'=' * 82}")
    print(f"{'City':<14} {'Scenarios':>10} {'Avg Baseline':>14} {'Avg Smart':>12} {'Avg Saved':>12} {'Best':>8}")
    print("-" * 82)
    for r in results:
        print(f"{r['city']:<14} {r['scenarios']:>10} {r['avg_baseline']:>12.1f}m {r['avg_smart']:>10.1f}m "
              f"{r['avg_saved']:>10.1f}% {r['best_case']:>6.1f}%")
    print("=" * 82)

    # Pick one explainability demo per city
    print("\n" + "=" * 82)
    print("  EXPLAINABILITY DEMOS - one per city")
    print("=" * 82)

    pune_scenarios = generate_pune_scenarios()
    explain_route_for_city('pune',
        next(s for s in pune_scenarios if "IPL" in s.name), store)

    mumbai_scenarios = generate_mumbai_scenarios()
    explain_route_for_city('mumbai',
        next(s for s in mumbai_scenarios if "Eastern Express" in s.name), store)

    bangalore_scenarios = generate_bangalore_scenarios()
    explain_route_for_city('bangalore',
        next(s for s in bangalore_scenarios if "Tech park" in s.name), store)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run validation for one city
        key = sys.argv[1].lower()
        if key not in list_cities():
            print(f"Unknown city: {key}. Available: {list_cities()}")
            sys.exit(1)
        store = LearningStore()
        store.reset()
        warm_up_learning_for_city(store, get_city(key), num_days=90)
        validate_city(key, store, verbose=True)
    else:
        run_all_cities()
