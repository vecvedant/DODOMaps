"""
Learning Loop - The Data Flywheel
==================================
Every completed trip updates per-edge historical delay factors.
The system learns from each customer's actual fleet operations -
this is the moat: Google can't learn from a customer's trucks
because Google doesn't see them.

Storage: simple JSON file. Production would use a time-series DB.
"""

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Tuple, Optional


HISTORY_FILE = os.path.join(os.path.dirname(__file__), "edge_history.json")


@dataclass
class EdgeStats:
    """Running statistics for an edge - exponentially weighted."""
    edge: Tuple[str, str]
    trips_observed: int = 0
    avg_actual_time: float = 0.0      # exponentially weighted moving average
    avg_baseline_time: float = 0.0    # what base_time predicted
    avg_delay_factor: float = 0.0     # (actual - baseline) / baseline, EWMA
    last_updated: str = ""

    def to_dict(self):
        d = asdict(self)
        d['edge'] = list(self.edge)  # JSON can't store tuples
        return d

    @staticmethod
    def from_dict(d):
        d = dict(d)
        d['edge'] = tuple(d['edge'])
        return EdgeStats(**d)


class LearningStore:
    """Manages persistent edge statistics that update from observed trips."""

    # Exponential weight - higher = react faster, lower = more stable
    LEARNING_RATE = 0.1

    def __init__(self, path: str = HISTORY_FILE):
        self.path = path
        self.stats: Dict[Tuple[str, str], EdgeStats] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    raw = json.load(f)
                self.stats = {
                    tuple(item['edge']): EdgeStats.from_dict(item)
                    for item in raw
                }
            except (json.JSONDecodeError, KeyError):
                self.stats = {}

    def save(self):
        data = [s.to_dict() for s in self.stats.values()]
        with open(self.path, 'w') as f:
            json.dump(data, f, indent=2)

    def get_delay_factor(self, from_node: str, to_node: str) -> float:
        """Returns learned historical_delay_factor for use in smart_weight."""
        edge = (from_node, to_node)
        s = self.stats.get(edge)
        return s.avg_delay_factor if s else 0.0

    def get_trip_count(self, from_node: str, to_node: str) -> int:
        edge = (from_node, to_node)
        s = self.stats.get(edge)
        return s.trips_observed if s else 0

    def update_from_trip(
        self,
        edge: Tuple[str, str],
        actual_time_min: float,
        baseline_time_min: float,
    ):
        """
        Update edge stats after a trip completes on this edge.
        Uses exponentially weighted moving average to adapt without
        being too jumpy on outliers.
        """
        s = self.stats.get(edge)
        if s is None:
            # First observation - initialize directly
            self.stats[edge] = EdgeStats(
                edge=edge,
                trips_observed=1,
                avg_actual_time=actual_time_min,
                avg_baseline_time=baseline_time_min,
                avg_delay_factor=(actual_time_min - baseline_time_min) / max(baseline_time_min, 0.1),
                last_updated=datetime.now().isoformat(),
            )
            return

        # Subsequent observations - EWMA update
        a = self.LEARNING_RATE
        s.trips_observed += 1
        s.avg_actual_time = (1 - a) * s.avg_actual_time + a * actual_time_min
        s.avg_baseline_time = (1 - a) * s.avg_baseline_time + a * baseline_time_min
        new_factor = (actual_time_min - baseline_time_min) / max(baseline_time_min, 0.1)
        s.avg_delay_factor = (1 - a) * s.avg_delay_factor + a * new_factor
        # clip extreme values - prevents one bad day from corrupting the model
        s.avg_delay_factor = max(0.0, min(1.5, s.avg_delay_factor))
        s.last_updated = datetime.now().isoformat()

    def summary(self) -> str:
        """For demo display - what has the system learned?"""
        if not self.stats:
            return "No trip data yet. System will learn from first trips."
        lines = [f"Learned from {sum(s.trips_observed for s in self.stats.values())} total trips:"]
        # Sort by delay factor descending to show worst-affected edges first
        sorted_edges = sorted(
            self.stats.values(),
            key=lambda s: s.avg_delay_factor,
            reverse=True
        )
        for s in sorted_edges[:10]:
            lines.append(
                f"  {s.edge[0]}->{s.edge[1]}: "
                f"{s.trips_observed} trips, "
                f"avg {s.avg_delay_factor:.0%} slower than baseline"
            )
        return "\n".join(lines)

    def reset(self):
        """Clear all learned state - for clean demos."""
        self.stats = {}
        if os.path.exists(self.path):
            os.remove(self.path)


if __name__ == "__main__":
    # Quick demo of the learning loop
    store = LearningStore()
    store.reset()  # start fresh

    print("Simulating 100 trips on edge B->D with chronic ~30% delay...")
    import random
    random.seed(7)
    for _ in range(100):
        # Simulate noisy observations of a chronically delayed edge
        actual = 11 * (1.30 + random.uniform(-0.1, 0.1))  # base 11 min + 30% +/- noise
        store.update_from_trip(('B', 'D'), actual, 11.0)

    print("Simulating 50 trips on edge C->F with normal performance...")
    for _ in range(50):
        actual = 18 * (1.0 + random.uniform(-0.05, 0.05))
        store.update_from_trip(('C', 'F'), actual, 18.0)

    print("\n" + store.summary())
    print(f"\nLearned delay factor on B->D: {store.get_delay_factor('B', 'D'):.2%}")
    print(f"Learned delay factor on C->F: {store.get_delay_factor('C', 'F'):.2%}")
    store.save()
    print(f"\nState persisted to {store.path}")
