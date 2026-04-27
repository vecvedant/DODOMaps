"""
DODOmaps — Smart Routing Demo Server
=====================================
Flask app that serves the DODOmaps UI and wraps the routing engine
behind real /api endpoints. Run this file, open the browser, demo live.

Usage:
    python app.py
    # then open http://localhost:5000

API endpoints:
    GET  /                    -> the landing page UI
    GET  /api/cities          -> list of cities + their nodes/edges/events
    GET  /api/scenarios/<city>-> available demo scenarios
    POST /api/route           -> compute baseline + smart route, return both
    GET  /api/validation      -> full multi-city validation results
"""

from dotenv import load_dotenv
load_dotenv()  # reads .env file from project root

import os

from flask import Flask, jsonify, request, render_template_string
import statistics
from datetime import datetime

import requests

import google.generativeai as genai

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = None

if GEMINI_API_KEY and GEMINI_API_KEY != "paste_your_key_here":
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # gemini-2.5-flash-lite: no thinking overhead, generous free-tier quota,
        # plenty capable for 60-word prose generation. Avoid 2.5-flash (thinking
        # tokens cause server-side timeouts) and 2.0-flash (very low daily RPD).
        GEMINI_MODEL = genai.GenerativeModel("gemini-2.5-flash-lite")
        print("[OK] Gemini initialized (gemini-2.5-flash-lite)")
    except Exception as e:
        print(f"[WARN] Gemini init failed: {e}")
        GEMINI_MODEL = None
else:
    print("[WARN] GEMINI_API_KEY not set - narrative will use template fallback")

from cities import get_city, list_cities, get_active_events, get_disrupted_edges
from learning_loop import LearningStore
from multicity_router import (
    Conditions, RoutingEngine, IntegratedConditionsProvider,
    warm_up_learning_for_city, validate_city,
    generate_pune_scenarios, generate_mumbai_scenarios, generate_bangalore_scenarios,
    SCENARIO_GENERATORS,
)
from fleet_dispatch import (
    DEMO_FLEET, fleet_metadata, simulate_fleet,
    fleet_result_to_dict, build_cascade_callouts,
    resolve_fleet_for,
)
from pune_real import get_pune_real, translate_handbuilt_events, nearest_node as _pune_real_nearest_node


app = Flask(__name__)


# =============================================================================
# WARM UP LEARNING ON STARTUP - so first API call is instant
# =============================================================================

print("Warming up learning loop for all cities...")
LEARNING_STORE = LearningStore()
LEARNING_STORE.reset()
for key in list_cities():
    warm_up_learning_for_city(LEARNING_STORE, get_city(key), num_days=90)
print(f"  Done. {sum(s.trips_observed for s in LEARNING_STORE.stats.values())} trips loaded.")


# Cache validation results (computed once at startup)
print("Pre-computing validation results...")
VALIDATION_RESULTS = {}
for key in list_cities():
    VALIDATION_RESULTS[key] = validate_city(key, LEARNING_STORE, verbose=False)
print("  Done.\n")


# Cache scenarios per city
SCENARIOS_BY_CITY = {
    'pune': generate_pune_scenarios(),
    'mumbai': generate_mumbai_scenarios(),
    'bangalore': generate_bangalore_scenarios(),
}


# =============================================================================
# PUNE (REAL GRAPH) — lazy-loaded on first request that needs it.
#
# Render's free tier caps at 512 MB. Module-level loading (unpickle 43k-node
# graph + translate 8 events × A* + resolve fleet) peaks well above that on
# cold start. Deferring to first request keeps the boot footprint small and
# the spike happens after the worker is healthy.
# =============================================================================

PUNE_REAL = None
RESOLVED_FLEET = None
_PUNE_REAL_LOADED = False


def _ensure_pune_real():
    """Idempotent. Loads cached OSM graph, translates calendar events onto
    OSM corridors, and resolves the demo fleet's hand-built endpoints to
    OSM node IDs. Heavy on the first call (~3-5s), no-op after."""
    global PUNE_REAL, RESOLVED_FLEET, _PUNE_REAL_LOADED
    if _PUNE_REAL_LOADED:
        return
    print("Loading Pune (Real) OSM graph (lazy)...")
    PUNE_REAL = get_pune_real()
    PUNE_REAL.events = translate_handbuilt_events(get_city('pune'), PUNE_REAL)
    print(f"  Translated {len(PUNE_REAL.events)} calendar events to "
          f"{sum(len(e.affected_edges) for e in PUNE_REAL.events)} OSM corridor edges.")
    RESOLVED_FLEET = resolve_fleet_for(PUNE_REAL, get_city('pune'), DEMO_FLEET)
    print(f"  Resolved {len(RESOLVED_FLEET)} fleet shipments onto pune_real graph.\n")
    _PUNE_REAL_LOADED = True


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.route('/api/cities')
def api_cities():
    """Return all city configs in JSON-friendly form."""
    out = []
    for key in list_cities():
        c = get_city(key)
        out.append({
            'key': c.key,
            'display_name': c.display_name,
            'node_count': len(c.nodes),
            'edge_count': len(c.edges),
            'event_count': len(c.events),
            'origin': c.default_origin,
            'destination': c.default_destination,
            'nodes': {nid: {'name': n.name, 'lat': n.lat, 'lon': n.lon}
                      for nid, n in c.nodes.items()},
        })
    return jsonify(out)


@app.route('/api/scenarios/<city_key>')
def api_scenarios(city_key):
    """Return demo scenarios for a city."""
    if city_key not in SCENARIOS_BY_CITY:
        return jsonify({'error': f'Unknown city {city_key}'}), 404
    scenarios = SCENARIOS_BY_CITY[city_key]
    return jsonify([
        {'index': i, 'name': s.name, 'date': s.query_date.isoformat()}
        for i, s in enumerate(scenarios)
    ])


@app.route('/api/route', methods=['POST'])
def api_route():
    """
    Compute baseline + smart routes for a given city/scenario.
    Body: { "city": "pune", "scenario_index": 11 }
    """
    data = request.get_json() or {}
    city_key = data.get('city', 'pune')
    scen_idx = int(data.get('scenario_index', 0))

    try:
        get_city(city_key)
    except KeyError:
        return jsonify({'error': 'Unknown city'}), 404
    scenarios = SCENARIOS_BY_CITY[city_key]
    if scen_idx >= len(scenarios):
        return jsonify({'error': 'Bad scenario index'}), 400

    city = get_city(city_key)
    scenario = scenarios[scen_idx]
    engine = RoutingEngine(city)
    provider = IntegratedConditionsProvider(
        city=city,
        realtime_overrides=scenario.realtime,
        query_date=scenario.query_date,
        learning_store=LEARNING_STORE,
    )

    base_path, _ = engine.find_route(
        city.default_origin, city.default_destination,
        provider.lookup, use_smart_weights=False)
    base_actual = engine.actual_travel_time(base_path, provider.lookup)

    smart_path, _ = engine.find_route(
        city.default_origin, city.default_destination,
        provider.lookup, use_smart_weights=True)
    smart_actual = engine.actual_travel_time(smart_path, provider.lookup)

    # Active calendar events for this date
    cal_events = get_active_events(city, scenario.query_date)

    # Per-edge signals on the baseline path (for explainability)
    signals = []
    for a, b in zip(base_path, base_path[1:]):
        c = provider.lookup(a, b)
        flags = []
        if c.traffic_level > 0.4:
            flags.append({'kind': 'traffic', 'source': 'live', 'value': round(c.traffic_level * 100), 'label': f'Traffic {round(c.traffic_level*100)}%'})
        if c.rain_intensity > 0.4:
            flags.append({'kind': 'rain', 'source': 'live', 'value': round(c.rain_intensity * 100), 'label': f'Rain {round(c.rain_intensity*100)}%'})
        if c.has_event:
            flags.append({'kind': 'event', 'source': 'live', 'label': c.event_name or 'Event'})
        if c.historical_delay_factor > 0.1:
            flags.append({'kind': 'history', 'source': 'learned', 'value': round(c.historical_delay_factor * 100), 'label': f'Learned delay {round(c.historical_delay_factor*100)}%'})
        if flags:
            signals.append({
                'from': city.nodes[a].name,
                'to': city.nodes[b].name,
                'flags': flags,
            })

    saved_min = base_actual - smart_actual
    saved_pct = (saved_min / base_actual * 100) if base_actual > 0 else 0

    return jsonify({
        'scenario': scenario.name,
        'date': scenario.query_date.isoformat(),
        'calendar_events': [
            {'name': e.name, 'severity': round(e.severity * 100)}
            for e in cal_events
        ],
        'baseline': {
            'path_ids': base_path,
            'path_names': [city.nodes[n].name for n in base_path],
            'time_min': round(base_actual, 1),
            'coords': [[city.nodes[n].lat, city.nodes[n].lon] for n in base_path],
        },
        'smart': {
            'path_ids': smart_path,
            'path_names': [city.nodes[n].name for n in smart_path],
            'time_min': round(smart_actual, 1),
            'coords': [[city.nodes[n].lat, city.nodes[n].lon] for n in smart_path],
        },
        'rerouted': base_path != smart_path,
        'saved_min': round(saved_min, 1),
        'saved_pct': round(saved_pct, 1),
        'signals': signals,
    })


@app.route('/api/validation')
def api_validation():
    """Multi-city validation summary."""
    return jsonify({
        'cities': VALIDATION_RESULTS,
        'total_trips_learned': sum(s.trips_observed for s in LEARNING_STORE.stats.values()),
    })


# =============================================================================
# MANUAL MODE — derive conditions from real signals (no sliders)
# =============================================================================

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# Per-city geocoordinates used for the weather lookup. Pune-only for now;
# manual mode is scoped to Pune in the UI.
CITY_LATLON = {
    'pune': (18.52, 73.85),
}

# Time-of-day → traffic baseline (FHWA-style commute curve).
TRAFFIC_BUCKETS = [
    (range(8, 11),  ("morning rush", 0.65)),
    (range(11, 17), ("midday",       0.35)),
    (range(17, 21), ("evening rush", 0.70)),
]
def _traffic_for_hour(h: int):
    for hours, (label, base) in TRAFFIC_BUCKETS:
        if h in hours:
            return label, base
    return "off-peak", 0.15


# Open-Meteo weather code → human-readable label (subset of WMO codes).
def _label_for_weather_code(code) -> str:
    try:
        c = int(code)
    except (TypeError, ValueError):
        return "Unknown"
    if c == 0: return "Clear"
    if c in (1, 2, 3): return "Partly cloudy"
    if c in (45, 48): return "Fog"
    if 51 <= c <= 57: return "Drizzle"
    if 61 <= c <= 67: return "Rain"
    if 71 <= c <= 77: return "Snow"
    if 80 <= c <= 82: return "Showers"
    if 95 <= c <= 99: return "Thunderstorm"
    return f"Code {c}"


def _rain_intensity_from_mm(mm) -> float:
    """Open-Meteo precipitation (mm/hr) → rain_intensity ∈ [0, 1]."""
    try:
        v = float(mm or 0)
    except (TypeError, ValueError):
        return 0.0
    if v <= 0:    return 0.0
    if v < 2.5:   return 0.3
    if v < 7.5:   return 0.6
    return 0.9


def fetch_weather(lat: float, lon: float):
    """
    Returns (weather_dict_or_None, error_string_or_None).
    Never raises — every failure path is reported in the second tuple element so
    the caller can surface it transparently in the assumptions panel.
    """
    try:
        r = requests.get(
            OPEN_METEO_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,precipitation,weather_code,wind_speed_10m",
            },
            timeout=3,
        )
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}"
        cur = (r.json() or {}).get("current") or {}
        return {
            "temperature_c":      cur.get("temperature_2m"),
            "precipitation_mm_hr": cur.get("precipitation"),
            "weather_code":        cur.get("weather_code"),
            "wind_speed_kph":      cur.get("wind_speed_10m"),
            "weather_label":       _label_for_weather_code(cur.get("weather_code")),
        }, None
    except requests.Timeout:
        return None, "timeout (3s)"
    except requests.RequestException as e:
        return None, f"network error: {type(e).__name__}"
    except (ValueError, KeyError) as e:
        return None, f"bad response: {type(e).__name__}"


def _fact(label, value, source, tag, **extra):
    f = {"label": label, "value": value, "source": source, "tag": tag}
    if extra:
        f["extra"] = extra
    return f


# =============================================================================
# GEMINI NARRATIVE — translate structured signals into operator-readable prose.
# Gemini does NOT make routing decisions; A* + smart weights still does that.
# Every failure path falls back to _template_narrative so the API stays alive.
# =============================================================================

def _format_path_names(path_names):
    """OSM intersections often have empty names — collapse to a summary."""
    named = [n for n in path_names if (n or '').strip()]
    if len(named) >= max(2, len(path_names) / 2):
        return " -> ".join(path_names)
    return f"{len(path_names)} OSM intersections (real-graph route)"


def _template_narrative(origin_name, dest_name, calendar_events,
                        learned_delays, saved_min, saved_pct, rerouted):
    """The original server-built narrative. Used as fallback whenever Gemini
    is unavailable so the API response always has a readable narrative."""
    parts = [f"{origin_name} to {dest_name}."]
    if calendar_events:
        ev = calendar_events[0]
        parts.append(f"Calendar event active: {ev['name']} ({ev['severity']}%).")
    if learned_delays:
        d = learned_delays[0]
        parts.append(
            f"Chronic delay learned on {d['edge']} ({d['delay_pct']}%, "
            f"{d.get('trips_observed', '?')} trips observed)."
            if 'trips_observed' in d
            else f"Chronic delay learned on {d['edge']} ({d['delay_pct']}%)."
        )
    if rerouted and saved_min > 0:
        parts.append(f"Smart route saves {saved_min:.1f} minutes ({saved_pct:.1f}%).")
    else:
        parts.append("No reroute justified - baseline path is already optimal.")
    return " ".join(parts)


def generate_narrative_with_gemini(origin_name, dest_name,
                                    assumptions_facts, calendar_events,
                                    learned_delays, baseline_path_names,
                                    baseline_time, smart_path_names,
                                    smart_time, saved_min, saved_pct,
                                    rerouted):
    """Convert structured routing signals into a 2-3 sentence dispatcher
    narrative. Returns (text, 'gemini'|'template'). Never raises."""
    if GEMINI_MODEL is None:
        return _template_narrative(
            origin_name, dest_name, calendar_events, learned_delays,
            saved_min, saved_pct, rerouted
        ), 'template'

    facts_text = "\n".join(
        f"- {f['label']}: {f['value']} (source: {f['source']}, tag: {f['tag']})"
        for f in assumptions_facts
    ) or "(no facts)"

    cal_text = (
        "Calendar events active today:\n" +
        "\n".join(f"  - {e['name']} (severity {e['severity']}%)" for e in calendar_events)
        if calendar_events else "No calendar events active today."
    )

    learned_text = (
        "Chronic delays on baseline path:\n" +
        "\n".join(
            f"  - {d['edge']} ({d['delay_pct']}% slower than baseline)"
            for d in learned_delays
        )
        if learned_delays else "No chronic delays on baseline path."
    )

    decision_text = (
        f"System REROUTED from baseline. Saved {saved_min:.1f} minutes ({saved_pct:.1f}%)."
        if rerouted else
        "System held the baseline route - no reroute justified."
    )

    prompt = f"""You are a logistics dispatcher's assistant. Given structured routing signals for a single delivery, write a 2-3 sentence plain-language explanation of why the system recommended this route. Keep it under 60 words.

Delivery: {origin_name} -> {dest_name}

Conditions detected:
{facts_text}

{cal_text}

{learned_text}

Routing decision:
- Baseline route: {_format_path_names(baseline_path_names)} ({baseline_time:.1f} min)
- Smart route:    {_format_path_names(smart_path_names)} ({smart_time:.1f} min)
- {decision_text}

STRICT RULES:
- Plain prose, no markdown, no bullets, no headings
- Reference specific numbers (minutes, percentages)
- Reference at least one specific cause (e.g. "IPL match day", "midday traffic", "chronic delay on Aundh-Shivajinagar")
- If the system did not reroute, explicitly say so
- Maximum 60 words total
- Do not invent information not present above

Output the narrative only, no preamble."""

    try:
        response = GEMINI_MODEL.generate_content(
            prompt,
            generation_config={
                "temperature": 0.4,
                "max_output_tokens": 256,
            },
            request_options={"timeout": 6},
        )
        text = (response.text or "").strip()
        if not text or len(text) > 600:
            raise ValueError(f"Gemini response invalid (len={len(text)})")
        print(f"  [OK] Gemini narrative generated ({len(text)} chars)")
        return text, 'gemini'
    except Exception as e:
        print(f"[WARN] Gemini call failed: {type(e).__name__}: {str(e)[:200]} - falling back to template")
        return _template_narrative(
            origin_name, dest_name, calendar_events, learned_delays,
            saved_min, saved_pct, rerouted
        ), 'template'


def _template_fleet_narrative(fleet_size, num_vehicles, baseline_late,
                               smart_late, recovered, late_minutes_avoided,
                               cascade_callouts, date_label):
    """Plain summary used whenever Gemini is unavailable on the fleet path."""
    bits = [
        f"{date_label}: {fleet_size} shipments across {num_vehicles} vehicles."
    ]
    if recovered > 0:
        bits.append(
            f"Smart dispatch recovered {recovered} shipment{'s' if recovered != 1 else ''} "
            f"vs baseline ({baseline_late} -> {smart_late} late) and avoided "
            f"{late_minutes_avoided:.1f} late-minutes across the fleet."
        )
    elif baseline_late == 0 and smart_late == 0:
        bits.append("All shipments on time in both modes - clean conditions, no reroute needed.")
    else:
        bits.append(f"No improvement: baseline {baseline_late} late, smart {smart_late} late.")
    return " ".join(bits)


def generate_fleet_narrative_with_gemini(fleet_size, num_vehicles,
                                          baseline_late, smart_late, recovered,
                                          late_minutes_avoided, cascade_callouts,
                                          date_label):
    """One Gemini call summarising the whole fleet comparison. Returns
    (text, 'gemini'|'template'). Never raises."""
    if GEMINI_MODEL is None:
        return _template_fleet_narrative(
            fleet_size, num_vehicles, baseline_late, smart_late, recovered,
            late_minutes_avoided, cascade_callouts, date_label
        ), 'template'

    callouts_text = (
        "\n".join(f"- {c['sentence']}" for c in (cascade_callouts or [])[:5])
        or "(no cascade callouts)"
    )

    prompt = f"""You are a logistics operations summary writer. Summarize today's fleet dispatch comparison in 2-3 sentences. Plain prose, under 70 words.

Date context: {date_label}
Fleet: {fleet_size} shipments across {num_vehicles} vehicles.

Result:
- Baseline dispatch: {baseline_late} shipments would be LATE.
- Smart dispatch: {smart_late} shipments LATE.
- Recovered: {recovered} shipments.
- Late minutes avoided across the fleet: {late_minutes_avoided:.1f}.

Cascade examples:
{callouts_text}

Write a short executive summary explaining what the system prevented today. Reference specific numbers. No markdown, no bullets."""

    try:
        response = GEMINI_MODEL.generate_content(
            prompt,
            generation_config={"temperature": 0.4, "max_output_tokens": 300},
            request_options={"timeout": 6},
        )
        text = (response.text or "").strip()
        if not text or len(text) > 700:
            raise ValueError(f"Gemini response invalid (len={len(text)})")
        print(f"  [OK] Gemini fleet narrative generated ({len(text)} chars)")
        return text, 'gemini'
    except Exception as e:
        print(f"[WARN] Gemini fleet summary failed: {type(e).__name__}: {str(e)[:200]} - falling back to template")
        return _template_fleet_narrative(
            fleet_size, num_vehicles, baseline_late, smart_late, recovered,
            late_minutes_avoided, cascade_callouts, date_label
        ), 'template'


@app.route('/api/manual_route', methods=['POST'])
def api_manual_route():
    """
    Compute baseline + smart routes from a user-picked start/end pair.
    Conditions are NOT user-set — they are derived live from:
      - system clock (time of day, day of week)
      - Open-Meteo (current weather)
      - calendar database (active events for the chosen date)
      - learning store (chronic per-edge delays)

    Body: { "city": "pune", "start_node": "SWG", "end_node": "WH",
            "date_override": "2026-05-02"  # optional ISO date string }
    """
    data = request.get_json() or {}
    city_key = data.get('city', 'pune_real')
    if city_key == 'pune_real':
        _ensure_pune_real()
    try:
        city = get_city(city_key)
    except KeyError:
        return jsonify({'error': 'Unknown city'}), 404

    start = data.get('start_node')
    end = data.get('end_node')
    if start not in city.nodes or end not in city.nodes:
        return jsonify({'error': 'Unknown start or end node'}), 400
    if start == end:
        return jsonify({'error': 'Start and end must differ'}), 400

    # ----- a) Date ---------------------------------------------------------
    now = datetime.now()
    date_override_raw = data.get('date_override')
    if date_override_raw:
        try:
            query_date = datetime.strptime(date_override_raw, "%Y-%m-%d").date()
            date_source = "user override"
            date_tag = "INFERRED"
        except ValueError:
            return jsonify({'error': f'Bad date_override: {date_override_raw!r}'}), 400
    else:
        query_date = now.date()
        date_source = "System clock"
        date_tag = "REAL"

    # ----- b) Time of day → traffic baseline ------------------------------
    hour_label, traffic_baseline = _traffic_for_hour(now.hour)

    # ----- c) Day of week adjustment --------------------------------------
    weekday_idx = now.weekday()  # 0=Mon ... 6=Sun
    weekday_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                    'Friday', 'Saturday', 'Sunday'][weekday_idx]
    if weekday_idx == 5:    dow_adjust = -0.10
    elif weekday_idx == 6:  dow_adjust = -0.20
    else:                   dow_adjust = 0.0
    traffic_level = max(0.0, min(1.0, traffic_baseline + dow_adjust))

    # ----- d) Weather (live API) ------------------------------------------
    lat, lon = CITY_LATLON.get(city_key, (18.52, 73.85))
    weather, weather_err = fetch_weather(lat, lon)
    if weather is not None:
        rain_intensity = _rain_intensity_from_mm(weather["precipitation_mm_hr"])
        weather_value = (
            f"{weather['temperature_c']}°C · {weather['weather_label']}"
            if weather.get('temperature_c') is not None
            else weather['weather_label']
        )
        weather_fact = _fact(
            "Weather", weather_value,
            "Open-Meteo API (live)", "REAL",
            **weather,
            rain_intensity_derived=rain_intensity,
        )
    else:
        rain_intensity = 0.0
        weather_fact = _fact(
            "Weather", "Unavailable",
            f"Open-Meteo unreachable — {weather_err}", "OFFLINE",
            fallback="assumed clear",
        )

    # ----- Apply derived conditions uniformly -----------------------------
    # has_event=False so the IntegratedConditionsProvider's calendar overlay
    # fires per-edge from PUNE_EVENTS rather than being clobbered.
    cond_template = Conditions(
        traffic_level=traffic_level,
        rain_intensity=rain_intensity,
        has_event=False,
    )
    realtime = {(e.from_node, e.to_node): cond_template for e in city.edges}

    engine = RoutingEngine(city)
    provider = IntegratedConditionsProvider(
        city=city,
        realtime_overrides=realtime,
        query_date=query_date,
        learning_store=LEARNING_STORE,
    )

    base_path, _ = engine.find_route(start, end, provider.lookup, use_smart_weights=False)
    smart_path, _ = engine.find_route(start, end, provider.lookup, use_smart_weights=True)
    if not base_path or not smart_path:
        return jsonify({'error': f'No path from {start} to {end}'}), 400

    base_actual = engine.actual_travel_time(base_path, provider.lookup)
    smart_actual = engine.actual_travel_time(smart_path, provider.lookup)

    # ----- e) Calendar events ---------------------------------------------
    # On pune_real these are translated corridor events — they fire correctly
    # but the per-edge tuples are OSM node-id pairs, not user-readable. We
    # still surface the event names + severity in the assumptions panel.
    cal_events_raw = get_active_events(city, query_date)
    cal_events_payload = [
        {
            'name': e.name,
            'severity': round(e.severity * 100),
            'affected_edges': [list(p) for p in e.affected_edges],
        }
        for e in cal_events_raw
    ]
    if cal_events_raw:
        cal_summary = ", ".join(f"{e.name} ({round(e.severity*100)}%)" for e in cal_events_raw[:2])
        if len(cal_events_raw) > 2:
            cal_summary += f", +{len(cal_events_raw)-2} more"
        calendar_fact = _fact("Calendar events", cal_summary,
                              "Pune calendar database", "CONFIGURED",
                              count=len(cal_events_raw))
    else:
        calendar_fact = _fact("Calendar events", "No events on this date",
                              "Pune calendar database", "CONFIGURED",
                              count=0)

    # ----- f) Learned delays touching the baseline path -------------------
    learned_on_route = []
    for a, b in zip(base_path, base_path[1:]):
        c = provider.lookup(a, b)
        if c.historical_delay_factor > 0.1:
            learned_on_route.append({
                'edge': f"{city.nodes[a].name} → {city.nodes[b].name}",
                'delay_factor': round(c.historical_delay_factor, 3),
                'delay_pct': round(c.historical_delay_factor * 100),
            })
    learned_on_route.sort(key=lambda x: -x['delay_factor'])
    learned_top = learned_on_route[:3]
    if learned_top:
        learned_summary = "; ".join(f"{x['edge']} {x['delay_pct']}%" for x in learned_top)
        learned_fact = _fact("Learned delays", learned_summary,
                             f"EWMA learning store · {sum(s.trips_observed for s in LEARNING_STORE.stats.values())} trips",
                             "CONFIGURED", edges=learned_top)
    else:
        learned_fact = _fact("Learned delays", "No chronic delays on this path",
                             f"EWMA learning store · {sum(s.trips_observed for s in LEARNING_STORE.stats.values())} trips",
                             "CONFIGURED", edges=[])

    # ----- Per-edge signals (same shape as scenario mode) -----------------
    signals = []
    for a, b in zip(base_path, base_path[1:]):
        c = provider.lookup(a, b)
        flags = []
        if c.traffic_level > 0.4:
            flags.append({'kind': 'traffic', 'source': 'live',
                          'label': f'Traffic {round(c.traffic_level*100)}%'})
        if c.rain_intensity > 0.4:
            flags.append({'kind': 'rain', 'source': 'live',
                          'label': f'Rain {round(c.rain_intensity*100)}%'})
        if c.has_event:
            flags.append({'kind': 'event', 'source': 'live',
                          'label': c.event_name or 'Event'})
        if c.historical_delay_factor > 0.1:
            flags.append({'kind': 'history', 'source': 'learned',
                          'label': f'Learned delay {round(c.historical_delay_factor*100)}%'})
        if flags:
            signals.append({
                'from': city.nodes[a].name,
                'to': city.nodes[b].name,
                'flags': flags,
            })

    saved_min = base_actual - smart_actual
    saved_pct = (saved_min / base_actual * 100) if base_actual > 0 else 0
    rerouted = base_path != smart_path

    facts = [
        _fact("Date", query_date.strftime("%a %b %d, %Y"), date_source, date_tag),
        _fact("Time of day",
              f"{now.strftime('%H:%M')} · {hour_label} ({round(traffic_baseline*100)}%)",
              "System clock + FHWA traffic patterns", "INFERRED",
              hour=now.hour, hour_label=hour_label, traffic_baseline=traffic_baseline),
        _fact("Day of week",
              f"{weekday_name} · adjustment {dow_adjust:+.0%}",
              "System clock", "INFERRED", weekday=weekday_name, adjustment=dow_adjust),
        weather_fact,
        calendar_fact,
        learned_fact,
    ]

    # ----- Narrative (Gemini if available, else server-built template) -----
    # Prefer the user's original query string (e.g. "Swargate") over an empty
    # OSM node name (most OSM intersections are unnamed). The UI sends these
    # alongside the snapped OSM IDs so the narrative reads naturally.
    cal_events_for_prompt = [
        {'name': e['name'], 'severity': e['severity']}
        for e in cal_events_payload
    ]
    start_query = (data.get('start_query') or '').strip()
    end_query = (data.get('end_query') or '').strip()
    origin_label = (city.nodes[start].name.strip()
                    if city.nodes[start].name and city.nodes[start].name.strip()
                    else (start_query or f"intersection {start}"))
    dest_label = (city.nodes[end].name.strip()
                  if city.nodes[end].name and city.nodes[end].name.strip()
                  else (end_query or f"intersection {end}"))
    print(f"  -> Calling Gemini for narrative: {origin_label} -> {dest_label} "
          f"(model_loaded={GEMINI_MODEL is not None})")
    narrative_text, narrative_source = generate_narrative_with_gemini(
        origin_name=origin_label,
        dest_name=dest_label,
        assumptions_facts=facts,
        calendar_events=cal_events_for_prompt,
        learned_delays=learned_top,
        baseline_path_names=[city.nodes[n].name for n in base_path],
        baseline_time=base_actual,
        smart_path_names=[city.nodes[n].name for n in smart_path],
        smart_time=smart_actual,
        saved_min=saved_min,
        saved_pct=saved_pct,
        rerouted=rerouted,
    )

    assumptions = {
        "facts": facts,
        "narrative": narrative_text,
        "narrative_source": narrative_source,  # 'gemini' or 'template'
        "derived": {
            "traffic_level": round(traffic_level, 3),
            "rain_intensity": round(rain_intensity, 3),
        },
    }

    return jsonify({
        'scenario': f'{city.nodes[start].name}  →  {city.nodes[end].name}',
        'date': query_date.isoformat(),
        'calendar_events': [
            {'name': e['name'], 'severity': e['severity']} for e in cal_events_payload
        ],
        'baseline': {
            'path_ids': base_path,
            'path_names': [city.nodes[n].name for n in base_path],
            'time_min': round(base_actual, 1),
            'coords': [[city.nodes[n].lat, city.nodes[n].lon] for n in base_path],
        },
        'smart': {
            'path_ids': smart_path,
            'path_names': [city.nodes[n].name for n in smart_path],
            'time_min': round(smart_actual, 1),
            'coords': [[city.nodes[n].lat, city.nodes[n].lon] for n in smart_path],
        },
        'rerouted': rerouted,
        'saved_min': round(saved_min, 1),
        'saved_pct': round(saved_pct, 1),
        'signals': signals,
        'assumptions': assumptions,
    })


# =============================================================================
# FLEET DISPATCH — multi-shipment cascade simulation
# =============================================================================

def _resolve_fleet_date(date_override_raw):
    """Same date-parsing rule as manual mode."""
    if date_override_raw:
        try:
            return datetime.strptime(date_override_raw, "%Y-%m-%d").date()
        except ValueError:
            return None    # caller treats None as bad input
    return datetime.now().date()


def _fleet_rain_intensity(city_key):
    """Fetch live weather once per fleet request and convert to rain_intensity.
    Failure-tolerant: returns (rain, weather_summary_dict_or_None)."""
    lat, lon = CITY_LATLON.get(city_key, (18.52, 73.85))
    weather, err = fetch_weather(lat, lon)
    if weather is None:
        return 0.0, {'available': False, 'error': err}
    return _rain_intensity_from_mm(weather['precipitation_mm_hr']), {
        'available': True,
        'temperature_c': weather.get('temperature_c'),
        'weather_label': weather.get('weather_label'),
        'precipitation_mm_hr': weather.get('precipitation_mm_hr'),
    }


def _fleet_for_city(city_key: str):
    """Pick the right fleet representation for the given city. On pune_real
    we hand back the pre-resolved (OSM-node-id) fleet, lazy-loading the OSM
    graph on first call so cold-start RAM stays under Render's free tier cap."""
    if city_key == 'pune_real':
        _ensure_pune_real()
        return RESOLVED_FLEET
    return DEMO_FLEET


# =============================================================================
# GEOCODING — Nominatim + nearest-node snap
# =============================================================================

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
# Pune bbox (must match pune_real.BBOX) — sent to Nominatim viewbox to bias
# matches inside the city. viewbox order is left, top, right, bottom.
_NOMINATIM_VIEWBOX = "73.72,18.64,73.96,18.46"
_GEOCODE_HEADERS = {"User-Agent": "obsidian-routing-demo/1.0 (hackathon)"}


@app.route('/api/geocode', methods=['POST'])
def api_geocode():
    """
    Resolve a free-text place name to a graph node.

    Body: { "query": "Swargate", "city": "pune_real" }
    Returns: { "lat", "lon", "display_name", "nearest_node", "node_distance_m" }
    or an "error" field on failure (no snap, network down, no results).
    """
    data = request.get_json() or {}
    q = (data.get('query') or '').strip()
    city_key = data.get('city', 'pune_real')
    if city_key == 'pune_real':
        _ensure_pune_real()

    if not q:
        return jsonify({'error': 'Empty query'}), 400
    try:
        city = get_city(city_key)
    except KeyError:
        return jsonify({'error': f"Unknown city '{city_key}'"}), 404

    try:
        r = requests.get(
            NOMINATIM_URL,
            params={
                "q": f"{q}, Pune, India",
                "format": "json",
                "limit": 1,
                "viewbox": _NOMINATIM_VIEWBOX,
                "bounded": 1,
            },
            headers=_GEOCODE_HEADERS,
            timeout=3,
        )
    except requests.Timeout:
        return jsonify({'error': 'Nominatim timed out (3s).'}), 504
    except requests.RequestException as e:
        return jsonify({'error': f'Nominatim network error: {type(e).__name__}'}), 502

    if r.status_code != 200:
        return jsonify({'error': f'Nominatim HTTP {r.status_code}'}), 502

    try:
        results = r.json() or []
    except ValueError:
        return jsonify({'error': 'Nominatim returned non-JSON'}), 502

    if not results:
        return jsonify({'error': f"No Pune match for {q!r}"}), 404

    hit = results[0]
    lat, lon = float(hit['lat']), float(hit['lon'])
    nearest, dist_m = _pune_real_nearest_node(city, lat, lon)
    return jsonify({
        'query': q,
        'lat': lat, 'lon': lon,
        'display_name': hit.get('display_name'),
        'nearest_node': nearest,
        'node_distance_m': round(dist_m, 1),
    })


@app.route('/api/fleet/demo')
def api_fleet_demo():
    """Returns the demo fleet so the UI can render the table layout even
    before a simulation runs. Defaults to pune_real."""
    city_key = request.args.get('city', 'pune_real')
    fleet = _fleet_for_city(city_key)
    try:
        city = get_city(city_key)
    except KeyError:
        return jsonify({'error': f"Unknown city '{city_key}'"}), 404
    return jsonify({
        **fleet_metadata(city_key, fleet),
        'graph': {
            'node_count': len(city.nodes),
            'edge_count': len(city.edges),
            'display_name': city.display_name,
        },
    })


@app.route('/api/fleet/simulate', methods=['POST'])
def api_fleet_simulate():
    """Run one mode (baseline OR smart) over the demo fleet.

    Body: { "city": "pune_real"|"pune", "date_override": "2026-05-02"|null,
            "mode": "baseline"|"smart" }
    """
    data = request.get_json() or {}
    city_key = data.get('city', 'pune_real')
    try:
        get_city(city_key)
    except KeyError:
        return jsonify({'error': 'Unknown city'}), 404
    mode = data.get('mode', 'smart')
    if mode not in ('baseline', 'smart'):
        return jsonify({'error': "mode must be 'baseline' or 'smart'"}), 400

    query_date = _resolve_fleet_date(data.get('date_override'))
    if query_date is None:
        return jsonify({'error': f"Bad date_override: {data.get('date_override')!r}"}), 400

    rain, weather_summary = _fleet_rain_intensity(city_key)
    result = simulate_fleet(
        city_key=city_key,
        fleet=_fleet_for_city(city_key),
        query_date=query_date,
        rain_intensity=rain,
        use_smart=(mode == 'smart'),
        learning_store=LEARNING_STORE,
    )
    return jsonify({
        **fleet_result_to_dict(result),
        'date': query_date.isoformat(),
        'weather': weather_summary,
        'calendar_events': [
            {'name': e.name, 'severity': round(e.severity * 100)}
            for e in get_active_events(get_city(city_key), query_date)
        ],
    })


@app.route('/api/fleet/compare', methods=['POST'])
def api_fleet_compare():
    """Run both modes and return the diff. This is the headline endpoint.

    Body: { "city": "pune_real"|"pune", "date_override": "2026-05-02"|null }
    """
    data = request.get_json() or {}
    city_key = data.get('city', 'pune_real')
    try:
        get_city(city_key)
    except KeyError:
        return jsonify({'error': 'Unknown city'}), 404

    query_date = _resolve_fleet_date(data.get('date_override'))
    if query_date is None:
        return jsonify({'error': f"Bad date_override: {data.get('date_override')!r}"}), 400

    rain, weather_summary = _fleet_rain_intensity(city_key)
    city = get_city(city_key)
    fleet = _fleet_for_city(city_key)

    baseline = simulate_fleet(city_key, fleet, query_date, rain,
                              use_smart=False, learning_store=LEARNING_STORE)
    smart = simulate_fleet(city_key, fleet, query_date, rain,
                           use_smart=True,  learning_store=LEARNING_STORE)

    callouts = build_cascade_callouts(baseline, smart, city)

    # Total minutes saved across the fleet = sum of route_time differences
    # (positive when smart is faster). Late-minute reduction is a separate stat.
    total_route_min_saved = round(sum(
        b.route_time - s.route_time for b, s in zip(baseline.shipments, smart.shipments)
    ), 1)

    active_events = get_active_events(city, query_date)
    if active_events:
        date_label = f"{query_date.isoformat()} ({active_events[0].name})"
    else:
        date_label = f"{query_date.isoformat()} (no calendar events)"

    recovered = baseline.total_late - smart.total_late
    late_minutes_avoided = round(baseline.total_minutes_late - smart.total_minutes_late, 1)
    fleet_size = len(baseline.shipments)
    num_vehicles = len({s.shipment.vehicle_id for s in baseline.shipments})

    fleet_narrative_text, fleet_narrative_source = generate_fleet_narrative_with_gemini(
        fleet_size=fleet_size,
        num_vehicles=num_vehicles,
        baseline_late=baseline.total_late,
        smart_late=smart.total_late,
        recovered=recovered,
        late_minutes_avoided=late_minutes_avoided,
        cascade_callouts=callouts,
        date_label=date_label,
    )

    return jsonify({
        'date': query_date.isoformat(),
        'weather': weather_summary,
        'calendar_events': [
            {'name': e.name, 'severity': round(e.severity * 100)}
            for e in active_events
        ],
        'baseline': fleet_result_to_dict(baseline),
        'smart':    fleet_result_to_dict(smart),
        'savings': {
            'shipments_recovered': recovered,
            'late_minutes_avoided': late_minutes_avoided,
            'total_route_min_saved': total_route_min_saved,
        },
        'cascade_callouts': callouts,
        'narrative': fleet_narrative_text,
        'narrative_source': fleet_narrative_source,  # 'gemini' or 'template'
    })


# =============================================================================
# UI — Neo-Brutalist
# =============================================================================

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>DODOmaps — Predictive Routing</title>
<link href="https://api.fontshare.com/v2/css?f[]=cabinet-grotesk@400,500,700,800&f[]=satoshi@400,500,700,900&display=swap" rel="stylesheet">
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --yellow: #ffe17c;
  --char: #171e19;
  --sage: #b7c6c2;
  --white: #ffffff;
  --black: #000000;
  --gray: #272727;
  --gray-soft: #f4f4f5;
  --star: #ffbc2e;
}
html, body {
  background: var(--white);
  color: var(--black);
  font-family: 'Satoshi', system-ui, -apple-system, sans-serif;
  font-weight: 500;
  -webkit-font-smoothing: antialiased;
  font-size: 16px;
  line-height: 1.5;
}
body { padding-top: 80px; }
h1, h2, h3, h4 {
  font-family: 'Cabinet Grotesk', system-ui, sans-serif;
  font-weight: 800;
  letter-spacing: -0.04em;
  line-height: 0.95;
}
a { color: inherit; text-decoration: none; }
button { font-family: inherit; cursor: pointer; background: none; border: none; color: inherit; }
ul { list-style: none; }

/* === BRUTALIST PRIMITIVES =================================== */
.btn-push {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  background: #000;
  color: #fff;
  padding: 1rem 2rem;
  border: 2px solid #000;
  border-radius: 0.75rem;
  box-shadow: 8px 8px 0 0 #000;
  font-weight: 700;
  font-family: 'Satoshi', sans-serif;
  font-size: 0.95rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  transition: all 0.2s cubic-bezier(0.175, 0.885, 0.32, 1.275);
  text-decoration: none;
  user-select: none;
}
.btn-push:hover, .btn-push:focus-visible {
  transform: translate(4px, 4px);
  box-shadow: 0 0 0 0 #000;
  outline: none;
}
.btn-push.sm {
  padding: 0.65rem 1.2rem;
  box-shadow: 4px 4px 0 0 #000;
  font-size: 0.8rem;
}
.btn-push.white {
  background: #fff;
  color: #000;
  box-shadow: 4px 4px 0 0 #000;
}
.btn-push.yellow {
  background: var(--yellow);
  color: #000;
}

.eyebrow {
  font-family: 'Satoshi', sans-serif;
  font-weight: 700;
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.12em;
}
.h-display { font-size: clamp(2.5rem, 8vw, 6.5rem); }
.h-section { font-size: clamp(2rem, 5.5vw, 4.5rem); }
.stroke-text { -webkit-text-stroke: 2px #000; color: transparent; }
.stroke-text-yel { -webkit-text-stroke: 2px var(--yellow); color: transparent; }

.container { max-width: 1280px; margin: 0 auto; padding: 0 1.5rem; }

/* === DOT PATTERN =========================================== */
.bg-yellow {
  background-color: var(--yellow);
  position: relative;
}
.bg-yellow::before {
  content: '';
  position: absolute; inset: 0;
  background-image: radial-gradient(#000 1.4px, transparent 1.4px);
  background-size: 32px 32px;
  background-position: 0 0;
  opacity: 0.1;
  pointer-events: none;
}
.bg-yellow > * { position: relative; z-index: 1; }

/* === NAV =================================================== */
.nav {
  position: fixed; top: 0; left: 0; right: 0;
  height: 80px;
  background: var(--yellow);
  border-bottom: 2px solid #000;
  z-index: 100;
  display: flex;
  align-items: center;
}
.nav-inner {
  display: flex; align-items: center; justify-content: space-between;
  width: 100%;
}
.logo { display: flex; align-items: center; gap: 0.6rem; }
.logo-mark {
  width: 40px; height: 40px;
  background: #000;
  border: 2px solid #000;
  border-radius: 0.4rem;
  display: flex; align-items: center; justify-content: center;
}
.logo-mark svg { width: 22px; height: 22px; }
.logo-text {
  font-family: 'Cabinet Grotesk', sans-serif;
  font-weight: 800;
  font-size: 1.4rem;
  letter-spacing: -0.04em;
}
.nav-links { display: flex; gap: 2rem; align-items: center; }
.nav-links a {
  font-weight: 700;
  font-size: 0.9rem;
}
.nav-links a:hover {
  text-decoration: underline;
  text-decoration-thickness: 2px;
  text-underline-offset: 4px;
}
@media (max-width: 768px) { .nav-links { display: none; } }

/* === SECTIONS ============================================== */
section { padding: 5rem 0; position: relative; }
.section-yellow { background: var(--yellow); border-top: 2px solid #000; border-bottom: 2px solid #000; }
.section-char { background: var(--char); color: #fff; border-top: 2px solid #000; border-bottom: 2px solid #000; }
.section-sage { background: var(--sage); border-top: 2px solid #000; border-bottom: 2px solid #000; }

.section-eyebrow-row { display: flex; align-items: center; gap: 0.85rem; margin-bottom: 1.25rem; }
.section-eyebrow-row .bar { display: inline-block; width: 36px; height: 3px; background: #000; }
.section-char .section-eyebrow-row .bar { background: var(--yellow); }
.section-char .eyebrow { color: var(--yellow); }

/* === HERO ================================================== */
.hero { padding: 5rem 0 6rem; }
.hero-grid {
  display: grid; grid-template-columns: 1.05fr 1fr;
  gap: 3.5rem; align-items: center;
}
@media (max-width: 960px) { .hero-grid { grid-template-columns: 1fr; gap: 3rem; } }

.badge {
  display: inline-flex; align-items: center; gap: 0.55rem;
  background: #fff;
  border: 2px solid #000;
  border-radius: 9999px;
  padding: 0.5rem 1rem;
  font-weight: 700; font-size: 0.78rem;
  box-shadow: 4px 4px 0 0 #000;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.badge .dot {
  width: 8px; height: 8px;
  background: #000;
  border-radius: 50%;
  animation: pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot { 0%,100% { opacity: 1; } 50% { opacity: 0.35; } }

.hero-stats { display: flex; gap: 2.5rem; flex-wrap: wrap; }
.hero-stat .num {
  font-family: 'Cabinet Grotesk', sans-serif;
  font-weight: 800;
  font-size: 2.1rem;
  letter-spacing: -0.04em;
  line-height: 1;
}
.hero-stat .lbl {
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-top: 0.4rem;
}

/* === BROWSER MOCKUP ======================================== */
.mockup {
  background: #fff;
  border: 2px solid #000;
  border-radius: 1rem;
  box-shadow: 12px 12px 0 0 #000;
  overflow: hidden;
}
.mockup-bar {
  background: #000;
  padding: 0.75rem 1rem;
  display: flex; gap: 0.4rem; align-items: center;
}
.mockup-dot { width: 12px; height: 12px; border-radius: 50%; }
.mockup-url {
  margin-left: 0.85rem;
  color: rgba(255,255,255,0.5);
  font-family: 'Satoshi', monospace;
  font-size: 0.72rem;
  letter-spacing: 0.05em;
}
.mockup-body { padding: 1.1rem; background: #fafafa; }
.mockup-grid { display: grid; grid-template-columns: 2fr 1fr; gap: 0.75rem; }
.mockup-card { padding: 0.95rem; border: 2px solid #000; border-radius: 0.5rem; }
.mockup-card .lbl {
  font-size: 0.65rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; opacity: 0.75;
}
.mockup-card .num {
  font-family: 'Cabinet Grotesk', sans-serif;
  font-weight: 800;
  font-size: 1.65rem;
  letter-spacing: -0.04em;
  margin-top: 0.15rem;
  line-height: 1;
}

/* === MARQUEE =============================================== */
.marquee-section { padding: 0; }
.marquee { overflow: hidden; padding: 1.4rem 0; }
.marquee-track {
  display: flex; gap: 3.5rem; align-items: center;
  animation: marquee 38s linear infinite;
  width: max-content;
  font-family: 'Cabinet Grotesk', sans-serif;
  font-weight: 800;
  font-size: 1.55rem;
  letter-spacing: -0.02em;
  color: var(--sage);
  opacity: 0.5;
  text-transform: uppercase;
}
.marquee-track .star { font-size: 1rem; }
@keyframes marquee { from { transform: translateX(0); } to { transform: translateX(-50%); } }

/* === PROBLEM VS SOLUTION =================================== */
.ps-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 2.5rem; }
@media (max-width: 900px) { .ps-grid { grid-template-columns: 1fr; } }
.ps-card { padding: 2.5rem; border-radius: 1.5rem; }
.ps-problem {
  background: var(--gray-soft);
  border: 2px dashed #888;
  opacity: 0.85;
}
.ps-solution {
  background: var(--yellow);
  border: 2px solid #000;
  box-shadow: 8px 8px 0 0 #000;
}
.ps-list { display: flex; flex-direction: column; gap: 1rem; margin-top: 1.75rem; }
.ps-list li { display: flex; gap: 0.85rem; align-items: flex-start; font-weight: 600; line-height: 1.45; }
.ps-icon {
  flex-shrink: 0;
  width: 28px; height: 28px;
  border: 2px solid #000;
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-weight: 800; font-size: 0.9rem;
  line-height: 1;
}
.ps-icon.x { background: #fff; }
.ps-icon.check { background: #000; color: var(--yellow); }

/* === DEMO CONSOLE ========================================== */
.demo-console {
  background: #fff;
  border: 2px solid #000;
  border-radius: 1.25rem;
  box-shadow: 8px 8px 0 0 #000;
  padding: 2.25rem;
}
@media (max-width: 600px) { .demo-console { padding: 1.5rem; } }
.demo-step { margin-bottom: 1.75rem; }
.demo-step-label {
  display: inline-flex; align-items: center; gap: 0.45rem;
  background: #000; color: var(--yellow);
  padding: 0.4rem 0.8rem;
  border-radius: 0.4rem;
  font-family: 'Cabinet Grotesk', sans-serif;
  font-weight: 800;
  font-size: 0.78rem;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  margin-bottom: 0.95rem;
}
.tab-row { display: flex; flex-wrap: wrap; gap: 0.55rem; }
.tab {
  background: #fff;
  border: 2px solid #000;
  padding: 0.5rem 0.95rem;
  border-radius: 0.5rem;
  font-weight: 700;
  font-size: 0.85rem;
  font-family: 'Satoshi', sans-serif;
  box-shadow: 4px 4px 0 0 #000;
  transition: all 0.18s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}
.tab:hover { transform: translate(4px, 4px); box-shadow: 0 0 0 0 #000; }
.tab.active { background: #000; color: var(--yellow); }
.scenario-row {
  max-height: 11rem; overflow-y: auto;
  padding: 0.85rem;
  border: 2px dashed #888; border-radius: 0.6rem;
  background: #fafafa;
}

/* === RESULTS =============================================== */
.results { margin-top: 2.25rem; padding-top: 2rem; border-top: 2px dashed #000; }
.results-head {
  display: flex; justify-content: space-between; align-items: flex-end;
  margin-bottom: 1.75rem; flex-wrap: wrap; gap: 1.5rem;
}
.saved-pct {
  font-family: 'Cabinet Grotesk', sans-serif;
  font-weight: 800;
  font-size: clamp(3rem, 9vw, 6.5rem);
  line-height: 0.9;
  letter-spacing: -0.06em;
}
.route-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.25rem; margin-bottom: 1.5rem; }
@media (max-width: 700px) { .route-grid { grid-template-columns: 1fr; } }
.route-card { padding: 1.5rem; border: 2px solid #000; border-radius: 0.75rem; }
.route-card.base { background: var(--sage); }
.route-card.smart { background: var(--yellow); box-shadow: 4px 4px 0 0 #000; }
.route-time {
  font-family: 'Cabinet Grotesk', sans-serif;
  font-weight: 800;
  font-size: 3.25rem;
  line-height: 1;
  letter-spacing: -0.04em;
  margin-top: 0.5rem;
}
.route-path {
  margin-top: 1rem;
  font-size: 0.82rem;
  font-weight: 600;
  line-height: 1.55;
  word-break: break-word;
}

.map-card {
  background: #fff;
  border: 2px solid #000;
  border-radius: 0.75rem;
  padding: 1.25rem;
  margin-bottom: 1.5rem;
}
.map-card svg {
  width: 100%; height: auto;
  background: var(--gray-soft);
  border: 2px solid #000;
  border-radius: 0.4rem;
  display: block;
}

.bottom-grid { display: grid; grid-template-columns: 1fr 2fr; gap: 1.25rem; }
@media (max-width: 700px) { .bottom-grid { grid-template-columns: 1fr; } }
.decision-card {
  background: var(--char); color: #fff;
  padding: 1.5rem;
  border: 2px solid #000;
  border-radius: 0.75rem;
}
.signals-card {
  background: var(--char); color: #fff;
  padding: 1.5rem;
  border: 2px solid #000;
  border-radius: 0.75rem;
}
.signal-row { padding: 0.7rem 0; border-bottom: 1px dashed #444; }
.signal-row:last-child { border-bottom: none; }
.signal-from {
  font-weight: 700;
  font-size: 0.88rem;
  letter-spacing: 0.02em;
}
.signal-flag {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.18rem 0.55rem;
  border-radius: 0.3rem;
  font-size: 0.7rem;
  font-weight: 700;
  margin-right: 0.4rem;
  margin-top: 0.4rem;
  border: 1.5px solid #000;
  color: #000;
}
.signal-flag.live    { background: var(--yellow); }
.signal-flag.learned { background: var(--sage); }
.signal-flag .src-tag {
  font-size: 0.55rem;
  font-weight: 800;
  letter-spacing: 0.1em;
  padding: 0 0.3rem;
  background: #000;
  color: #fff;
  border-radius: 0.18rem;
  line-height: 1.4;
}

/* === MANUAL MODE ========================================== */
.manual-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.5rem;
  margin-bottom: 1.75rem;
}
@media (max-width: 700px) { .manual-grid { grid-template-columns: 1fr; } }

.brut-select {
  background: #fff;
  color: #000;
  border: 2px solid #000;
  border-radius: 0.5rem;
  padding: 0.65rem 2.5rem 0.65rem 1rem;
  box-shadow: 4px 4px 0 0 #000;
  font-family: 'Satoshi', sans-serif;
  font-weight: 700;
  font-size: 0.92rem;
  width: 100%;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
  background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 12 8' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><polyline points='1 1.5 6 6.5 11 1.5'/></svg>");
  background-repeat: no-repeat;
  background-position: right 0.85rem center;
  background-size: 12px;
  cursor: pointer;
  transition: all 0.18s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}
.brut-select:hover { box-shadow: 6px 6px 0 0 #000; }
.brut-select:focus { box-shadow: 2px 2px 0 0 #000; outline: none; }

.brut-input {
  background: #fff;
  color: #000;
  border: 2px solid #000;
  border-radius: 0.5rem;
  padding: 0.65rem 1rem;
  box-shadow: 4px 4px 0 0 #000;
  font-family: 'Satoshi', sans-serif;
  font-weight: 700;
  font-size: 0.95rem;
  width: 100%;
  outline: none;
  transition: box-shadow 0.18s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}
.brut-input:focus { box-shadow: 2px 2px 0 0 #000; }
.brut-input::placeholder { color: #999; font-weight: 600; }

.manual-console {
  background: #fff;
  color: #000;
  border: 2px solid #000;
  border-radius: 1.25rem;
  box-shadow: 8px 8px 0 0 #000;
  padding: 2.25rem;
}
@media (max-width: 600px) { .manual-console { padding: 1.5rem; } }
.section-char .manual-console h3 { color: #000; }

/* === WHY-THIS-ROUTE PANEL ================================ */
.why-card {
  background: #fafafa;
  border: 2px solid #000;
  border-radius: 0.85rem;
  box-shadow: 4px 4px 0 0 #000;
  padding: 1.5rem;
  margin-bottom: 1.5rem;
}
.why-grid { display: flex; flex-direction: column; gap: 0.6rem; margin-top: 0.85rem; }
.why-row {
  display: grid;
  grid-template-columns: 9rem 1fr auto;
  gap: 1rem;
  align-items: center;
  padding: 0.55rem 0;
  border-bottom: 1px dashed #ccc;
}
.why-row:last-child { border-bottom: none; }
@media (max-width: 700px) {
  .why-row { grid-template-columns: 1fr; gap: 0.25rem; }
  .why-row .src { justify-self: start; }
}
.why-row .lbl {
  font-family: 'Cabinet Grotesk', sans-serif;
  font-weight: 800;
  font-size: 0.85rem;
  letter-spacing: -0.01em;
  text-transform: uppercase;
}
.why-row .val {
  font-weight: 600;
  font-size: 0.92rem;
  word-break: break-word;
}
.why-row .src {
  display: inline-flex;
  align-items: center;
  font-family: 'Satoshi', sans-serif;
  font-weight: 800;
  font-size: 0.62rem;
  letter-spacing: 0.12em;
  padding: 0.22rem 0.55rem;
  border: 2px solid #000;
  border-radius: 0.35rem;
  white-space: nowrap;
  text-transform: uppercase;
}
.src.tag-REAL       { background: #b6f0c5; }
.src.tag-INFERRED   { background: var(--sage); }
.src.tag-CONFIGURED { background: #e6e6e6; }
.src.tag-OFFLINE    { background: #ff9b9b; color: #000; }

.why-narrative {
  margin-top: 1.25rem;
  padding: 1rem 1.2rem;
  background: var(--char);
  color: #fff;
  border: 2px solid #000;
  border-radius: 0.5rem;
  font-weight: 600;
  line-height: 1.55;
  font-size: 0.95rem;
}
.why-narrative::before {
  content: "/ Narrative";
  display: block;
  font-family: 'Satoshi', sans-serif;
  font-weight: 700;
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--yellow);
  margin-bottom: 0.5rem;
}
.why-source-line {
  font-size: 0.7rem;
  font-weight: 600;
  color: #666;
  letter-spacing: 0.04em;
  margin-top: 0.18rem;
}

.gemini-badge {
  display: inline-block;
  font-family: 'Geist Mono', 'Satoshi', monospace;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  background: linear-gradient(135deg, #fef3c7 0%, #fbbf24 100%);
  color: #000;
  padding: 4px 10px;
  border: 2px solid #000;
  border-radius: 4px;
  box-shadow: 3px 3px 0 0 #000;
  margin-top: 10px;
}
.template-badge {
  display: inline-block;
  font-family: 'Geist Mono', 'Satoshi', monospace;
  font-size: 9px;
  font-weight: 700;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: rgba(255,255,255,0.45);
  margin-top: 10px;
}
.cascade-callouts .gemini-badge,
.cascade-callouts .template-badge { margin-top: 12px; }
.cascade-callouts .template-badge { color: rgba(255,255,255,0.45); }
.fleet-narrative-block {
  margin-bottom: 1rem;
  padding: 1rem 1.2rem;
  background: var(--char);
  color: #fff;
  border: 2px solid #000;
  border-radius: 0.5rem;
  font-weight: 600;
  line-height: 1.55;
  font-size: 0.92rem;
}
.fleet-narrative-block::before {
  content: "/ Executive summary";
  display: block;
  font-family: 'Satoshi', sans-serif;
  font-weight: 700;
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--yellow);
  margin-bottom: 0.5rem;
}

/* === FLEET DISPATCH ======================================== */
.fleet-console {
  background: #fff;
  color: #000;
  border: 2px solid #000;
  border-radius: 1.25rem;
  box-shadow: 8px 8px 0 0 #000;
  padding: 2.25rem;
}
@media (max-width: 600px) { .fleet-console { padding: 1.5rem; } }

.fleet-buttons {
  display: flex; flex-wrap: wrap; gap: 1rem;
  margin: 1.25rem 0;
}
.fleet-buttons .btn-push.gray {
  background: #fff; color: #000;
  box-shadow: 4px 4px 0 0 #000;
}

.fleet-metrics {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1rem;
  margin-top: 2rem;
}
@media (max-width: 800px) { .fleet-metrics { grid-template-columns: repeat(2, 1fr); } }
.fleet-metric {
  background: var(--yellow);
  border: 2px solid #000;
  border-radius: 0.75rem;
  padding: 1.25rem;
  box-shadow: 4px 4px 0 0 #000;
}
.fleet-metric.dark { background: var(--char); color: #fff; }
.fleet-metric.sage { background: var(--sage); }
.fleet-metric .num {
  font-family: 'Cabinet Grotesk', sans-serif;
  font-weight: 800;
  font-size: 2.6rem;
  line-height: 1;
  letter-spacing: -0.04em;
  margin-top: 0.5rem;
}
.fleet-metric .lbl {
  font-size: 0.7rem; font-weight: 800;
  text-transform: uppercase; letter-spacing: 0.1em;
}

.fleet-tables {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.25rem;
  margin-top: 2rem;
}
@media (max-width: 900px) { .fleet-tables { grid-template-columns: 1fr; } }
.fleet-table-card {
  background: #fff;
  border: 2px solid #000;
  border-radius: 0.85rem;
  box-shadow: 4px 4px 0 0 #000;
  overflow: hidden;
}
.fleet-table-card.smart-col { background: var(--yellow); }
.fleet-table-head {
  padding: 1rem 1.25rem;
  border-bottom: 2px solid #000;
  background: #000;
  color: var(--yellow);
}
.fleet-table-card.smart-col .fleet-table-head { background: #000; color: var(--yellow); }
.fleet-table-head .lbl {
  font-family: 'Cabinet Grotesk', sans-serif;
  font-weight: 800; font-size: 1.4rem;
  letter-spacing: -0.02em;
}
.fleet-table-head .stats {
  display: flex; gap: 1rem; margin-top: 0.4rem;
  font-size: 0.7rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: 0.1em;
}
.fleet-rows { padding: 0; }
.fleet-row {
  display: grid;
  grid-template-columns: 5rem 1fr auto;
  gap: 0.75rem;
  align-items: center;
  padding: 0.75rem 1rem;
  border-bottom: 1px dashed rgba(0,0,0,0.2);
  font-size: 0.85rem;
}
.fleet-row:last-child { border-bottom: none; }
.fleet-row .vid {
  font-family: 'Cabinet Grotesk', sans-serif;
  font-weight: 800;
  font-size: 0.95rem;
  letter-spacing: -0.02em;
}
.fleet-row .od {
  font-weight: 600; line-height: 1.35;
}
.fleet-row .od .meta {
  font-size: 0.72rem; font-weight: 600;
  color: rgba(0,0,0,0.55);
  margin-top: 0.15rem;
  font-family: 'Satoshi', sans-serif;
}
.fleet-row.late .meta { color: #b80000; font-weight: 800; }
.fleet-row .pill {
  display: inline-block;
  padding: 0.22rem 0.6rem;
  border: 2px solid #000;
  border-radius: 0.4rem;
  font-family: 'Satoshi', sans-serif;
  font-size: 0.65rem;
  font-weight: 800;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  white-space: nowrap;
}
.pill.ON_TIME { background: #b6f0c5; color: #000; }
.pill.AT_RISK { background: var(--yellow); color: #000; }
.pill.LATE    { background: #ff9b9b; color: #000; }

.cascade-callouts {
  background: var(--char);
  color: #fff;
  border: 2px solid #000;
  border-radius: 0.85rem;
  padding: 1.5rem;
  margin-top: 1.5rem;
}
.cascade-callouts h3 {
  color: var(--yellow);
  font-size: 1.25rem;
  margin-bottom: 0.85rem;
}
.cascade-row {
  padding: 0.85rem 0;
  border-bottom: 1px dashed rgba(255,255,255,0.18);
  line-height: 1.55;
  font-weight: 500;
  font-size: 0.95rem;
}
.cascade-row:last-child { border-bottom: none; }
.cascade-row strong { color: var(--yellow); }

.cal-pill {
  display: inline-flex; align-items: center; gap: 0.4rem;
  background: #fff;
  border: 2px solid #000;
  padding: 0.35rem 0.8rem;
  border-radius: 9999px;
  font-size: 0.78rem;
  font-weight: 700;
  box-shadow: 3px 3px 0 0 #000;
  margin-right: 0.5rem;
  margin-bottom: 0.5rem;
}

/* === FEATURE GRID ========================================== */
.feat-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem; }
@media (max-width: 900px) { .feat-grid { grid-template-columns: 1fr; } }
.feat-card {
  background: #fff;
  border: 2px solid #000;
  border-radius: 0.75rem;
  box-shadow: 4px 4px 0 0 #000;
  padding: 1.85rem;
  transition: all 0.2s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}
.feat-card:hover { transform: translate(4px, 4px); box-shadow: 0 0 0 0 #000; }
.feat-icon {
  width: 64px; height: 64px;
  background: var(--sage);
  border: 2px solid #000;
  border-radius: 0.5rem;
  display: flex; align-items: center; justify-content: center;
  margin-bottom: 1.25rem;
  transition: background 0.2s;
}
.feat-card:hover .feat-icon { background: var(--yellow); }
.feat-card h3 { font-size: 1.55rem; margin: 0.5rem 0 0.65rem; }
.feat-card p { line-height: 1.55; font-weight: 500; font-size: 0.95rem; }

/* === HOW IT WORKS ========================================== */
.steps-wrap { position: relative; margin-top: 3rem; }
.steps-line {
  position: absolute;
  left: 12%; right: 12%;
  top: 48px;
  height: 3px;
  background: var(--gray);
  z-index: 0;
}
@media (max-width: 900px) { .steps-line { display: none; } }
.steps {
  display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem;
  position: relative; z-index: 1;
}
@media (max-width: 900px) { .steps { grid-template-columns: 1fr; gap: 2.5rem; } }
.step { text-align: center; padding: 0 0.75rem; }
.step-circle {
  width: 96px; height: 96px;
  border-radius: 50%;
  background: var(--char);
  display: flex; align-items: center; justify-content: center;
  margin: 0 auto 1.25rem;
  font-family: 'Cabinet Grotesk', sans-serif;
  font-weight: 800;
  font-size: 2rem;
  color: #fff;
  border: 4px solid var(--sage);
}
.step:nth-child(2) .step-circle { border-color: var(--yellow); }
.step:nth-child(3) .step-circle { border-color: #fff; }
.step h3 { color: #fff; font-size: 1.5rem; margin-bottom: 0.6rem; }
.step p { color: var(--sage); font-weight: 500; line-height: 1.5; font-size: 0.95rem; }

/* === CITIES (USE CASE) ==================================== */
.cities-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.75rem; }
@media (max-width: 900px) { .cities-grid { grid-template-columns: 1fr; } }
.city-card {
  padding: 2rem;
  border: 2px solid #000;
  border-radius: 0.85rem;
  min-height: 340px;
  display: flex; flex-direction: column;
}
.city-card.sage { background: var(--sage); box-shadow: 4px 4px 0 0 #000; }
.city-card.yel { background: var(--yellow); box-shadow: 8px 8px 0 0 #000; }
.city-card.dark { background: var(--gray); color: #fff; box-shadow: 4px 4px 0 0 #000; }
.persona-pill {
  background: #fff; color: #000;
  border: 2px solid #000;
  padding: 0.32rem 0.8rem;
  border-radius: 9999px;
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  align-self: flex-start;
  margin-bottom: 1.5rem;
}
.city-stat {
  font-family: 'Cabinet Grotesk', sans-serif;
  font-weight: 800;
  font-size: 4.25rem;
  line-height: 0.95;
  letter-spacing: -0.05em;
  margin-bottom: 0.35rem;
}
.city-stat-pct { font-size: 2.5rem; }
.city-meta {
  font-size: 0.82rem; line-height: 1.5;
  font-weight: 600;
  margin-top: auto;
  padding-top: 1.25rem;
}

/* === RECEIPTS ============================================= */
.test-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.75rem; }
@media (max-width: 900px) { .test-grid { grid-template-columns: 1fr; } }
.test-card {
  background: #fff;
  padding: 2rem;
  border: 2px solid #000;
  border-radius: 2px 1.5rem 2px 1.5rem;
  box-shadow: 4px 4px 0 0 #000;
}
.stars { color: var(--star); margin-bottom: 1rem; font-size: 1.15rem; letter-spacing: 0.18rem; }
.test-card h3 { font-size: 1.4rem; line-height: 1.15; margin-bottom: 0.85rem; }
.test-card .body { line-height: 1.55; font-weight: 500; margin-bottom: 1.5rem; font-size: 0.95rem; }
.test-meta { display: flex; align-items: center; gap: 0.85rem; }
.test-avatar {
  width: 42px; height: 42px;
  background: var(--yellow);
  border: 2px solid #000;
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-family: 'Cabinet Grotesk', sans-serif;
  font-weight: 800;
  font-size: 1.2rem;
}

/* === FINAL CTA ============================================ */
.cta-final { text-align: center; padding: 6rem 0; }
.cta-final h2 { font-size: clamp(3rem, 10vw, 8rem); margin-bottom: 2rem; }

/* === FOOTER =============================================== */
.footer { background: var(--char); color: #fff; padding: 4rem 0 2rem; border-top: 2px solid #000; }
.footer-grid { display: grid; grid-template-columns: 2fr 1fr 1fr 1fr; gap: 2rem; margin-bottom: 3rem; }
@media (max-width: 900px) { .footer-grid { grid-template-columns: repeat(2, 1fr); } }
.footer-col h4 { color: var(--yellow); font-size: 0.95rem; margin-bottom: 1rem; }
.footer-col a {
  color: var(--sage); display: block;
  padding: 0.3rem 0;
  font-size: 0.88rem;
  font-weight: 500;
}
.footer-col a:hover { color: var(--yellow); }
.footer-tagline { color: var(--sage); font-size: 0.88rem; line-height: 1.55; margin-top: 1rem; max-width: 22rem; }
.social { display: flex; gap: 0.5rem; margin-top: 1.25rem; }
.social-sq {
  width: 40px; height: 40px;
  background: var(--gray);
  border: 2px solid #888;
  border-radius: 0.4rem;
  display: flex; align-items: center; justify-content: center;
  color: #fff;
  transition: all 0.18s;
}
.social-sq:hover {
  background: var(--yellow);
  color: #000;
  border-color: #000;
  transform: translate(-2px, -2px);
  box-shadow: 4px 4px 0 0 #000;
}
.footer-bottom {
  border-top: 2px solid var(--gray);
  padding-top: 1.5rem;
  display: flex; justify-content: space-between;
  flex-wrap: wrap; gap: 1rem;
  color: var(--sage);
  font-size: 0.82rem;
  font-weight: 600;
}

.muted { color: #555; }
.hidden { display: none !important; }

</style>
</head>
<body>

<!-- ============== NAV ============== -->
<header class="nav">
  <div class="container nav-inner">
    <a href="#" class="logo">
      <div class="logo-mark">
        <svg viewBox="0 0 24 24" fill="#ffe17c" stroke="#ffe17c" stroke-width="1.5" stroke-linejoin="round">
          <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
        </svg>
      </div>
      <div class="logo-text">DODOmaps</div>
    </a>
    <nav class="nav-links">
      <a href="#system">System</a>
      <a href="#demo">Scenarios</a>
      <a href="#manual">Manual</a>
      <a href="#fleet">Fleet</a>
    </nav>
    <a href="#demo" class="btn-push sm">Run Demo →</a>
  </div>
</header>

<!-- ============== HERO ============== -->
<section class="hero bg-yellow">
  <div class="container hero-grid">
    <div>
      <div class="badge"><span class="dot"></span> Predictive Routing v3.0 · Live</div>
      <h1 class="h-display" style="margin-top: 1.5rem;">
        Routes that<br/>
        <span class="stroke-text">anticipate.</span>
      </h1>
      <p style="margin-top: 1.5rem; font-size: 1.1rem; max-width: 32rem; line-height: 1.55; font-weight: 500;">
        A predictive routing engine for Indian logistics. Real-time conditions,
        city-aware event calendars, and per-fleet learning — fused into one graph.
      </p>
      <div style="margin-top: 2rem; display: flex; gap: 1.25rem; flex-wrap: wrap;">
        <a href="#demo" class="btn-push">Run Live Demo →</a>
        <a href="#system" class="btn-push white sm">See The System</a>
      </div>
      <div class="hero-stats" style="margin-top: 3rem;">
        <div class="hero-stat"><div class="num" id="hero-avg">—</div><div class="lbl">Avg time saved</div></div>
        <div class="hero-stat"><div class="num" id="hero-best">—</div><div class="lbl">Best case</div></div>
        <div class="hero-stat"><div class="num" id="hero-trips">—</div><div class="lbl">Trips learned</div></div>
      </div>
    </div>
    <div>
      <div class="mockup">
        <div class="mockup-bar">
          <div class="mockup-dot" style="background: #ff5f57;"></div>
          <div class="mockup-dot" style="background: #febc2e;"></div>
          <div class="mockup-dot" style="background: #28c840;"></div>
          <div class="mockup-url">obsidian.app/console</div>
        </div>
        <div class="mockup-body">
          <div class="mockup-grid">
            <div class="mockup-card" style="background: var(--char); color: #fff;">
              <div class="lbl">Today's avg saved</div>
              <div class="num">12.4%</div>
              <svg viewBox="0 0 200 60" style="width: 100%; height: 56px; margin-top: 0.4rem;">
                <polyline points="0,40 25,30 50,35 75,20 100,25 125,15 150,18 175,8 200,12" fill="none" stroke="#ffe17c" stroke-width="2.5"/>
                <polyline points="0,55 25,50 50,52 75,48 100,50 125,45 150,47 175,40 200,42" fill="none" stroke="#b7c6c2" stroke-width="2" stroke-dasharray="4 3" opacity="0.7"/>
              </svg>
            </div>
            <div style="display: flex; flex-direction: column; gap: 0.75rem;">
              <div class="mockup-card" style="background: var(--sage);">
                <div class="lbl">Live</div>
                <div class="num">3 cities</div>
              </div>
              <div class="mockup-card" style="background: var(--yellow);">
                <div class="lbl">Reroutes</div>
                <div class="num">+18 today</div>
              </div>
            </div>
          </div>
          <div class="mockup-card" style="margin-top: 0.75rem; background: #fff; display: flex; justify-content: space-between; align-items: center;">
            <div>
              <div class="lbl">Active scenario</div>
              <div style="font-weight: 700; font-size: 0.92rem; margin-top: 0.15rem;">Pune · IPL match day</div>
            </div>
            <div style="background: #000; color: var(--yellow); padding: 0.32rem 0.6rem; border-radius: 0.3rem; font-weight: 800; font-size: 0.7rem; letter-spacing: 0.05em;">REROUTED</div>
          </div>
        </div>
      </div>
    </div>
  </div>
</section>

<!-- ============== MARQUEE ============== -->
<section class="section-char marquee-section">
  <div class="marquee">
    <div class="marquee-track">
      <span>PUNE</span><span class="star">✦</span>
      <span>MUMBAI</span><span class="star">✦</span>
      <span>BANGALORE</span><span class="star">✦</span>
      <span>A* + EWMA</span><span class="star">✦</span>
      <span>CALENDAR LAYER</span><span class="star">✦</span>
      <span>PREDICTIVE</span><span class="star">✦</span>
      <span>FLEET-LEVEL</span><span class="star">✦</span>
      <span>SUB-SECOND</span><span class="star">✦</span>
      <span>PUNE</span><span class="star">✦</span>
      <span>MUMBAI</span><span class="star">✦</span>
      <span>BANGALORE</span><span class="star">✦</span>
      <span>A* + EWMA</span><span class="star">✦</span>
      <span>CALENDAR LAYER</span><span class="star">✦</span>
      <span>PREDICTIVE</span><span class="star">✦</span>
      <span>FLEET-LEVEL</span><span class="star">✦</span>
      <span>SUB-SECOND</span><span class="star">✦</span>
    </div>
  </div>
</section>

<!-- ============== PROBLEM VS SOLUTION ============== -->
<section id="system" style="background: #fff;">
  <div class="container">
    <div class="section-eyebrow-row"><span class="bar"></span><span class="eyebrow">The System</span></div>
    <h2 class="h-section" style="max-width: 40rem; margin-bottom: 3rem;">Why generic maps <span class="stroke-text">break</span> in India.</h2>
    <div class="ps-grid">
      <div class="ps-card ps-problem">
        <div class="eyebrow muted">Generic Routing</div>
        <h3 style="font-size: 1.85rem; margin-top: 0.5rem;">Global, blind, untuned.</h3>
        <ul class="ps-list">
          <li><span class="ps-icon x">✕</span> Doesn't know Karnataka Bandh on Feb 14</li>
          <li><span class="ps-icon x">✕</span> Trains on the world's data, not your fleet's</li>
          <li><span class="ps-icon x">✕</span> Optimizes one driver's next turn, not dispatch</li>
          <li><span class="ps-icon x">✕</span> Same engine for Pune, Mumbai, Tokyo, Berlin</li>
        </ul>
      </div>
      <div class="ps-card ps-solution">
        <div class="eyebrow">DODOmaps</div>
        <h3 style="font-size: 1.85rem; margin-top: 0.5rem;">City-aware, fleet-tuned, predictive.</h3>
        <ul class="ps-list">
          <li><span class="ps-icon check">✓</span> Per-city event calendars: Ganpati, IPL, monsoon</li>
          <li><span class="ps-icon check">✓</span> EWMA learning loop on every fleet trip</li>
          <li><span class="ps-icon check">✓</span> Fleet-level dispatch, not turn-by-turn</li>
          <li><span class="ps-icon check">✓</span> New city = config file, not engine rewrite</li>
        </ul>
      </div>
    </div>
  </div>
</section>

<!-- ============== LIVE DEMO ============== -->
<section id="demo" class="bg-yellow" style="border-top: 2px solid #000; border-bottom: 2px solid #000;">
  <div class="container">
    <div class="section-eyebrow-row"><span class="bar"></span><span class="eyebrow">Live Engine</span></div>
    <h2 class="h-section" style="margin-bottom: 2.5rem;">Run it <span class="stroke-text">yourself.</span></h2>
    <div class="demo-console">
      <div class="demo-step">
        <div class="demo-step-label">01 / Select City</div>
        <div id="city-tabs" class="tab-row"></div>
      </div>
      <div class="demo-step">
        <div class="demo-step-label">02 / Select Scenario</div>
        <div id="scenario-tabs" class="tab-row scenario-row"></div>
      </div>
      <div class="demo-step">
        <button id="compute-btn" class="btn-push">⚡ Compute Route</button>
      </div>
    </div>
  </div>
</section>

<!-- ============== MANUAL MODE ============== -->
<section id="manual" class="section-char">
  <div class="container">
    <div class="section-eyebrow-row"><span class="bar"></span><span class="eyebrow">Manual Mode</span></div>
    <h2 class="h-section" style="color: #fff; margin-bottom: 0.85rem;">Type it. <span class="stroke-text-yel">Trust it.</span></h2>
    <p style="color: var(--sage); font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase; font-size: 0.82rem; margin-bottom: 2.25rem;">Pick two points · System derives every condition · Read every assumption</p>
    <div class="manual-console">
      <div class="manual-grid">
        <div>
          <div class="demo-step-label">01 / Type Start</div>
          <input id="m-start" class="brut-input" type="text" value="Swargate" placeholder="e.g. Swargate" autocomplete="off"/>
        </div>
        <div>
          <div class="demo-step-label">02 / Type End</div>
          <input id="m-end" class="brut-input" type="text" value="Hinjewadi" placeholder="e.g. Hinjewadi" autocomplete="off"/>
        </div>
      </div>
      <div style="margin-bottom: 1.5rem;">
        <div class="demo-step-label">03 / Date (optional)</div>
        <select id="m-date" class="brut-select" style="max-width: 28rem;">
          <option value="">Today</option>
          <option value="2026-05-02">IPL match day · May 2, 2026</option>
          <option value="2026-09-18">Ganpati Visarjan · Sep 18, 2026</option>
          <option value="2026-12-06">Pune Marathon · Dec 6, 2026</option>
        </select>
      </div>
      <button id="manual-btn" class="btn-push">⚡ Compute Route</button>
      <div id="m-error" class="hidden" style="margin-top: 1.25rem; padding: 0.85rem 1rem; background: #fff; border: 2px solid #000; border-radius: 0.4rem; box-shadow: 4px 4px 0 0 #000; color: #000; font-weight: 700;"></div>
      <div style="margin-top: 1.25rem; color: var(--sage); font-size: 0.78rem; font-weight: 600; line-height: 1.5;">
        Conditions are derived live from your system clock, Open-Meteo weather, the city's calendar, and the learning store. Every value is shown in the result with its source.
      </div>
    </div>
  </div>
</section>

<!-- ============== FLEET DISPATCH ============== -->
<section id="fleet" class="bg-yellow" style="border-top: 2px solid #000; border-bottom: 2px solid #000;">
  <div class="container">
    <div class="section-eyebrow-row"><span class="bar"></span><span class="eyebrow">Fleet Dispatch · Cascade-Aware Optimization</span></div>
    <h2 class="h-section" style="margin-bottom: 0.85rem;">Twelve trucks. <span class="stroke-text">One day.</span></h2>
    <p style="font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase; font-size: 0.82rem; margin-bottom: 2.25rem;">Fleet dispatch · Cascade-aware optimization · 12 shipments · 4 vehicles</p>
    <div class="fleet-console">
      <div class="manual-grid" style="margin-bottom: 0.5rem;">
        <div>
          <div class="demo-step-label">/ Date</div>
          <select id="f-date" class="brut-select">
            <option value="">Today</option>
            <option value="2026-05-02" selected>IPL match day · May 2, 2026</option>
            <option value="2026-09-18">Ganpati Visarjan · Sep 18, 2026</option>
            <option value="2026-12-06">Pune Marathon · Dec 6, 2026</option>
          </select>
        </div>
        <div>
          <div class="demo-step-label">/ Graph</div>
          <select id="f-graph" class="brut-select">
            <option value="pune_real" selected>Real Pune (OSM, 44k nodes)</option>
            <option value="pune">Hand-built (19 nodes · cascade demo)</option>
          </select>
        </div>
      </div>
      <div id="f-graph-meta" style="font-size: 0.75rem; font-weight: 600; color: #555; margin-bottom: 1rem;"></div>
      <div class="fleet-buttons">
        <button id="f-baseline-btn" class="btn-push gray sm">Run Baseline</button>
        <button id="f-smart-btn"    class="btn-push yellow sm">Run Smart</button>
        <button id="f-compare-btn"  class="btn-push">⚡ Run Comparison</button>
      </div>
      <div id="f-error" class="hidden" style="margin-top: 1rem; padding: 0.85rem 1rem; background: #fff; border: 2px solid #000; border-radius: 0.4rem; box-shadow: 4px 4px 0 0 #000; color: #000; font-weight: 700;"></div>
      <div style="color: #555; font-size: 0.78rem; font-weight: 600; line-height: 1.5; margin-top: 0.85rem;">
        Each vehicle runs its shipments sequentially. A late shipment shifts the next one's start time by the same amount — that's the cascade. Smart routing detours around disrupted edges before the cascade propagates.
      </div>

      <!-- Renders only after a comparison runs -->
      <div id="f-output" class="hidden" style="margin-top: 1.75rem;">
        <div id="f-narrative" class="fleet-narrative-block hidden">
          <div id="f-narrative-text"></div>
          <div id="f-narrative-badge"></div>
        </div>
        <div id="f-metrics" class="fleet-metrics"></div>
        <div id="f-tables" class="fleet-tables"></div>
        <div id="f-callouts" class="cascade-callouts hidden">
          <h3>Cascade callouts</h3>
          <div id="f-callout-list"></div>
        </div>
        <div id="f-single-note" class="hidden" style="margin-top: 1rem; font-size: 0.85rem; color: #555; font-weight: 600;"></div>
      </div>
    </div>
  </div>
</section>

<!-- ============== SHARED RESULTS PANEL ============== -->
<section id="results-section" style="background: #fff;">
  <div class="container">
    <div id="loading" class="hidden">
      <div style="font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; font-size: 0.9rem;">⚡ Computing route...</div>
    </div>
    <div id="results-empty">
      <div class="section-eyebrow-row"><span class="bar"></span><span class="eyebrow">Result</span></div>
      <h2 class="h-section muted" style="opacity: 0.4; max-width: 36rem;">Run a scenario or set conditions <span class="stroke-text" style="-webkit-text-stroke-color:#bbb;">above.</span></h2>
    </div>
    <div id="results" class="hidden">
      <div class="section-eyebrow-row"><span class="bar"></span><span class="eyebrow">Result</span></div>
      <div class="results-head" style="margin-bottom: 2rem;">
        <div>
          <div class="eyebrow muted">Scenario</div>
          <h3 style="font-size: 2rem; margin-top: 0.35rem;" id="r-scenario">—</h3>
          <div class="muted" style="font-weight: 600; font-size: 0.85rem; margin-top: 0.25rem;" id="r-date">—</div>
        </div>
        <div style="text-align: right;">
          <div class="eyebrow muted">Time saved</div>
          <div class="saved-pct" id="r-saved-pct">—</div>
          <div class="muted" style="font-weight: 600; margin-top: 0.25rem; font-size: 0.85rem;" id="r-saved-min">—</div>
        </div>
      </div>
      <div id="r-calendar" class="hidden" style="margin-bottom: 1.5rem;">
        <div class="eyebrow muted" style="margin-bottom: 0.6rem;">Calendar events active</div>
        <div id="r-calendar-list"></div>
      </div>
      <div id="r-why" class="why-card hidden">
        <div class="section-eyebrow-row" style="margin-bottom: 0.4rem;"><span class="bar"></span><span class="eyebrow">Why this route</span></div>
        <div id="r-why-grid" class="why-grid"></div>
        <div id="r-why-narrative" class="why-narrative"></div>
        <div id="r-why-narrative-badge"></div>
      </div>
      <div class="route-grid">
        <div class="route-card base">
          <div class="eyebrow">Baseline · Shortest path</div>
          <div class="route-time" id="r-base-time">—</div>
          <div style="font-size: 0.78rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.08em;">Minutes</div>
          <div class="route-path" id="r-base-path"></div>
        </div>
        <div class="route-card smart">
          <div class="eyebrow">Smart · Predictive ⚡</div>
          <div class="route-time" id="r-smart-time">—</div>
          <div style="font-size: 0.78rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.08em;">Minutes</div>
          <div class="route-path" id="r-smart-path"></div>
        </div>
      </div>
      <div class="map-card">
        <div class="eyebrow muted" style="margin-bottom: 0.65rem;">Route comparison</div>
        <svg id="r-map" viewBox="0 0 800 400"></svg>
      </div>
      <div class="bottom-grid">
        <div class="decision-card">
          <div class="eyebrow" style="color: var(--yellow);">Decision</div>
          <h3 id="r-decision" style="font-size: 1.85rem; color: #fff; margin-top: 0.5rem;">—</h3>
        </div>
        <div class="signals-card">
          <div class="eyebrow" style="color: var(--yellow); margin-bottom: 0.85rem;">Signals on baseline path</div>
          <div id="r-signals"></div>
        </div>
      </div>
    </div>
  </div>
</section>

<!-- ============== FEATURE GRID ============== -->
<section style="background: var(--yellow); border-top: 2px solid #000; border-bottom: 2px solid #000;">
  <div class="container">
    <div class="section-eyebrow-row"><span class="bar"></span><span class="eyebrow">Three signals · One decision</span></div>
    <h2 class="h-section" style="margin-bottom: 3rem;">Built for what <span class="stroke-text">Google can't see.</span></h2>
    <div class="feat-grid">
      <div class="feat-card">
        <div class="feat-icon">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#000" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
        </div>
        <div class="eyebrow muted">01 · Real-time</div>
        <h3>Now.</h3>
        <p>Live traffic and weather feed into edge weights every cycle. Highway segments weighted differently from local roads.</p>
      </div>
      <div class="feat-card">
        <div class="feat-icon">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#000" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="18" rx="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg>
        </div>
        <div class="eyebrow muted">02 · Calendar</div>
        <h3>Soon.</h3>
        <p>Per-city event calendars: Ganpati, Marathon, IPL match days, monsoon peaks, bandhs. Auto-disrupts on date.</p>
      </div>
      <div class="feat-card">
        <div class="feat-icon">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#000" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 1 1-9-9c2.49 0 4.77.94 6.5 2.5"/><polyline points="21 4 21 12 13 12"/></svg>
        </div>
        <div class="eyebrow muted">03 · Learning</div>
        <h3>Always.</h3>
        <p>Every trip updates per-edge delay factors via EWMA. Per-fleet, per-city. The data flywheel.</p>
      </div>
    </div>
  </div>
</section>

<!-- ============== FINAL CTA ============== -->
<section class="bg-yellow cta-final" style="border-top: 2px solid #000;">
  <div class="container">
    <h2>Reroute <span class="stroke-text">before</span><br/>the jam.</h2>
    <a href="#demo" class="btn-push">Run the live demo →</a>
  </div>
</section>

<!-- ============== FOOTER ============== -->
<footer class="footer">
  <div class="container">
    <div class="footer-grid">
      <div class="footer-col">
        <div class="logo" style="margin-bottom: 0.5rem;">
          <div class="logo-mark">
            <svg viewBox="0 0 24 24" fill="#ffe17c" stroke="#ffe17c" stroke-width="1.5" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
          </div>
          <div class="logo-text" style="color: #fff;">DODOmaps</div>
        </div>
        <p class="footer-tagline">A* + EWMA + calendar. Predictive routing for Indian fleets. Built in Pune.</p>
        <div class="social">
          <a href="#" class="social-sq" aria-label="Twitter"><svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M22 4.01c-1 .49-1.98.689-3 .99-1.121-1.265-2.783-1.335-4.38-.737S11.977 6.323 12 8v1c-3.245.083-6.135-1.395-8-4 0 0-4.182 7.433 4 11-1.872 1.247-3.739 2.088-6 2 3.308 1.803 6.913 2.423 10.034 1.517 3.58-1.04 6.522-3.723 7.651-7.742a13.84 13.84 0 0 0 .497-3.753c-.003-.249 1.51-2.772 1.818-4.013z"/></svg></a>
          <a href="#" class="social-sq" aria-label="GitHub"><svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0 1 12 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.852 0 1.336-.012 2.415-.012 2.743 0 .267.18.578.688.48C19.138 20.193 22 16.44 22 12.017 22 6.484 17.522 2 12 2z"/></svg></a>
          <a href="#" class="social-sq" aria-label="LinkedIn"><svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/></svg></a>
        </div>
      </div>
      <div class="footer-col">
        <h4>Product</h4>
        <a href="#demo">Scenario Mode</a>
        <a href="#manual">Manual Mode</a>
        <a href="#fleet">Fleet Dispatch</a>
        <a href="#system">The System</a>
      </div>
      <div class="footer-col">
        <h4>Company</h4>
        <a href="#">About</a>
        <a href="#">Careers</a>
        <a href="#">Press Kit</a>
      </div>
      <div class="footer-col">
        <h4>Legal</h4>
        <a href="#">Privacy</a>
        <a href="#">Terms</a>
        <a href="#">Security</a>
      </div>
    </div>
    <div class="footer-bottom">
      <div>DODOmaps © 2026 · Built in Pune</div>
      <div>A* + EWMA + Calendar</div>
    </div>
  </div>
</footer>

<script>
  let CITIES = [];
  let SCENARIOS = [];
  let VALIDATION = null;
  let activeCity = "pune";
  let activeScenario = 0;

  // Stash for the most recent geocode results so renderResults can stamp
  // them onto the "Why this route" panel as REAL fact rows.
  let LAST_GEOCODE = { start: null, end: null };

  Promise.all([
    fetch("/api/cities").then(r => r.json()),
    fetch("/api/validation").then(r => r.json()),
  ]).then(([cities, validation]) => {
    CITIES = cities;
    VALIDATION = validation;
    renderHeroStats();
    renderCityTabs();
    loadScenarios(activeCity);
  }).catch(err => console.error("Init failed:", err));

  // --- Manual mode wiring -----------------------------------------------
  function geocode(query) {
    return fetch("/api/geocode", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, city: "pune_real" }),
    }).then(r => r.json());
  }

  document.getElementById("manual-btn").addEventListener("click", () => {
    const startQ = document.getElementById("m-start").value.trim();
    const endQ   = document.getElementById("m-end").value.trim();
    const dateOverride = document.getElementById("m-date").value || null;
    const errBox = document.getElementById("m-error");
    errBox.classList.add("hidden");
    if (!startQ || !endQ) {
      errBox.textContent = "Type both a start and end location.";
      errBox.classList.remove("hidden");
      return;
    }
    if (startQ.toLowerCase() === endQ.toLowerCase()) {
      errBox.textContent = "Start and end must differ.";
      errBox.classList.remove("hidden");
      return;
    }
    showLoading();

    // Geocode both endpoints in parallel, then route on pune_real.
    Promise.all([geocode(startQ), geocode(endQ)]).then(([gs, ge]) => {
      if (gs.error) throw new Error(`Start: ${gs.error}`);
      if (ge.error) throw new Error(`End: ${ge.error}`);
      LAST_GEOCODE = { start: gs, end: ge };
      return fetch("/api/manual_route", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          city: "pune_real",
          start_node: gs.nearest_node,
          end_node: ge.nearest_node,
          date_override: dateOverride,
        }),
      }).then(r => r.json());
    }).then(data => {
      if (data.error) {
        document.getElementById("loading").classList.add("hidden");
        document.getElementById("results-empty").classList.remove("hidden");
        errBox.textContent = data.error;
        errBox.classList.remove("hidden");
        return;
      }
      setTimeout(() => renderResults(data), 200);
    }).catch(e => {
      document.getElementById("loading").classList.add("hidden");
      document.getElementById("results-empty").classList.remove("hidden");
      errBox.textContent = e.message || String(e);
      errBox.classList.remove("hidden");
    });
  });

  // --- Fleet Dispatch wiring -------------------------------------------
  function fleetCity() {
    return document.getElementById('f-graph').value || 'pune_real';
  }
  function fleetBody() {
    return {
      city: fleetCity(),
      date_override: document.getElementById('f-date').value || null,
    };
  }

  function refreshFleetMeta() {
    const city = fleetCity();
    fetch(`/api/fleet/demo?city=${encodeURIComponent(city)}`)
      .then(r => r.json())
      .then(d => {
        const meta = document.getElementById('f-graph-meta');
        if (!d || d.error) { meta.textContent = ''; return; }
        const g = d.graph || {};
        const cascadeNote = (city === 'pune_real')
          ? ' \u00b7 Cascade demo is muted on the real graph (Hinjewadi has limited road alternatives in OSM); switch to hand-built for the 5\u21920 LATE story.'
          : ' \u00b7 Hand-built abstraction \u2014 the 5\u21920 LATE cascade demo runs here.';
        meta.textContent = `${g.display_name || city} \u00b7 ${(g.node_count||0).toLocaleString()} nodes \u00b7 ${(g.edge_count||0).toLocaleString()} edges${cascadeNote}`;
      })
      .catch(() => {});
  }
  document.getElementById('f-graph').addEventListener('change', refreshFleetMeta);
  refreshFleetMeta();

  function fleetSetButtonsLoading(loading) {
    ['f-baseline-btn','f-smart-btn','f-compare-btn'].forEach(id => {
      document.getElementById(id).disabled = loading;
    });
    document.getElementById('f-error').classList.add('hidden');
  }

  function fleetShowError(msg) {
    const e = document.getElementById('f-error');
    e.textContent = msg;
    e.classList.remove('hidden');
  }

  document.getElementById('f-baseline-btn').addEventListener('click', () => {
    fleetSetButtonsLoading(true);
    fetch('/api/fleet/simulate', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({...fleetBody(), mode: 'baseline'}),
    }).then(r => r.json()).then(d => {
      fleetSetButtonsLoading(false);
      if (d.error) return fleetShowError(d.error);
      renderFleetSingle(d, 'baseline');
    }).catch(e => { fleetSetButtonsLoading(false); fleetShowError(String(e)); });
  });

  document.getElementById('f-smart-btn').addEventListener('click', () => {
    fleetSetButtonsLoading(true);
    fetch('/api/fleet/simulate', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({...fleetBody(), mode: 'smart'}),
    }).then(r => r.json()).then(d => {
      fleetSetButtonsLoading(false);
      if (d.error) return fleetShowError(d.error);
      renderFleetSingle(d, 'smart');
    }).catch(e => { fleetSetButtonsLoading(false); fleetShowError(String(e)); });
  });

  document.getElementById('f-compare-btn').addEventListener('click', () => {
    fleetSetButtonsLoading(true);
    fetch('/api/fleet/compare', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify(fleetBody()),
    }).then(r => r.json()).then(d => {
      fleetSetButtonsLoading(false);
      if (d.error) return fleetShowError(d.error);
      renderFleetCompare(d);
    }).catch(e => { fleetSetButtonsLoading(false); fleetShowError(String(e)); });
  });

  function shipmentRow(s, isLate) {
    const marginClass = (s.margin < 0) ? 'late' : '';
    const marginText = (s.margin < 0)
      ? `LATE by ${Math.abs(s.margin).toFixed(1)} min`
      : `${s.margin.toFixed(1)} min margin`;
    return `
      <div class="fleet-row ${marginClass}">
        <div class="vid">${s.vehicle_id}/${s.shipment_id}</div>
        <div class="od">
          ${s.origin_name || s.origin_node} → ${s.destination_name || s.destination_node}
          <div class="meta">DL ${s.deadline_min} min · ETA ${s.end_time.toFixed(1)} min · ${marginText}</div>
        </div>
        <span class="pill ${s.status}">${s.status.replace('_',' ')}</span>
      </div>
    `;
  }

  function tableCard(title, fleetResult, smartCol) {
    const cls = smartCol ? 'fleet-table-card smart-col' : 'fleet-table-card';
    return `
      <div class="${cls}">
        <div class="fleet-table-head">
          <div class="lbl">${title}</div>
          <div class="stats">
            <span>${fleetResult.total_late} LATE</span>
            <span>${fleetResult.total_at_risk} AT-RISK</span>
            <span>${fleetResult.total_on_time} ON-TIME</span>
          </div>
        </div>
        <div class="fleet-rows">
          ${fleetResult.shipments.map(s => shipmentRow(s, s.status === 'LATE')).join('')}
        </div>
      </div>
    `;
  }

  function renderFleetCompare(d) {
    const out = document.getElementById('f-output');
    out.classList.remove('hidden');
    document.getElementById('f-single-note').classList.add('hidden');

    const narrWrap = document.getElementById('f-narrative');
    if (d.narrative) {
      document.getElementById('f-narrative-text').textContent = d.narrative;
      const src = d.narrative_source || 'template';
      document.getElementById('f-narrative-badge').innerHTML = src === 'gemini'
        ? '<span class="gemini-badge">\u2728 Generated by Gemini</span>'
        : '<span class="template-badge">Template fallback</span>';
      narrWrap.classList.remove('hidden');
    } else {
      narrWrap.classList.add('hidden');
    }

    document.getElementById('f-metrics').innerHTML = `
      <div class="fleet-metric"><div class="lbl">Baseline late</div><div class="num">${d.baseline.total_late}</div></div>
      <div class="fleet-metric sage"><div class="lbl">Smart late</div><div class="num">${d.smart.total_late}</div></div>
      <div class="fleet-metric dark"><div class="lbl" style="color: var(--yellow);">Recovered</div><div class="num">${d.savings.shipments_recovered}</div></div>
      <div class="fleet-metric"><div class="lbl">Late mins avoided</div><div class="num">${d.savings.late_minutes_avoided}</div></div>
    `;

    document.getElementById('f-tables').innerHTML =
      tableCard('Baseline Dispatch', d.baseline, false) +
      tableCard('Smart Dispatch', d.smart, true);

    const callouts = d.cascade_callouts || [];
    const cwrap = document.getElementById('f-callouts');
    if (callouts.length) {
      document.getElementById('f-callout-list').innerHTML =
        callouts.map(c => `<div class="cascade-row">${c.sentence}</div>`).join('');
      cwrap.classList.remove('hidden');
    } else {
      cwrap.classList.add('hidden');
    }

    out.scrollIntoView({behavior: 'smooth', block: 'nearest'});
  }

  function renderFleetSingle(d, mode) {
    const out = document.getElementById('f-output');
    out.classList.remove('hidden');
    document.getElementById('f-callouts').classList.add('hidden');
    document.getElementById('f-narrative').classList.add('hidden');

    document.getElementById('f-metrics').innerHTML = `
      <div class="fleet-metric"><div class="lbl">Late</div><div class="num">${d.total_late}</div></div>
      <div class="fleet-metric sage"><div class="lbl">At risk</div><div class="num">${d.total_at_risk}</div></div>
      <div class="fleet-metric"><div class="lbl">On time</div><div class="num">${d.total_on_time}</div></div>
      <div class="fleet-metric dark"><div class="lbl" style="color: var(--yellow);">Total mins late</div><div class="num">${d.total_minutes_late}</div></div>
    `;
    const title = mode === 'smart' ? 'Smart Dispatch' : 'Baseline Dispatch';
    document.getElementById('f-tables').innerHTML = tableCard(title, d, mode === 'smart');

    const note = document.getElementById('f-single-note');
    note.textContent = `Showing ${title} only. Click "Run Comparison" to see baseline vs smart side-by-side and the cascade callouts.`;
    note.classList.remove('hidden');

    out.scrollIntoView({behavior: 'smooth', block: 'nearest'});
  }

  function renderHeroStats() {
    const v = VALIDATION.cities;
    const allAvg = Object.values(v).map(c => c.avg_saved);
    const allBest = Object.values(v).map(c => c.best_case);
    const overallAvg = allAvg.reduce((a,b)=>a+b,0) / allAvg.length;
    const overallBest = Math.max(...allBest);
    document.getElementById("hero-avg").textContent = overallAvg.toFixed(1) + "%";
    document.getElementById("hero-best").textContent = overallBest.toFixed(1) + "%";
    document.getElementById("hero-trips").textContent = VALIDATION.total_trips_learned.toLocaleString();
  }

  function renderCityTabs() {
    const wrap = document.getElementById("city-tabs");
    wrap.innerHTML = CITIES.map(c => `
      <button data-city="${c.key}" class="city-tab tab ${c.key === activeCity ? "active" : ""}">${c.display_name}</button>
    `).join("");
    wrap.querySelectorAll(".city-tab").forEach(btn => {
      btn.addEventListener("click", () => {
        activeCity = btn.dataset.city;
        renderCityTabs();
        loadScenarios(activeCity);
      });
    });
  }

  function loadScenarios(cityKey) {
    fetch(`/api/scenarios/${cityKey}`).then(r => r.json()).then(data => {
      SCENARIOS = data;
      activeScenario = 0;
      renderScenarioTabs();
    });
  }

  function renderScenarioTabs() {
    const wrap = document.getElementById("scenario-tabs");
    wrap.innerHTML = SCENARIOS.map(s => `
      <button data-idx="${s.index}" class="scen-tab tab ${s.index === activeScenario ? "active" : ""}">${s.name}</button>
    `).join("");
    wrap.querySelectorAll(".scen-tab").forEach(btn => {
      btn.addEventListener("click", () => {
        activeScenario = +btn.dataset.idx;
        renderScenarioTabs();
      });
    });
  }

  function showLoading() {
    document.getElementById("results").classList.add("hidden");
    document.getElementById("results-empty").classList.add("hidden");
    document.getElementById("loading").classList.remove("hidden");
  }

  document.getElementById("compute-btn").addEventListener("click", () => {
    showLoading();
    fetch("/api/route", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ city: activeCity, scenario_index: activeScenario }),
    }).then(r => r.json()).then(data => {
      setTimeout(() => renderResults(data), 200);
    });
  });

  function renderResults(d) {
    document.getElementById("loading").classList.add("hidden");
    document.getElementById("results-empty").classList.add("hidden");
    const r = document.getElementById("results");
    r.classList.remove("hidden");

    document.getElementById("r-scenario").textContent = d.scenario;
    document.getElementById("r-date").textContent = `Date · ${d.date}`;

    const savedPctEl = document.getElementById("r-saved-pct");
    if (d.saved_pct > 0) {
      savedPctEl.textContent = d.saved_pct + "%";
      savedPctEl.style.color = "#000";
    } else {
      savedPctEl.textContent = "0%";
      savedPctEl.style.color = "#888";
    }
    document.getElementById("r-saved-min").textContent =
      d.saved_min > 0 ? `${d.saved_min} minutes saved` : "No reroute · baseline already optimal";

    document.getElementById("r-base-time").textContent = d.baseline.time_min;
    document.getElementById("r-smart-time").textContent = d.smart.time_min;
    // OSM intersection nodes have no readable name. If most path_names are empty,
    // render a path summary instead of an unreadable string of arrows.
    function summarisePath(p) {
      const named = p.path_names.filter(n => (n || '').trim());
      if (named.length >= Math.max(2, p.path_names.length / 2)) {
        return p.path_names.join("  \u2192  ");
      }
      return `${p.path_names.length} OSM intersections \u00b7 real-graph route`;
    }
    document.getElementById("r-base-path").textContent = summarisePath(d.baseline);
    document.getElementById("r-smart-path").textContent = summarisePath(d.smart);

    const calWrap = document.getElementById("r-calendar");
    if (d.calendar_events.length) {
      calWrap.classList.remove("hidden");
      document.getElementById("r-calendar-list").innerHTML = d.calendar_events.map(e => `
        <span class="cal-pill">${e.name} · ${e.severity}%</span>
      `).join("");
    } else {
      calWrap.classList.add("hidden");
    }

    // --- "Why this route" panel: only present in manual mode ---
    const whyCard = document.getElementById("r-why");
    if (d.assumptions && Array.isArray(d.assumptions.facts)) {
      const facts = d.assumptions.facts.slice();

      // If we just geocoded for this manual request, prepend a Geocoding fact
      // row so the user can see exactly which OSM points the queries snapped to.
      if (LAST_GEOCODE.start && LAST_GEOCODE.end) {
        const gs = LAST_GEOCODE.start, ge = LAST_GEOCODE.end;
        const geoVal =
          `${gs.query} \u2192 "${(gs.display_name || '').split(',')[0]}" (snap ${gs.node_distance_m}m); ` +
          `${ge.query} \u2192 "${(ge.display_name || '').split(',')[0]}" (snap ${ge.node_distance_m}m)`;
        facts.unshift({
          label: 'Geocoding',
          value: geoVal,
          source: 'Nominatim (OpenStreetMap) + nearest-node snap',
          tag: 'REAL',
        });
        // One-shot: don't restamp on the next non-manual render.
        LAST_GEOCODE = { start: null, end: null };
      }

      const grid = document.getElementById("r-why-grid");
      grid.innerHTML = facts.map(f => `
        <div class="why-row">
          <div class="lbl">${f.label}</div>
          <div>
            <div class="val">${f.value}</div>
            <div class="why-source-line">${f.source}</div>
          </div>
          <span class="src tag-${f.tag}">${f.tag}</span>
        </div>
      `).join("");
      document.getElementById("r-why-narrative").textContent =
        d.assumptions.narrative || "";
      const narrSrc = d.assumptions.narrative_source || 'template';
      const badgeEl = document.getElementById("r-why-narrative-badge");
      badgeEl.innerHTML = narrSrc === 'gemini'
        ? '<span class="gemini-badge">\u2728 Generated by Gemini</span>'
        : '<span class="template-badge">Template fallback</span>';
      whyCard.classList.remove("hidden");
    } else {
      whyCard.classList.add("hidden");
    }

    document.getElementById("r-decision").textContent = d.rerouted ? "Rerouted." : "Held route.";

    const sig = document.getElementById("r-signals");
    if (d.signals.length === 0) {
      sig.innerHTML = `<div style="opacity: 0.65; font-weight: 500;">No active signals on baseline path.</div>`;
    } else {
      sig.innerHTML = d.signals.map(s => `
        <div class="signal-row">
          <div class="signal-from">${s.from} → ${s.to}</div>
          <div>${s.flags.map(f => {
            const src = (f.source === "learned") ? "learned" : "live";
            const tag = src.toUpperCase();
            return `<span class="signal-flag ${src}"><span class="src-tag">${tag}</span>${f.label}</span>`;
          }).join("")}</div>
        </div>
      `).join("");
    }

    drawMap(d);
    setTimeout(() => r.scrollIntoView({ behavior: "smooth", block: "nearest" }), 50);
  }

  function drawMap(d) {
    const svg = document.getElementById("r-map");
    svg.innerHTML = "";
    const allCoords = [...d.baseline.coords, ...d.smart.coords];
    if (allCoords.length === 0) return;
    const lats = allCoords.map(c => c[0]);
    const lons = allCoords.map(c => c[1]);
    const minLat = Math.min(...lats), maxLat = Math.max(...lats);
    const minLon = Math.min(...lons), maxLon = Math.max(...lons);
    const W = 800, H = 400, PAD_X = 140, PAD_Y = 60;
    const project = ([lat, lon]) => {
      const x = (lon - minLon) / (maxLon - minLon || 1) * (W - 2*PAD_X) + PAD_X;
      const y = H - ((lat - minLat) / (maxLat - minLat || 1) * (H - 2*PAD_Y) + PAD_Y);
      return [x, y];
    };
    let grid = "";
    for (let i = 0; i <= 16; i++) { const x = (W/16)*i; grid += `<line x1="${x}" y1="0" x2="${x}" y2="${H}" stroke="#000" stroke-opacity="0.06" stroke-width="1"/>`; }
    for (let i = 0; i <= 8; i++) { const y = (H/8)*i; grid += `<line x1="0" y1="${y}" x2="${W}" y2="${y}" stroke="#000" stroke-opacity="0.06" stroke-width="1"/>`; }
    svg.innerHTML += grid;
    const basePts = d.baseline.coords.map(project);
    svg.innerHTML += `<path d="${basePts.map((p,i)=>(i?'L ':'M ')+p[0]+','+p[1]).join(' ')}" fill="none" stroke="#171e19" stroke-width="3" stroke-dasharray="8 6" opacity="0.5"/>`;
    const smartPts = d.smart.coords.map(project);
    svg.innerHTML += `<path d="${smartPts.map((p,i)=>(i?'L ':'M ')+p[0]+','+p[1]).join(' ')}" fill="none" stroke="#000" stroke-width="4"/>`;
    const seen = new Set();
    const allPts = [];
    d.baseline.coords.forEach((c, i) => { const k = c.join(","); if(!seen.has(k)){seen.add(k);allPts.push({coord:c,name:d.baseline.path_names[i]});}});
    d.smart.coords.forEach((c, i) => { const k = c.join(","); if(!seen.has(k)){seen.add(k);allPts.push({coord:c,name:d.smart.path_names[i]});}});
    const endName = d.smart.path_names[d.smart.path_names.length - 1];
    allPts.forEach((pt, idx) => {
      const [x, y] = project(pt.coord);
      const isStart = idx === 0;
      const isEnd = pt.name === endName;
      const r = (isStart || isEnd) ? 10 : 6;
      const fill = isStart ? "#ffe17c" : isEnd ? "#000" : "#fff";
      svg.innerHTML += `<circle cx="${x}" cy="${y}" r="${r}" fill="${fill}" stroke="#000" stroke-width="2.5"/>` +
                       `<text x="${x}" y="${y - 16}" text-anchor="middle" font-family="Cabinet Grotesk, sans-serif" font-weight="800" font-size="11" fill="#000">${pt.name.toUpperCase()}</text>`;
    });
    svg.innerHTML += `<g transform="translate(20, 20)">
      <rect x="-5" y="-5" width="170" height="55" fill="#fff" stroke="#000" stroke-width="2" rx="6"/>
      <line x1="6" y1="12" x2="36" y2="12" stroke="#171e19" stroke-width="3" stroke-dasharray="8 6" opacity="0.5"/>
      <text x="44" y="16" font-family="Satoshi, sans-serif" font-size="10" font-weight="700" fill="#000">BASELINE</text>
      <line x1="6" y1="34" x2="36" y2="34" stroke="#000" stroke-width="4"/>
      <text x="44" y="38" font-family="Satoshi, sans-serif" font-size="10" font-weight="700" fill="#000">SMART ROUTE</text>
    </g>`;
  }
</script>

</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(INDEX_HTML)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  DODOmaps — Smart Routing Demo")
    print("=" * 60)
    print(f"  Cities loaded: {list_cities()}")
    print(f"  Trips learned from: {sum(s.trips_observed for s in LEARNING_STORE.stats.values())}")
    print()
    print("  Open in browser:  http://localhost:5000")
    print()
    print("=" * 60)
    app.run(debug=False, port=5000, host='0.0.0.0')
