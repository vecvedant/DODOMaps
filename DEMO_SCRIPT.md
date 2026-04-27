# DEMO SCRIPT v3 — Multi-City Smart Routing

**Total time: ~5 min spoken + ~2 min live demo = 7 min**

---

## What's NEW in v3

- **Multi-city support**: Pune (validated), Mumbai, Bangalore
- **City-as-config**: cities defined in `cities.py`, no code changes per city
- **Per-city event calendars**: Mumbai monsoon ≠ Pune monsoon ≠ Bangalore bandhs
- **Namespaced learning**: each city's fleet learns its own patterns

---

## 1. THE OPENING HOOK (20 sec)

> "Logistics fleets across India lose hours every month to disruptions
> they could have predicted: Mumbai monsoon flooding, Bangalore ORR tech
> traffic, Pune Ganpati closures, IPL match day jams. Today's tools react
> after the truck is stuck. Our system anticipates — combining real-time
> data, city-specific event calendars, and learning from every trip the
> fleet completes."

---

## 2. THE PROBLEM (15 sec)

> "Three signals matter for predicting delay: real-time conditions, known
> upcoming events, and historical patterns specific to each fleet's
> routes. No single system today combines all three across multiple
> Indian cities. We do."

---

## 3. THE ARCHITECTURE (30 sec)

```
   Real-time          Calendar (per-city)        Learning
   (traffic, weather) (Pune: Ganpati, IPL...     (per-edge,
                       Mumbai: monsoon...         per-city
                       Bangalore: bandh...)       trip logs)
            \              |                       /
             \             |                      /
              v            v                     v
            Integrated Conditions Provider
                         |
                         v
              Smart Edge Weight formula
                         |
                         v
                A* routing on city graph
                         |
                         v
                Best route + ETA + Why
```

> "Each city is a config bundle — graph, calendar, learned history. The
> routing engine is city-agnostic. Adding Delhi or Kolkata is a data
> task, not an engineering one."

---

## 4. THE LIVE DEMO (2 min — the showpiece)

```bash
python multicity_router.py
```

**Step 4a — Show the warm-up output:**

> "First, we warm up the learning loop with 90 days of simulated trip
> history per city. **1,350 trips total across three cities.** Each
> city's deployment learns its own patterns — Pune's chronic delays
> don't pollute Mumbai's data."

**Step 4b — Walk through the Pune validation table.**

> "Pune is our headline city — 20 scenarios validated. Average **11.2%**
> time saved, best case **28.2%** on storm-plus-accident. Zero savings
> on clear days because the system correctly avoids unnecessary reroutes."

**Step 4c — Show Mumbai and Bangalore numbers.**

> "Same architecture, different cities. Mumbai: **11.1% average, 30.7%
> best case** on an Eastern Express accident. Bangalore: **14.9% average,
> 22.7% best**. Bangalore actually saves *more* on average because ORR
> tech traffic is so chronic — every reroute compounds."

**Step 4d — Point at the multi-city comparison table.**

> "This is the proof of generalization. Same system, three cities,
> consistent double-digit savings. Different graphs, different events,
> different warehouses, different deliveries — same engine."

**Step 4e — Walk through the three explainability demos.**

> **Pune (the calendar showcase):**
>
> "May 2nd — IPL match day at MCA Stadium. Calendar layer auto-detects.
> Baseline takes 76 minutes; smart route delivers in 57. **19 minutes
> saved automatically because we know the cricket schedule.**"

> **Mumbai (the real-time showcase):**
>
> "Accident on the Eastern Express — real-time event detection. Baseline
> would take 97 minutes; we reroute via Bandra-Kurla and deliver in 67.
> **30 minutes saved**, our biggest demo win, by detecting the disruption
> and finding an alternative."

> **Bangalore (the chronic-delay showcase):**
>
> "Standard tech park evening rush — no special events, just everyday
> ORR congestion. Baseline routes straight through Marathahalli; smart
> route diverts via Silk Board. **21 minutes saved** on a regular
> Tuesday, just from the learning loop knowing ORR is chronically
> slow."

---

## 5. THE THREE DIFFERENTIATORS (30 sec)

> **One — calendar awareness across India.** Pune Ganpati, Mumbai
> Marathon, Bangalore Bandh, monsoon timings — pre-loaded per city.
> Google Maps has no notion of "Feb 14 is Karnataka Bandh."

> **Two — per-fleet, per-city learning.** Each customer deployment
> learns the routes its trucks actually take. Generic tools can't learn
> patterns specific to one company's fleet.

> **Three — fleet dispatch optimization, not driver navigation.** Our
> conditions provider is decoupled from routing — designed to plug into
> OR-Tools VRP for multi-vehicle, multi-stop optimization in Phase 2."

---

## 6. SCALING STORY (15 sec)

> "Three cities live, six on roadmap (Delhi, Kolkata, Vizag plus the
> three you've seen). Adding a city takes adding 10-20 nodes, 5-10
> calendar events, and a chronic-delay map. **It's a config file.**
> Same engine, same validation methodology, same demo flow."

---

## 7. CLOSING (10 sec)

> "Three integrated layers — real-time, calendar, learning — driving
> graph optimization across multiple Indian cities. Validated, working,
> and ready for fleet integration. Thank you."

---

# THE SHOWPIECE NUMBERS (memorize these in order)

| Metric | Number |
|--------|--------|
| Pune avg / best | **11.2% / 28.2%** |
| Mumbai avg / best | **11.1% / 30.7%** |
| Bangalore avg / best | **14.9% / 22.7%** |
| IPL day savings (Pune) | **19 min** (calendar) |
| Eastern Express accident (Mumbai) | **30 min** (real-time) |
| Tech park rush (Bangalore) | **21 min** (learning) |
| Total trips learned from | **1,350** across 3 cities |

# THE THREE SHOWPIECE LINES (memorize)

> *"Same architecture, three cities, consistent double-digit time
> savings. That's the generalization."*

> *"19 minutes saved on IPL day because we know the cricket schedule.
> 30 minutes saved on a Mumbai accident because we react in real time.
> 21 minutes saved in Bangalore because we learned ORR is chronically
> slow. Three different mechanisms, one system."*

> *"Adding Delhi is a config file. The architecture is generalized."*

---

# COMMAND REFERENCE FOR LIVE DEMO

```bash
# Run all three cities (default demo)
python multicity_router.py

# Run just one city (if you need to drill in)
python multicity_router.py pune
python multicity_router.py mumbai
python multicity_router.py bangalore

# Show the cities config (proves generalization)
python cities.py

# Show learning loop in isolation
python learning_loop.py
```

If demo machine breaks, fall back to `validation_output.txt` for screenshots.
