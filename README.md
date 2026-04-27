# OBSIDIAN — Predictive Logistics Intelligence

Integrated Smart Routing System with web UI. Pune, Mumbai, and Bangalore.

## Setup

1. Clone this repo.
2. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
3. Get a free Gemini API key: https://aistudio.google.com/app/apikey
4. Paste the key into `.env`, replacing `paste_your_key_here`.
5. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
6. Run:
   ```bash
   python app.py
   ```
7. Open **http://localhost:5000**.

The Gemini integration generates natural-language explanations in the "Why this route" panel (manual mode) and the executive summary above the fleet comparison. If `GEMINI_API_KEY` is not set, the system falls back to template narratives — every other feature still works.

## TL;DR — How to demo

```bash
pip install -r requirements.txt
python app.py
```

Then open **http://localhost:5000** in your browser.

That's it. Pick a city, pick a scenario, click Compute Route. The system runs the actual A\* engine, shows you the baseline route vs the smart route on a map, and tells you how many minutes were saved and why.

## What's running under the hood

The web UI is just a face on top of the same backend that runs the validation:
- City graphs and event calendars from `cities.py`
- A\* routing with smart edge weights from `multicity_router.py`
- Per-edge delay learning from `learning_loop.py` (warmed up with 1,350 simulated trips on startup)
- Flask exposes 4 API endpoints (`/api/cities`, `/api/scenarios/<city>`, `/api/route`, `/api/validation`)

When you click "Compute Route" in the browser, it makes a real `POST /api/route` call that runs the routing engine for that scenario and returns both routes with their actual times.

## Files

| File | Purpose |
|------|---------|
| **`app.py`** | **The web app — run this.** Flask backend + Modern Obsidian UI |
| `multicity_router.py` | Multi-city routing engine (CLI demo too) |
| `cities.py` | Graphs, edges, event calendars per city |
| `learning_loop.py` | Per-edge EWMA delay learning, JSON-persisted |
| `event_calendar.py` | Original Pune-only calendar (kept for reference) |
| `integrated_router.py` | Single-city v2 (kept as fallback) |
| `smart_router.py` | v1 standalone (reference) |
| `validation_output.txt` | Captured output of last CLI run |
| `DEMO_SCRIPT.md` | What to say during the demo, in order |
| `QA_CHEATSHEET.md` | 16 judge questions + prepared answers |

## CLI fallback (if web app dies)

If anything breaks on demo day, drop to the CLI:

```bash
python multicity_router.py            # full multi-city validation
python multicity_router.py pune       # one city
```

## What the web UI shows

1. **Hero section** — headline numbers (12.4% avg, 30.7% best, 1,350 trips, 3/6 cities)
2. **System explainer** — three signal sources (real-time, calendar, learning)
3. **City cards** — Pune / Mumbai / Bangalore with their per-city metrics
4. **Live demo** — pick a city + scenario, click compute, see live routing
5. **Metrics bento** — full validation receipts
6. **Differentiators** — what Google can't do
7. **Final CTA**

The Live Demo is the centerpiece — it's the only place a judge sees the system actually compute a route. Spend 60-90 seconds there.

## Headline numbers (memorize)

| City | Avg saved | Best case | Demo highlight |
|------|-----------|-----------|----------------|
| Pune | **11.2%** | 28.2% | IPL day · 19 min saved |
| Mumbai | **11.1%** | 30.7% | Eastern Express · 30 min saved |
| Bangalore | **14.9%** | 22.7% | Tech park rush · 21 min saved |

## The pitch (memorize one line)

> *"19 minutes saved on Pune's IPL day because we know the cricket schedule. 30 minutes saved on a Mumbai accident because we react in real time. 21 minutes saved in Bangalore because we learned ORR is chronically slow. Three different mechanisms, one system."*

## What's honestly NOT done

These are real gaps. The Q&A cheat sheet has prepared answers:

- **OSM integration** — graphs are hand-built (5-8 nodes per city). Production uses real OpenStreetMap data with thousands of nodes.
- **Three more cities on roadmap** — Delhi, Kolkata, Vizag. Adding a city = config file in `cities.py`.
- **Multi-vehicle / VRP** — single-route only. Phase 2 stacks OR-Tools.
- **Trained ML model** — currently EWMA + rule-based weights. Phase 2 = XGBoost on real trip logs.

## Demo day checklist

- [ ] `python app.py` runs cleanly on demo machine
- [ ] Browser opens http://localhost:5000 without issues
- [ ] Run through the demo script 3 times out loud
- [ ] Memorize the three headline numbers (11.2 / 11.1 / 14.9)
- [ ] Memorize the three demo highlights (19 / 30 / 21 minutes)
- [ ] Pre-click through Pune IPL → Mumbai accident → Bangalore tech rush so you know which scenarios to pick
- [ ] Have CLI fallback ready: `python multicity_router.py`

Good luck. The system is real. Now go own the room.
