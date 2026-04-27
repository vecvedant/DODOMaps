# JUDGE Q&A CHEAT SHEET v3 — Multi-City

The questions judges WILL ask, with answers that win.
Memorize the **bold** parts.

---

## Q1. "How is this different from Google Maps?"

> **"Three things Google can't do across Indian cities. One — we know
> each city's calendar: Pune Ganpati, Mumbai Marathon, Bangalore Bandhs.
> Two — we learn from each fleet's specific trips. Three — we optimize
> for fleet dispatch, not single-driver navigation."**

The IPL day demo is your concrete proof. Use it.

---

## Q2. "What's your accuracy?"

> **"Across three cities: Pune 11.2% average, Mumbai 11.1%, Bangalore
> 14.9%. Best cases hit 28-31%. Same architecture, consistent results
> — the system generalizes."**

---

## Q3. "Why only three cities? What about Delhi, Kolkata, Vizag?"

> **"Three cities are validated end-to-end. The architecture supports
> any city — adding Delhi is a config file with nodes, edges, and
> calendar events. We deliberately scoped to validated cities for the
> demo. The roadmap config exists; integration is data work."**

This answer is gold. It turns "limitation" into "discipline."

---

## Q4. "Show me a city actually working."

Run `python multicity_router.py`. Three things to point at:
- Multi-city comparison table (proves consistent results)
- Pune IPL day demo (calendar layer)
- Mumbai Eastern Express demo (real-time layer)
- Bangalore tech rush demo (learning layer)

> **"Three different cities. Three different disruption types. One
> system handles all of them."**

---

## Q5. "Where does the calendar data come from?"

> **"For the prototype: hardcoded per-city events for 2026 — Ganpati,
> Diwali, Marathon, IPL match days, monsoon, bandhs. In production:
> India public-holiday APIs, municipal corporation event notices, and
> scraped news. The data layer is decoupled from the routing logic."**

---

## Q6. "Where is the actual machine learning?"

> **"The learning loop is live and namespaced per city. 1,350
> simulated trips across three cities. The system correctly identified
> chronic delays — Pune's B-D corridor 25% slow, Mumbai's M2-M3 30%
> slow, Bangalore's ORR 35% slow — with no manual tagging. Phase 2
> replaces EWMA with XGBoost once real trip logs accumulate."**

---

## Q7. "What about RAG / GenAI / LLMs?"

> **"RAG is for text retrieval. Our problem is graph optimization with
> numerical edge weights — different problem class. Adding an LLM would
> be a buzzword, not an improvement."**

This makes you sound smarter than the buzzword crowd.

---

## Q8. "Why these specific weight coefficients?"

> **"Anchored to public domain studies — FHWA shows heavy rain reduces
> highway speed by ~30%; TomTom Traffic Index shows peak congestion adds
> up to 80%. The same coefficients work across all three cities — we
> verified this by validating each city independently."**

---

## Q9. "How does this scale to a 50,000-node city graph?"

> **"Prototype handles ~500 nodes with sub-second A\*. For city-scale
> graphs, production uses contraction hierarchies — well-established in
> the routing literature. The bottleneck isn't the algorithm; it's the
> data pipeline, which we've designed for streaming."**

---

## Q10. "Mumbai Marathon doesn't reroute in your demo. Is that a bug?"

> **"No — it's correct behavior. Marathon affects routes the baseline
> already avoids, so no reroute is needed. The system reroutes only
> when math justifies it. Compare to Eastern Express accident: same
> Mumbai system saves 30 minutes when the disruption hits the actual
> baseline path."**

This is a subtle question and answering it well shows depth.

---

## Q11. "Multiple vehicles? Multiple stops?"

> **"Phase 2 — plugs into Google's OR-Tools for full Vehicle Routing
> Problem with capacity and time windows. The conditions provider is
> already decoupled from routing — clean separation, OR-Tools stacks on
> top as the cost matrix."**

---

## Q12. "What if APIs go down or we're offline?"

> **"Three layers of graceful degradation. Real-time data drops:
> calendar layer still works. Both drop: learning store still gives
> chronic-pattern routing. Even fully offline, the system beats baseline
> on chronic edges because historical delay factors are stored
> locally."**

---

## Q13. "Why simulated, not real data?"

> **"Simulated, but principled. Disruption distributions anchor to
> ground-truth — IMD weather reports, TomTom Traffic Index, observed
> PMC event durations. Real-world validation is the obvious next step;
> simulation is how we validated the algorithm before paying for
> commercial APIs across multiple cities."**

---

## Q14. "Different cities have different traffic cultures. Does one model fit all?"

> **"That's exactly why we namespace the learning loop per city.
> Pune-specific delays don't bleed into Mumbai's model. The weight
> formula is universal — base + traffic + weather + event +
> historical — but the *learned values* differ city to city. ORR
> traffic in Bangalore looks different from Eastern Express in
> Mumbai, and our model captures both."**

---

## Q15. "What's the business model?"

> **"SaaS for mid-size 3PL operators running 20-200 trucks across
> Indian metros. Pricing per-vehicle-per-month. The data flywheel is
> the moat — every trip a customer's fleet runs in any of our cities
> makes the system smarter for that customer. Google can't replicate
> that because they don't see the fleet's trips."**

---

## Q16. "How does Bangalore have higher savings than Pune?"

> **"Because Bangalore's ORR has chronic congestion the learning loop
> picked up — 35% slow on the Whitefield-Marathahalli stretch. Every
> reroute saves more there. Pune's congestion is more episodic — heavier
> on event days like IPL, lighter on regular days. Same algorithm,
> different city characteristics."**

This shows you understand your own results.

---

# THE THREE NEVER-SAY-I-DON'T-KNOW FALLBACKS

**Technical you don't know:**
> "Good architectural question. Our focus was multi-city integration —
> that's part of our Phase 2 engineering."

**Business:**
> "Indian logistics is roughly $200B with single-digit margins.
> Multi-city is the natural unit for 3PL operators — that's our
> beachhead."

**'But X already exists':**
> "X exists for [their case]. Our gap is [calendar across multiple
> Indian cities / fleet learning / dispatch optimization]."

---

# THE GOLDEN RULE

**Confidence > completeness.** A confident "we have three cities
validated, three more on the roadmap" beats a nervous "we don't have
Delhi yet."

You're selling: **principled architecture, validated execution, and a
credible scaling story across India.**

Judges fund people who know what they built — and what they didn't.
