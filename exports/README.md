# AlertAhead dispatch exports

Scenario `99b551` · issue week 2025-06-30 · +3 weeks (target 2025-07-21) · stock 60% · crews 8 · engine: backend

Generated offline by `python make_exports.py` from the templates in `dengue/notify.py` (no LLM, no network).

| Folder / file | What |
|---|---|
| `sms/` | One-segment English SMS (GSM-7, ≤160 chars) and Bangla SMS (≤3 parts) per ward |
| `whatsapp/` | Bangla + English WhatsApp notice per ward |
| `briefs/` | Printable HTML commissioner / health-centre briefs (open → Ctrl+P → Save as PDF) |
| `city_summary/` | Bed headroom, routing and alerts per city |
| `escalation/` | Civil-surgeon escalation, only for cities with CODE BLACK or an exhausted bed pool |
| `allocation.csv`, `routing.csv` | The numbers behind every notice |
