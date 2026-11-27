"""Write sample dispatch exports to exports/ (R4 + R5), fully offline.

Usage:  python make_exports.py [--date 2025-06-30] [--horizon 3] [--stock 60] [--crews 8] [--no-pdf]

Uses the same engine as the app (dengue.ui_command.get_engine -> backend, or the labelled preview)
and the same templates (dengue/notify.py), so the files always match what the app shows.
Deterministic: running it twice gives byte-identical files.
"""
import argparse
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd

os.environ.setdefault("ALERTAHEAD_OFFLINE", "1")   # exports never touch the network

from dengue import notify, outbox  # noqa: E402
from dengue.ui_command import adapt_result, get_engine, scenario_id  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "exports"
FONT = ROOT / "assets" / "fonts" / "bn-subset.woff2"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--date", default="2025-06-30")
    ap.add_argument("--horizon", type=int, default=3, choices=(2, 3, 4))
    ap.add_argument("--stock", type=int, default=60)
    ap.add_argument("--crews", type=int, default=8)
    ap.add_argument("--efficacy", type=float, default=0.30)
    ap.add_argument("--pdf", action="store_true", help="(default) print sample briefs to PDF with a local headless Edge/Chrome")
    ap.add_argument("--no-pdf", action="store_true", help="skip the sample PDFs")
    args = ap.parse_args()

    Params, run, kind = get_engine()
    p = Params(issue_date=args.date if kind == "backend" else pd.Timestamp(args.date), horizon=args.horizon, city=None,
               stock_pct=args.stock, buffer_pct=0, overrides=None, total_crews=args.crews, efficacy=args.efficacy,
               reserve_beds=0, offline=True)
    wards, transfers, cities, _alerts, source = adapt_result(run(p))
    sid = scenario_id(issue=args.date, h=args.horizon, city=None, stock=args.stock, buffer=0, crews=args.crews,
                      eff=round(args.efficacy * 100), reserve=0, ov={}, cap=100)
    ctx = notify.NoticeContext(target=pd.Timestamp(args.date) + pd.Timedelta(weeks=args.horizon), horizon=args.horizon,
                               stock_pct=args.stock, total_crews=args.crews, scenario=sid, source=source)

    font_url = None
    if OUT.exists():
        shutil.rmtree(OUT)
    if FONT.exists():
        (OUT / "fonts").mkdir(parents=True, exist_ok=True)
        shutil.copy2(FONT, OUT / "fonts" / FONT.name)
        font_url = "../fonts/" + FONT.name
    files = notify.export_files(wards, cities, transfers, ctx, font_rel_url=font_url)
    files.update(outbox.outbox_files(outbox.build_outbox(wards, cities, transfers, ctx)))
    files["README.md"] = (
        "# AlertAhead dispatch exports\n\n"
        f"Scenario `{sid}` · issue week {args.date} · +{args.horizon} weeks (target {ctx.target:%Y-%m-%d}) · "
        f"stock {args.stock}% · crews {args.crews} · engine: {kind}\n\n"
        "Generated offline by `python make_exports.py` from the templates in `dengue/notify.py` (no LLM, no network).\n\n"
        "| Folder / file | What |\n|---|---|\n"
        "| `sms/` | One-segment English SMS (GSM-7, ≤160 chars) and Bangla SMS (≤3 parts) per ward |\n"
        "| `whatsapp/` | Bangla + English WhatsApp notice per ward |\n"
        "| `briefs/` | Printable HTML commissioner / health-centre briefs (open → Ctrl+P → Save as PDF) |\n"
        "| `city_summary/` | Bed headroom, routing and alerts per city |\n"
        "| `escalation/` | Civil-surgeon escalation, only for cities with CODE BLACK or an exhausted bed pool |\n"
        "| `allocation.csv`, `routing.csv` | The numbers behind every notice |\n"
        "| `outbox/` | Alert routing: who gets which message (`outbox.csv`, priority + recipient + SMS text) and one file per "
        "recipient with a RED, CODE BLACK or escalation alert (`by_recipient/`) |\n"
    )
    for rel, text in files.items():
        path = OUT / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
    print(f"Wrote {len(files)} files to {OUT} (engine: {kind}, scenario {sid})")
    if not args.no_pdf:   # exports/ is rebuilt from scratch, so the sample PDFs are re-made every run
        print_sample_pdfs(wards)


BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe", "/usr/bin/chromium", "/usr/bin/google-chrome"]


def print_sample_pdfs(wards: pd.DataFrame) -> None:
    """Print a CODE BLACK, a red-alert and a no-shortfall ward brief to PDF (offline, local browser)."""
    browser = next((b for b in BROWSERS if Path(b).exists()), None)
    if browser is None:
        print("No local Edge/Chrome found - skipped PDFs (open a brief and press Ctrl+P instead).")
        return
    w = wards.sort_values("ward_id")
    picks = [w[w["code_black"]], w[w["red"] & ~w["code_black"]], w[~w["deficit"]]]
    out_dir = OUT / "briefs_pdf"
    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in picks:
        if sub.empty:
            continue
        wid = sub.iloc[0]["ward_id"]
        src = (OUT / "briefs" / f"{wid}_commissioner.html").resolve()
        dst = (out_dir / f"{wid}_commissioner.pdf").resolve()
        subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={dst}", src.as_uri()],
                       check=False, capture_output=True, timeout=120)
        print(("PDF  " if dst.exists() else "PDF FAILED ") + str(dst.relative_to(ROOT)))


if __name__ == "__main__":
    main()
