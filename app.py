"""AI Data Analyzer — Flask web app.

    Local:   python app.py              ->  http://127.0.0.1:5000
    Online:  gunicorn app:app           (see render.yaml / README "Deploy to Render")
"""
from __future__ import annotations

import hmac
import json
import logging
import os
import re
import shutil
import time
import uuid
from datetime import timedelta
from functools import wraps
from pathlib import Path

from dotenv import load_dotenv
from flask import (Flask, abort, flash, redirect, render_template, request,
                   send_file, session, url_for)
from markupsafe import Markup, escape
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.utils import secure_filename

load_dotenv()

from analyzer import pipeline, sources  # noqa: E402  (load .env before importing)
from analyzer.ai_client import PROVIDERS, available_providers  # noqa: E402

BASE = Path(__file__).parent
UPLOADS = BASE / "uploads"
UPLOADS.mkdir(exist_ok=True)
ALLOWED = {".csv", ".xlsx", ".xlsm", ".xls"}
MAX_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))

ON_SERVER = bool(os.getenv("RENDER"))          # Render sets RENDER=true
APP_PASSWORD = os.getenv("APP_PASSWORD", "")
JOB_TTL_HOURS = 24                              # old reports are deleted after this

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY") or ("dev-only-change-me" if not ON_SERVER else None)
if not app.secret_key:
    raise RuntimeError("FLASK_SECRET_KEY must be set on the server")
if ON_SERVER and not APP_PASSWORD:
    raise RuntimeError("APP_PASSWORD must be set on the server, otherwise anyone can use your API keys")
app.config.update(
    MAX_CONTENT_LENGTH=MAX_MB * 1024 * 1024,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=ON_SERVER,            # HTTPS-only cookie online
    PERMANENT_SESSION_LIFETIME=timedelta(days=30),
)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)  # Render sits behind a proxy
logging.basicConfig(level=logging.INFO)


# --------------------------------------------------------------------------- #
# Password protection (only when APP_PASSWORD is set)
# --------------------------------------------------------------------------- #
_failed: dict[str, list[float]] = {}            # ip -> timestamps of failed logins
MAX_FAILS, FAIL_WINDOW = 5, 15 * 60


def _too_many_attempts(ip: str) -> bool:
    now = time.time()
    _failed[ip] = [t for t in _failed.get(ip, []) if now - t < FAIL_WINDOW]
    return len(_failed[ip]) >= MAX_FAILS


@app.before_request
def require_login():
    if not APP_PASSWORD or request.endpoint in ("login", "static", "health"):
        return None
    if session.get("auth") != _auth_token():
        return redirect(url_for("login", next=request.path))
    return None


def _auth_token() -> str:
    # Changes whenever the password changes, so old sessions are logged out.
    return hmac.new(app.secret_key.encode(), APP_PASSWORD.encode(), "sha256").hexdigest()


@app.route("/login", methods=["GET", "POST"])
def login():
    if not APP_PASSWORD:
        return redirect(url_for("index"))
    error = None
    if request.method == "POST":
        ip = request.remote_addr or "?"
        if _too_many_attempts(ip):
            error = "Too many wrong attempts. Try again in 15 minutes."
        elif hmac.compare_digest(request.form.get("password", "").encode(), APP_PASSWORD.encode()):
            _failed.pop(ip, None)
            session.clear()
            session.permanent = True
            session["auth"] = _auth_token()
            nxt = request.args.get("next", "")
            return redirect(nxt if nxt.startswith("/") and not nxt.startswith("//") else url_for("index"))
        else:
            _failed.setdefault(ip, []).append(time.time())
            error = "That password is incorrect."
    return render_template("login.html", error=error), (401 if error else 200)


@app.post("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.get("/healthz")
def health():
    return "ok"


@app.context_processor
def inject_globals():
    return {"auth_enabled": bool(APP_PASSWORD)}


def _cleanup_old_jobs() -> None:
    cutoff = time.time() - JOB_TTL_HOURS * 3600
    for d in UPLOADS.iterdir():
        if d.is_dir() and d.stat().st_mtime < cutoff:
            shutil.rmtree(d, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Template filters
# --------------------------------------------------------------------------- #
@app.template_filter("num")
def fmt_num(v):
    if v is None:
        return "–"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    if f.is_integer():
        return f"{int(f):,}"
    return f"{f:,.2f}"


@app.template_filter("compact")
def fmt_compact(v):
    """266030000 -> 266.03M, for headline figures that must fit a narrow column."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    for size, unit in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if abs(f) >= size:
            return f"{f / size:,.2f}".rstrip("0").rstrip(".") + unit
    return fmt_num(f)


def _ref_link(n: int, articles: list[dict]) -> str:
    a = next((x for x in articles if x.get("ref") == n), None)
    if not a:
        return f"[{n}]"
    return (f'<a class="ref" href="{escape(a["url"])}" target="_blank" rel="noopener" '
            f'title="{escape(a["title"])}">[{n}]</a>')


@app.template_filter("cite")
def cite(text, articles):
    """Turn [3] / [1, 4] in AI text into links to the source articles."""
    if not text:
        return ""
    safe = str(escape(text))
    return Markup(re.sub(r"\[(\d+(?:\s*,\s*\d+)*)\]",
                         lambda m: "".join(_ref_link(int(n), articles or []) for n in m.group(1).split(",")),
                         safe))


@app.template_filter("refs")
def refs(nums, articles):
    return Markup("".join(_ref_link(int(n), articles or []) for n in (nums or []) if str(n).isdigit()))


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/")
def index():
    return render_template("index.html", providers=available_providers(), max_mb=MAX_MB,
                           robot_email=sources.service_account_email(),
                           link_mb=int(os.getenv("MAX_LINK_MB", "200")))


@app.post("/analyze")
def analyze():
    link = request.form.get("link", "").strip()
    f = request.files.get("file")
    has_file = bool(f and f.filename)
    if not link and not has_file:
        flash("Choose a file or paste a Google Sheets / Drive link.")
        return redirect(url_for("index"))
    if has_file and not link and Path(f.filename).suffix.lower() not in ALLOWED:
        flash("Upload a .csv or .xlsx file.")
        return redirect(url_for("index"))

    _cleanup_old_jobs()
    job = uuid.uuid4().hex[:12]
    job_dir = UPLOADS / job
    job_dir.mkdir()

    provider = request.form.get("provider") or None
    if provider and provider not in PROVIDERS:
        provider = None
    context = {k: request.form.get(k, "").strip() for k in
               ("market", "industry", "brand", "competitors", "goal", "language")}
    run_args = dict(provider=provider, context=context,
                    auto_clean=bool(request.form.get("auto_clean")),
                    with_trends=bool(request.form.get("trends")),
                    cleaned_out=job_dir / "cleaned")

    try:
        if link:  # a link wins if both were given
            src = sources.fetch(link, job_dir)
            result = pipeline.run(src.path, raw_frames=src.frames if src.path is None else None,
                                  name=src.name, **run_args)
            result["source"] = {"type": "google", "via": src.via, "link": link}
        else:
            ext = Path(f.filename).suffix.lower()
            path = job_dir / (secure_filename(f.filename) or f"upload{ext}")
            if path.suffix.lower() != ext:          # secure_filename can drop non-latin names
                path = job_dir / f"upload{ext}"
            f.save(path)
            result = pipeline.run(path, name=f.filename, **run_args)
            result["source"] = {"type": "upload"}
    except sources.SourceError as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        flash(str(e))
        return redirect(url_for("index"))
    except Exception as e:  # show a friendly message, keep the details in the log
        app.logger.exception("analysis failed")
        shutil.rmtree(job_dir, ignore_errors=True)
        flash(f"This file couldn't be analysed: {e}")
        return redirect(url_for("index"))

    # the original download isn't needed any more; free the disk
    for p in job_dir.glob("source.*"):
        p.unlink(missing_ok=True)
    (job_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return redirect(url_for("report", job=job))


def _job_dir(job: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{12}", job):
        abort(404)
    d = UPLOADS / job
    if not d.exists():
        abort(404)
    return d


@app.get("/report/<job>")
def report(job):
    r = json.loads((_job_dir(job) / "result.json").read_text(encoding="utf-8"))
    label = PROVIDERS.get(r.get("provider") or "", {}).get("label")
    return render_template("report.html", r=r, job=job, provider_label=label)


@app.get("/download/<job>")
def download(job):
    d = _job_dir(job)
    r = json.loads((d / "result.json").read_text(encoding="utf-8"))
    stored = r.get("cleaned_file") or "cleaned.xlsx"
    if not re.fullmatch(r"cleaned\.(xlsx|zip)", stored) or not (d / stored).exists():
        abort(404)
    name = f"{Path(r['file']).stem}_cleaned{Path(stored).suffix}"
    return send_file(d / stored, as_attachment=True, download_name=name)


@app.errorhandler(413)
def too_big(_):
    flash(f"This file is larger than {MAX_MB} MB. Split it or remove unused sheets, then upload again.")
    return redirect(url_for("index"))


if __name__ == "__main__":
    # Local only. Online, gunicorn starts the app and debug mode is never used:
    # the debugger lets anyone who triggers an error run code on the server.
    app.run(debug=os.getenv("FLASK_DEBUG") == "1" and not ON_SERVER, port=int(os.getenv("PORT", "5000")))
