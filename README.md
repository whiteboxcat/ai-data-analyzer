# AI Data Analyzer

Upload a CSV / Excel file or paste a Google Sheets / Drive link and get, **sheet by sheet**:

1. **Profile**: what each column really is (date, metric, ID, category, text), with stats
2. **Data cleaning**: issues found, safe fixes applied automatically, judgement calls flagged for review, plus a cleaned Excel to download
3. **Charts**: the AI picks the charts, Python checks the plan and draws them
4. **Insights and recommendations**: written by GPT, Claude or Gemini from the real aggregated numbers
5. **Industry trends**: the industry, market and competitors are detected, real recent headlines are pulled from Google News, and the AI summarises them with citations

## How it works

```
Upload CSV/XLSX
   │  loader.py      read every sheet, find the real header row (skips title rows)
   ▼
profiler.py         column roles: date / metric / id / category / text / boolean
   ▼
cleaner.py          detect issues → apply SAFE fixes → log everything
   ▼
planner.py   ──AI──▶ chart plan + industry detection (JSON)
   │                 every chart validated by charts.validate_spec()
   ▼
charts.py           pandas aggregates (SUM GMV by month, top 10 + Other, ...)
   ▼
insights.py  ──AI──▶ executive summary, insights, recommendations (from real numbers)
   ▼
industry.py         Google News RSS (industry + competitors) ──AI──▶ trends with [n] citations
   ▼
Flask report: Summary · Data cleaning · one tab per sheet · Industry trends
```

**Design rule:** Python does data processing, validation, aggregation and chart execution. The AI does understanding, planning, insights and recommendations. The AI only sees the profile and aggregated results, never the full raw data. That keeps it cheap, private and grounded.

Without any API key the app still works. Profiling, cleaning and default charts run, and you get basic computed observations instead of written insights.

## Setup (WSL / Ubuntu)

```bash
cd ~/ai-lab/ai-data-analyzer
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then add your key(s)
python tests/make_sample.py # creates a messy sample workbook
python tests/test_pipeline.py   # offline test, no keys needed
python tests/test_sources.py    # Google links (simulated) + a 600k-row file
python app.py               # http://127.0.0.1:5000  (add FLASK_DEBUG=1 to .env for auto-reload)
```

Try `tests/sample_messy.xlsx` first. It has title rows, mixed date formats, "Rp 1.250.000" text numbers, duplicate rows, inconsistent spelling, a YYYYMMDD date sheet and a blank sheet.

## Deploy to Render (free)

The app is password-protected online: it refuses to start on Render without `APP_PASSWORD`, so strangers can't spend your API credits.

1. **Push to GitHub.** Run `git status` first and make sure `.env` is *not* listed (`.gitignore` excludes it).
   ```bash
   git add . && git commit -m "Add login and Render deploy" && git push
   ```
2. **Create the service.** On [render.com](https://render.com), sign in with GitHub, then **New → Blueprint** and pick this repo. Render reads `render.yaml`.
3. **Fill in the secrets** when Render asks: `APP_PASSWORD` (choose a long one) and the AI keys you use. Leave unused keys empty. `FLASK_SECRET_KEY` is generated for you.
4. **Deploy.** The first build takes a few minutes. Your app is then at `https://ai-data-analyzer-xxxx.onrender.com`.
5. Every `git push` redeploys automatically.

**Good to know about the free plan**
- It sleeps after 15 minutes without visits; the first visit after that takes about a minute.
- Uploaded files and reports are wiped when it sleeps or redeploys. Download the cleaned Excel when you need it.
- 512 MB memory: fine for normal spreadsheets; very large files (hundreds of thousands of rows) may fail.

**Keep your keys safe**
- Keys live only in Render's Environment settings, never in GitHub.
- Set a monthly spending limit in your OpenAI / Anthropic / Google billing settings.
- Use a separate key for this app, so you can delete and replace it without breaking anything else.
- To change the password, edit `APP_PASSWORD` in Render; everyone signed in is logged out.

## Analyse a Google Sheet or Drive file

On the upload page, switch to **Google Sheets or Drive link** and paste the link. It works with Google Sheets, and with `.xlsx` / `.csv` files stored in Drive. Every tab of a Google Sheet is analysed.

**Quick way (public link):** in Google click **Share → General access → Anyone with the link (Viewer)**. No setup, but anyone who gets the link can open the file.

**Private way (recommended for business data):** give the app its own Google "robot" account (a *service account*) and share files only with it. Free, one-time setup, about 10 minutes:

1. Go to [console.cloud.google.com](https://console.cloud.google.com), create a project (e.g. `ai-data-analyzer`).
2. **APIs & Services → Library**: enable **Google Sheets API** and **Google Drive API**.
3. **IAM & Admin → Service Accounts → Create service account**. Any name; skip the optional role steps.
4. Open the new account → **Keys → Add key → Create new key → JSON**. A `.json` file downloads. Treat it like a password.
5. In Render → your service → **Environment**, add `GOOGLE_SERVICE_ACCOUNT_JSON` and paste the *entire* content of that file. Locally, put it in `.env` on one line, or set `GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json`.
6. The upload page now shows the robot's email (like `ai-analyzer@your-project.iam.gserviceaccount.com`). In Google Sheets/Drive, **Share** the file with that email as **Viewer**.

The app tries private access first, then the public link, and tells you which sharing step is missing if neither works. Google Sheets are read through the API as values, so numbers formatted as `Rp 1.250.000` arrive as real numbers.

## Big files

- The AI never receives your rows, only column summaries, a few sample rows and chart totals (a few KB), so file size doesn't affect AI cost.
- CSVs are read in two passes (header detection on a sample, then pandas' fast parser), so numeric columns use 8 bytes per cell instead of a text string.
- `.xlsx` files are read with the faster, lighter `python-calamine` engine when installed.
- On Render's free plan (512 MB) the app accepts up to about 120 MB of loaded data (`MAX_DATA_MB`) and explains what to do if a file is bigger. In testing, a 600,000-row CSV used under 250 MB peak and took about 8 seconds.
- When cleaned data is too big for Excel (over about 1 million cells), the download is a `.zip` of CSVs instead.
- Linked files are capped at `MAX_LINK_MB` (200 MB by default).

## Files

| File | Job |
|---|---|
| `app.py` | Flask routes: login, upload, report, download cleaned Excel |
| `render.yaml` | Render deploy settings (start command, secrets to ask for) |
| `analyzer/loader.py` | CSV/XLSX → `{sheet: DataFrame}`, header detection, delimiter sniffing, low-memory reading |
| `analyzer/sources.py` | Google Sheets / Drive links (private via service account, or public) |
| `analyzer/profiler.py` | Column roles and statistics (EN and Bahasa Indonesia name hints) |
| `analyzer/cleaner.py` | Data quality checks and safe auto-fixes |
| `analyzer/utils.py` | Smart date parsing (dd/mm vs ISO, 20240131) and money-text parsing ("Rp 1.250.000", "(300)", "15%") |
| `analyzer/ai_client.py` | One `ask_json()` for OpenAI / Claude / Gemini |
| `analyzer/planner.py` | AI chart plan + industry detection, and the no-AI fallback |
| `analyzer/charts.py` | Validates chart specs, aggregates data, KPIs |
| `analyzer/insights.py` | AI insights and recommendations (fallback: computed notes) |
| `analyzer/industry.py` | News fetching + AI trend/competitor summary |
| `analyzer/pipeline.py` | Runs everything in order |
| `static/charts.js` | Small dependency-free SVG chart renderer (works offline, light and dark mode) |

## What gets auto-fixed vs flagged

| Auto-fixed (safe) | Flagged for review |
|---|---|
| Exact duplicate rows | Negative qty / price |
| Extra spaces, "dress"/"Dress " spelling variants | Extreme outliers (likely typos) |
| Numbers stored as text ("Rp 1.250.000") | Missing numbers (blank ≠ 0 automatically) |
| Dates stored as text / 20240131 integers | Missing dates or IDs |
| Missing category → "Unknown" | Repeated IDs |
| Fully empty columns | Constant columns |

You can switch auto-fix off on the upload form.

## Ideas for next steps

- Let the user tick which fixes to apply and re-run
- Chat with the data: ask a question, the AI writes a pandas query, Python runs it in a sandbox
- Cross-sheet joins (e.g. Sales.SKU → Products.SKU) suggested by the AI
- Export the report to PDF
- Cache results by file hash so re-uploads are instant
