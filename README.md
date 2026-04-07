# TrialFair - Clinical Trial Eligibility Criteria Auditor

A web application for analyzing clinical trial eligibility criteria to identify enrollment barriers (gates) and assess representativeness.

## Features

- **Gate Detection**: Identifies 11 types of eligibility barriers (pregnancy exclusion, language requirements, technology access, etc.)
- **R-Index Calculation**: Measures trial representativeness compared to US population (when demographics provided)
- **AI Analysis**: Uses OpenAI to explain gates and suggest rewording
- **NCT Lookup**: Search trials by NCT ID from loaded AACT data

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

Edit `app.py` and replace:
```python
OPENAI_API_KEY = "YOUR_API_KEY_HERE"
```

Or set as environment variable:
```bash
export OPENAI_API_KEY="sk-..."
```

### 3. (Optional) Add AACT data for NCT lookup

Place your master CSV file at:
```
final_data/aact_master.csv
```

Required columns for full functionality:
- `nct_id`
- `eligibility_text` or `eligibility_criteria`
- `phase`
- `pct_female` (optional, for R-index)
- `age_*_prop` columns (optional, for R-index)
- `pct_white`, `pct_black`, etc. (optional, for R-index)

### 4. Run the server

```bash
python app.py
```

Or with uvicorn directly:
```bash
uvicorn app:app --reload --port 8000
```

### 5. Open in browser

Go to: http://localhost:8000

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve frontend |
| `/api/analyze` | POST | Analyze eligibility text |
| `/api/lookup/{nct_id}` | GET | Lookup trial by NCT ID |
| `/api/stats` | GET | Get dataset statistics |
| `/api/health` | GET | Health check |

### POST /api/analyze

Request body:
```json
{
  "eligibility_text": "INCLUSION: Adults 18-75...",
  "phase": "Phase 2",
  "condition": "Type 2 Diabetes",
  "demographics": {
    "sex_female": 0.45,
    "sex_male": 0.55,
    "age_18_44": 0.30,
    "age_45_64": 0.50,
    "age_65plus": 0.20
  }
}
```

Response:
```json
{
  "gates": [...],
  "gating_score": 5,
  "flags": {...},
  "rindex": {"r_sex": 0.91, "r_age": 0.72, "r_overall": 0.82},
  "llm_analysis": {...}
}
```

## Deployment

### Docker

```bash
docker build -t trialfair .
docker run -p 8000:8000 -e OPENAI_API_KEY="sk-..." trialfair
```

### Railway/Render/Heroku

1. Push to GitHub
2. Connect repository to hosting platform
3. Set `OPENAI_API_KEY` environment variable
4. Deploy

## Project Structure

```
trialfair_deploy/
├── app.py              # FastAPI backend
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container config
├── static/
│   └── index.html      # Frontend
└── final_data/
    ├── trialfair_exceptions.json  # Exception rules
    └── aact_master.csv            # Trial data (you provide)
```

## Without OpenAI API Key

The app works without an API key - you just won't get LLM explanations:
- Gate detection: ✅ Works
- R-index calculation: ✅ Works
- NCT lookup: ✅ Works
- AI explanations: ❌ Disabled

## License

Stanford affiliated research project.
