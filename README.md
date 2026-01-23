# TrailsenseAI
A strava companion AI for trailsense and text parsing
# Strava-Inspired TrailSense AI

A small FastAPI service that:
- Loads an OSMnx walking graph for Berlin at startup
- Uses a lightweight RandomForest model to compute edge weights ("athlete_score")
- Exposes:
  - POST /v1/route  (direct coordinate routing)
  - POST /v1/search (LLM prompt -> structured intent -> route)

I guess osmnx creates a DiGraph so i converted the data structure to exctract the length
<img width="1506" height="978" src="Screenshot from 2026-01-23 11-27-23.png" />

demo.py
img
## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export OPENAI_API_KEY="..."
uvicorn main:app --reload
