
# Chatbot Assistant — FAQ Retrieval API (FastAPI + TF-IDF)

This is the backend repository for a FastAPI-based question-answering system. It responds to user queries using an FAQ stored as a dictionary and applies TF-IDF with cosine similarity for matching. The model and artifacts are loaded at startup for fast responses, and it supports hot-reloading the FAQ via a protected endpoint.

## Features

🔎 TF-IDF (1–2 grams) + cosine similarity for FAQ retrieval

⚡ Persisted artifacts via joblib for quick startup

🔁 POST /reload to retrain & hot-reload without restarting (token-guarded)

🌐 CORS restricted to https://malakatech.com (configurable)

👋 Quick greeting detection (“hai/hi/halo/…”) with a canned reply


## Tools & Libraries

- **FastAPI** – Python API framework ([https://fastapi.tiangolo.com](https://fastapi.tiangolo.com)) 
- **scikit-learn** – Machine learning & NLP library ([https://scikit-learn.org](https://scikit-learn.org))
- **Joblib** – Model & artifact serialization ([https://joblib.readthedocs.io](https://joblib.readthedocs.io))
- **TfidfVectorizer** – NLP feature extraction for FAQ questions
- **Railway** – Backend deployment and hosting platform ([https://railway.app](https://railway.app))


## Running Tests

To run tests, run the following command

```bash
python -m venv .venv
venv\Scripts\activate
pip install -r requirements.txt

# Optional: enable /reload
export RELOAD_TOKEN="a_strong_secret_token"

uvicorn main:app --reload --host 0.0.0.0 --port 8000

```
open swagger UI: http://localhost:8000/docs

open reddoc UI: http://localhost:8000/redoc



## Simple Front-End Integration

```bash 
async function ask(q) {
  const url = new URL("http://localhost:8000/chatbot");
  url.searchParams.set("q", q);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

ask("What is MalakaTech?").then(console.log).catch(console.error);

```



## Credits
This project was designed and developed by [tegar Ramadhan](https://www.instagram.com/tegar_361) as the back-end for a chatbot. 
