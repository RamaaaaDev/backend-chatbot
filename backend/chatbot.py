# main.py
from fastapi import FastAPI, Query, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from joblib import dump, load
from pathlib import Path
import json, os, re

app = FastAPI()

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # ganti dengan domain frontend kamu di produksi
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Path & Config ---
SRC_FAQ_PATH = Path("faq_malakatech.json")            # sumber FAQ (repo)
ARTIFACT_DIR = Path(os.getenv("MODEL_DIR", "artifacts"))
ARTIFACT_DIR.mkdir(exist_ok=True)
VEC_PATH = ARTIFACT_DIR / "tfidf_vectorizer.joblib"
X_PATH   = ARTIFACT_DIR / "tfidf_matrix.joblib"
FAQ_CACHE_PATH = ARTIFACT_DIR / "faq_cache.json"
RELOAD_TOKEN = os.getenv("RELOAD_TOKEN")  # set di Railway env var kalau pakai /reload

# --- Helpers ---
def text_clean(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)  # hapus tanda baca
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_faq_source():
    try:
        data = json.loads(SRC_FAQ_PATH.read_text(encoding="utf-8"))
        # fallback minimal bila kosong
        return data if data else [{"question": "default", "answer": "default"}]
    except Exception as e:
        print(f"Error loading FAQ data: {e}")
        return [{"question": "default", "answer": "default"}]

def train_and_save():
    """Dipakai HANYA jika artefak belum ada (first run) atau saat reload manual."""
    faq_data = load_faq_source()
    questions_clean = [text_clean(item["question"]) for item in faq_data]

    # TF-IDF (bisa disetel sesuai kebutuhan)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X = vectorizer.fit_transform(questions_clean)

    dump(vectorizer, VEC_PATH)
    dump(X, X_PATH)
    FAQ_CACHE_PATH.write_text(json.dumps(faq_data, ensure_ascii=False), encoding="utf-8")

def artifacts_exist() -> bool:
    return VEC_PATH.exists() and X_PATH.exists() and FAQ_CACHE_PATH.exists()

def load_artifacts():
    vectorizer = load(VEC_PATH)
    X = load(X_PATH)
    faq_data = json.loads(FAQ_CACHE_PATH.read_text(encoding="utf-8"))
    return vectorizer, X, faq_data

def ensure_model_loaded():
    """Load artefak jika ada; latih dulu kalau belum ada (sekali saja)."""
    if not artifacts_exist():
        train_and_save()
    vec, X, faq = load_artifacts()
    return vec, X, faq

# --- Startup: load artefak (tanpa fit ulang) ---
@app.on_event("startup")
def _startup():
    app.state.vectorizer, app.state.X, app.state.faq = ensure_model_loaded()

# --- Sapaan ---
SAPAAAN = {"hai", "hi", "halo", "hallo", "helo"}
SAPAAAN_RESPONSE = {
    "answer": (
        "Hai! 👋 Selamat datang di *MalakaTech Assistant*.\n"
        "Saya di sini untuk membantu pertanyaan seputar layanan kami.\n\n"
        "Contoh:\n"
        "• Apa itu MalakaTech?\n"
        "• Layanan apa saja yang tersedia?\n"
        "• Kenapa harus memilih tim malakatech?"
    )
}

# --- Chatbot Endpoint ---
@app.get("/chatbot")
def chatbot(q: str = Query(..., min_length=2)):
    q_clean = text_clean(q)

    if q_clean in SAPAAAN:
        return SAPAAAN_RESPONSE

    vec = app.state.vectorizer
    X = app.state.X
    faq = app.state.faq

    if X is None or X.shape[0] == 0:
        return {"answer": "Data FAQ belum cukup, tambahkan lebih banyak pertanyaan."}

    q_vec = vec.transform([q_clean])
    sims = cosine_similarity(q_vec, X).ravel()
    best_idx = int(sims.argmax())

    if sims[best_idx] > 0.10:
        return {
            "question": faq[best_idx]["question"],
            "answer": faq[best_idx]["answer"],
            "score": float(sims[best_idx])
        }
    return {"answer": "Maaf, Saya belum mengerti pertanyaan Anda."}

# --- Opsional: Reload artefak setelah update faq_malakatech.json ---
@app.post("/reload")
def reload_model(x_admin_token: str | None = Header(default=None, alias="X-ADMIN-TOKEN")):
    if not RELOAD_TOKEN:
        raise HTTPException(status_code=403, detail="Reload dimatikan (RELOAD_TOKEN tidak diset).")
    if x_admin_token != RELOAD_TOKEN:
        raise HTTPException(status_code=401, detail="Token tidak valid.")

    train_and_save()
    app.state.vectorizer, app.state.X, app.state.faq = load_artifacts()
    return {"status": "reloaded", "items": len(app.state.faq)}
