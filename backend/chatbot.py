from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Mengatasi CORS agar bisa diakses dari frontend React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load data FAQ dari JSON
def load_faq_data():
    try:
        with open("faq_malakatech.json", "r", encoding="utf-8") as file:
            data = json.load(file)
            return data if data else [{"question": "default", "answer": "default"}]  # Mencegah error jika kosong
    except Exception as e:
        print(f"Error loading FAQ data: {e}")
        return [{"question": "default", "answer": "default"}]

faq_data = load_faq_data()

# Ambil semua pertanyaan untuk diproses TF-IDF
questions = [item["question"] for item in faq_data]

# Pastikan ada lebih dari satu data agar TF-IDF bisa berjalan
if len(questions) > 1:
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(questions)
else:
    vectorizer = None
    X = None

# Tambahkan list sapaan
sapaan_keywords = ["hai", "hi", "halo", "hallo", "helo"]
sapaan_response = {
    "answer": (
        "Hai hai! 🤖 Si Tegar siap bantu kamu, tanya aja yuk!\n"
        "Berikut beberapa hal yang bisa Tegar bantu:\n"
        "• Siapa Tegar?\n"
        "• Apa keahlian Tegar?\n"
        "• Bisa lihat biodata Tegar?\n"
        "• Apakah Tegar tersedia untuk freelance?\n"
        "• Layanan apa saja yang ditawarkan Tegar?"
    )
}

# API Endpoint untuk chatbot FAQ
@app.get("/chatbot")
def chatbot(q: str = Query(..., min_length=2)):
    q_lower = q.lower().strip()

    # Cek apakah input termasuk sapaan
    if q_lower in sapaan_keywords:
        return sapaan_response

    if vectorizer is None or X is None or X.shape[0] == 0:  
        return {"answer": "Data FAQ belum cukup, tambahkan lebih banyak pertanyaan."}

    # Transformasi pertanyaan pengguna ke dalam vektor
    query_vec = vectorizer.transform([q])

    # Hitung kemiripan dengan semua pertanyaan di database
    similarities = cosine_similarity(query_vec, X).flatten()

    # Ambil indeks pertanyaan yang paling mirip
    best_match_idx = similarities.argmax()

    # Threshold agar pertanyaan bisa dijawab
    if similarities[best_match_idx] > 0.1:
        return {"question": q, "answer": faq_data[best_match_idx]["answer"]}

    return {"answer": "Maaf, Tegar belum mengerti pertanyaan Anda."}
