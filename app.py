import os
from flask import Flask, render_template, request, jsonify
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# -------- PDF TEXT EXTRACTION --------
def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# -------- FIND COMMON QUESTIONS --------
def find_common_questions(all_questions, threshold=0.7):
    if not all_questions:
        return []

    vectorizer = TfidfVectorizer().fit_transform(all_questions)
    sim_matrix = cosine_similarity(vectorizer)

    # Count occurrences of similar matches
    freq_counter = Counter()
    for i in range(len(all_questions)):
        for j in range(i + 1, len(all_questions)):
            if sim_matrix[i, j] >= threshold:
                freq_counter[all_questions[i]] += 1
                freq_counter[all_questions[j]] += 1

    # Rank top 10 questions
    ranked = freq_counter.most_common(10)
    results = [{"question": q, "count": c} for q, c in ranked]
    return results

# -------- ROUTES --------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_files():
    files = request.files.getlist("pdfs")
    all_text = []

    for file in files[:10]:  # limit 10 PDFs
        path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(path)
        text = extract_text(path)

        cleaned_questions = []
        current_question = ""

        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Safe check for numbered lines to prevent IndexError
            is_numbered = len(line) > 1 and line[0].isdigit() and line[1] in [".", ")"]
            is_question_start = line.lower().startswith(("q", "question")) or is_numbered

            if is_question_start:
                if current_question:
                    main_q = current_question.split(" - ")[0].strip()
                    if len(main_q.split()) > 3:
                        cleaned_questions.append(main_q)
                current_question = line
            else:
                current_question += " " + line

        if current_question:
            main_q = current_question.split(" - ")[0].strip()
            if len(main_q.split()) > 3:
                cleaned_questions.append(main_q)

        if cleaned_questions:
            all_text.extend(cleaned_questions)

    if not all_text:
        return jsonify({"error": "No questions detected in PDFs."})

    common = find_common_questions(all_text)
    return jsonify(common)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
