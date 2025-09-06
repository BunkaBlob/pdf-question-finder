import os
from flask import Flask, render_template, request, jsonify
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import re

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# -------- PDF TEXT EXTRACTION --------
def extract_text(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return ""

# -------- FIND COMMON QUESTIONS --------
def find_common_questions(all_questions, threshold=0.7):
    if not all_questions:
        return []

    vectorizer = TfidfVectorizer().fit_transform(all_questions)
    sim_matrix = cosine_similarity(vectorizer)

    freq_counter = Counter()
    n = len(all_questions)
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= threshold:
                freq_counter[all_questions[i]] += 1
                freq_counter[all_questions[j]] += 1

    # Include questions with no similar pair
    for q in all_questions:
        if q not in freq_counter:
            freq_counter[q] = 1

    ranked = freq_counter.most_common()
    results = [{"question": q, "count": c} for q, c in ranked]
    return results

# -------- ROUTES --------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_files():
    files = request.files.getlist("pdfs")
    if not files:
        return jsonify({"error": "No files received."})

    all_questions = []

    # Flexible question pattern:
    # Starts with Q/q, optional spaces, optional digits, then ) . or :
    question_pattern = re.compile(r"^Q\s*\d*[\)\.:]\s*", re.IGNORECASE)

    for file in files[:10]:
        path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(path)
        text = extract_text(path)

        lines = [line.strip() for line in text.split("\n") if line.strip()]
        current_question = ""
        recording = False

        for line in lines:
            if question_pattern.match(line):
                # Start of new question
                if current_question:
                    if '?' in current_question:
                        q_text = current_question.split('?')[0].strip() + '?'
                        if len(q_text.split()) > 3:
                            all_questions.append(q_text)
                    current_question = ""

                # Remove the Q prefix
                current_question = question_pattern.sub("", line)
                recording = True
            elif recording:
                # Continue multi-line question
                current_question += " " + line

            # Finalize when ? is found
            if recording and '?' in line:
                q_text = current_question.split('?')[0].strip() + '?'
                if len(q_text.split()) > 3:
                    all_questions.append(q_text)
                current_question = ""
                recording = False

        # Add last question if it ends with ?
        if current_question and '?' in current_question:
            q_text = current_question.split('?')[0].strip() + '?'
            if len(q_text.split()) > 3:
                all_questions.append(q_text)

    if not all_questions:
        return jsonify({"error": "No questions detected in PDFs."})

    common = find_common_questions(all_questions)
    return jsonify(common)

# -------- RUN APP (Render-ready) --------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render sets PORT environment variable
    app.run(host="0.0.0.0", port=port, debug=False)
