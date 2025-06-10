# requirements:
# pip install pandas spacy sentence-transformers torch transformers

import pandas as pd
import re
import json
import spacy
from sentence_transformers import SentenceTransformer, util
from transformers import T5ForConditionalGeneration, T5Tokenizer

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


# 1. Load a Kaggle resume dataset
# Fill in path to your downloaded CSV
df = pd.read_csv("resume_questions_dataset.csv")
resumes = df['resume_text'].tolist()

# 2. Load spaCy and a skills dictionary
nlp = spacy.load("en_core_web_sm")

skills = [
    "python", "sql", "machine learning", "reactjs", "html", "css",
    "nlp", "data visualization", "deep learning", "r", "javascript"
]

with open("skills_dictionary.txt", "w") as f:
    f.write("\n".join(skills))

def extract_skills(text):
    doc = nlp(text.lower())
    tokens = {token.text for token in doc}
    return sorted(tokens.intersection(skills))

# 3. Load T5 question‑generator
qg_model = T5ForConditionalGeneration.from_pretrained("valhalla/t5-small-qg-hl")
qg_tokenizer = T5Tokenizer.from_pretrained("valhalla/t5-small-qg-hl")

def generate_question(context, answer):
    input_text = f"generate question: {answer} context: {context}"
    inputs = qg_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    outs = qg_model.generate(inputs, max_length=64)
    return qg_tokenizer.decode(outs[0], skip_special_tokens=True)

# 4. Semantic similarity model for answer checking
sim_model = SentenceTransformer('all-MiniLM-L6-v2')

def similarity_score(a, b):
    emb = sim_model.encode([a, b])
    return util.cos_sim(emb[0], emb[1]).item()

# 5. Run pipeline on first resume
resume_text = resumes[0]
skills = extract_skills(resume_text)
print("Extracted skills:", skills)

qa_data = {}

for skill in skills:
    context = resume_text
    question = generate_question(context, skill)
    correct_answer = skill  # placeholder; you can refine
    qa_data[skill] = {
        "question": question,
        "answer": correct_answer
    }
print(json.dumps(qa_data, indent=2))

# 6. Example answer checking
user_ans = input(f"{qa_data[skills[0]]['question']} ")
true_ans = qa_data[skills[0]]['answer']
score = similarity_score(user_ans, true_ans)
print(f"Your answer: '{user_ans}'\nCorrect: '{true_ans}'\nSimilarity score: {score:.2f}")
