import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
import string

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Load your labeled resume dataset
df = pd.read_csv("resume_feedback_dataset.csv")  # Make sure your file has "text" and "grammar_label" columns

# Basic text cleaning function
def clean_text(text):
    text = text.lower()
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df["clean_text"] = df["text"].apply(clean_text)

# Vectorize text
tfidf = TfidfVectorizer(max_features=300)
X = tfidf.fit_transform(df["clean_text"]).toarray()

# Encode labels (e.g., Good => 2, Average => 1, Poor => 0)
le = LabelEncoder()
y = le.fit_transform(df["grammar_label"])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the XGBoost model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
model.fit(X_train, y_train)

# Save model and vectorizer
import joblib
joblib.dump(model, "xgb_grammar_model.pkl")
joblib.dump(tfidf, "xgb_tfidf_vectorizer.pkl")
joblib.dump(le, "xgb_label_encoder.pkl")

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))


def predict_grammar_label(new_text):
    # Load everything
    model = joblib.load("xgb_grammar_model.pkl")
    tfidf = joblib.load("xgb_tfidf_vectorizer.pkl")
    le = joblib.load("xgb_label_encoder.pkl")

    # Clean and transform
    clean = clean_text(new_text)
    vector = tfidf.transform([clean])
    
    # Predict
    pred = model.predict(vector)
    label = le.inverse_transform(pred)
    
    return label[0]