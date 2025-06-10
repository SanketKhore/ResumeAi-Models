import torch
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertModel
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Step 1: Custom Dataset
class ResumeDataset(Dataset):
    def __init__(self, texts, targets, tokenizer, max_length=512):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].squeeze()
        attention_mask = encoded["attention_mask"].squeeze()
        target = torch.tensor(self.targets[idx], dtype=torch.float)
        return input_ids, attention_mask, target

# Step 2: Regression Model
class DistilBERTRegressor(nn.Module):
    def __init__(self):
        super(DistilBERTRegressor, self).__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = output.last_hidden_state[:, 0, :]
        return self.regressor(cls_token).squeeze()

# Step 3: Train Function
def train_model(texts, targets, score_name, model_save_path):
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    X_train, X_val, y_train, y_val = train_test_split(texts, targets, test_size=0.1, random_state=42)

    train_dataset = ResumeDataset(X_train, y_train, tokenizer)
    val_dataset = ResumeDataset(X_val, y_val, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    model = DistilBERTRegressor()
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(3):  # Increase for better performance
        total_loss = 0
        for input_ids, attention_mask, targets in tqdm(train_loader, desc=f"Training {score_name}"):
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"{score_name} Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), model_save_path)
    print(f"{score_name} model saved at: {model_save_path}")
    return model

# Step 4: Load Data and Train All 5 Models
df = pd.read_csv("resume_scores_dataset.csv")  # Make sure it includes all 5 score columns
texts = df["resume_text"].tolist()

# Train each score model
score_models = [
    ("grammar_score", "Grammar Score", "grammar_model.pt"),
    ("ats_score", "ATS Score", "ats_model.pt"),
    ("readability_score", "Readability Score", "readability_model.pt"),
    ("content_score", "Content Score", "content_model.pt"),
    ("clarity_score", "Clarity Score", "clarity_model.pt"),
]

for column, name, path in score_models:
    print(f"\n=== Training {name} Model ===")
    train_model(texts, df[column].tolist(), name, path)


# resume analysis
def predict_score(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        return outputs["logits"].item()

# Example
print(predict_score("This resume is very well written with clear structure and no grammar issues."))

