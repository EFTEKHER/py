import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import joblib

# Load datasets
data_fake = pd.read_csv('Fake.csv')
data_true = pd.read_csv('True.csv')

# Label datasets
data_fake['class'] = 0
data_true['class'] = 1

# Combine datasets
data = pd.concat([data_fake, data_true], axis=0)
data = data.sample(frac=1).reset_index(drop=True)  # Shuffle data

# Preprocessing function
def wordopt(text):
    text = text.lower()
    text = re.sub('<.*?>', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'[\d\s]+', ' ', text)
    text = text.strip()
    return text

# Apply preprocessing
data['text'] = data['text'].apply(wordopt)

# Define features and labels
X = data['text']
y = data['class']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Vectorization
vectorization = TfidfVectorizer()
X_train_vectorized = vectorization.fit_transform(X_train)
X_test_vectorized = vectorization.transform(X_test)

# Train models
LR = LogisticRegression()
DT = DecisionTreeClassifier()
GB = GradientBoostingClassifier(random_state=0)
RF = RandomForestClassifier(random_state=0)

LR.fit(X_train_vectorized, y_train)
DT.fit(X_train_vectorized, y_train)
GB.fit(X_train_vectorized, y_train)
RF.fit(X_train_vectorized, y_train)

# Save models and vectorizer
joblib.dump(LR, 'logistic_regression_model.pkl')
joblib.dump(DT, 'decision_tree_model.pkl')
joblib.dump(GB, 'gradient_boosting_model.pkl')
joblib.dump(RF, 'random_forest_model.pkl')
joblib.dump(vectorization, 'vectorizer.pkl')

print("Models and vectorizer saved successfully.")
