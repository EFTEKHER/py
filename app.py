from flask import Flask, request, jsonify, render_template
import joblib
import re
import string

# Load models and vectorizer
LR = joblib.load('logistic_regression_model.pkl')
DT = joblib.load('decision_tree_model.pkl')
GB = joblib.load('gradient_boosting_model.pkl')
RF = joblib.load('random_forest_model.pkl')
vectorization = joblib.load('vectorizer.pkl')

app = Flask(__name__)

def wordopt(text):
    text = text.lower()
    text = re.sub('<.*?>', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'[\d\s]+', ' ', text)
    text = text.strip()
    return text

def output_label(n):
    return "Fake news" if n == 0 else "Not a Fake news"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news = request.json['news']
    processed_news = wordopt(news)
    vectorized_news = vectorization.transform([processed_news])
    
    pred_LR = LR.predict(vectorized_news)
    pred_DT = DT.predict(vectorized_news)
    pred_GB = GB.predict(vectorized_news)
    pred_RF = RF.predict(vectorized_news)
    
    return jsonify({
        'LR prediction': output_label(pred_LR[0]),
        'DT prediction': output_label(pred_DT[0]),
        'GB prediction': output_label(pred_GB[0]),
        'RF prediction': output_label(pred_RF[0])
    })

if __name__ == '__main__':
    app.run(debug=True)
