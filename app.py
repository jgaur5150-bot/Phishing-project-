from flask import Flask, request, jsonify
import gdown
import joblib
import pickle

# 🔽 Model download from Google Drive
url = "https://drive.google.com/uc?export=download&id=1ih2E_Uy3-QxUn4B_Wo3MwKvJesLl-EtU"
gdown.download(url, "model.pkl", quiet=False)

# 🔽 Load model & vectorizer
model = joblib.load("model.pkl")
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# 🔽 Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Flask app ready 🚀"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['url']
    data_vec = vectorizer.transform([data])
    prediction = model.predict(data_vec)

    return jsonify({
        "prediction": int(prediction[0])
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
