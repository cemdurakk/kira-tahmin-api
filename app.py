from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Modeli yükle
model = joblib.load("kira_tahmin_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("📥 Gelen veri:", data)  # 👈 👈 👈 TAM BURASI 🔥

        input_df = pd.DataFrame([data])

        for col in model.feature_names_in_:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[model.feature_names_in_]

        prediction = model.predict(input_df)[0]
        return jsonify({'tahmini_kira': round(prediction, 2)})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
