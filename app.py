from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load ML components
model = joblib.load("Models/model.pkl")
scaler = joblib.load("Models/scaler.pkl")
feature_columns = joblib.load("Models/feature_columns.pkl")
label_encoder = joblib.load("Models/label_encoder.pkl")


def process_input(form_data):
    input_data = []

    for feature in feature_columns:
        raw_val = str(form_data.get(feature, "")).strip().lower()

        # Gender encoding
        if raw_val == "male":
            val = 0
        elif raw_val == "female":
            val = 1

        # Boolean encoding
        elif raw_val in ("true", "yes", "on"):
            val = 1
        elif raw_val in ("false", "no", "off"):
            val = 0

        # Numeric
        else:
            try:
                val = float(raw_val) if raw_val != "" else 0.0
            except:
                val = 0.0

        input_data.append(val)

    X = np.array(input_data).reshape(1, -1)
    X_scaled = scaler.transform(X)

    return X_scaled


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/recommend")
def recommend():
    return render_template("recommend.html")


@app.route("/pred", methods=["POST"])
def pred():
    X_scaled = process_input(request.form)

    # If model supports probability
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_scaled)
        top_indices = probabilities[0].argsort()[-3:][::-1]

        recommendations = []
        for i in top_indices:
            study_name = label_encoder.inverse_transform([i])[0]
            probability = round(probabilities[0][i] * 100, 2)
            recommendations.append((study_name, probability))

        return render_template("results.html", recommendations=recommendations)

    # If model does NOT support probability
    else:
        prediction = model.predict(X_scaled)
        result = label_encoder.inverse_transform(prediction)[0]
        return render_template("results.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)
