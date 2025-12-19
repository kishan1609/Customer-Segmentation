from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

kmeans = joblib.load("models/kmeans.pkl")
pca = joblib.load("models/pca.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_names = joblib.load("models/features.pkl")

persona_map = {
    0: "Budget-Conscious Families – Offer discounts & coupons",
    1: "Older Conservative Families – Loyalty programs",
    2: "Premium Customers – High-value & premium offers",
    3: "Mass Family Customers – Bundles & value packs"
}

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    age = float(request.form['age'])
    income = float(request.form['income'])
    spent = float(request.form['spent'])
    family_size = float(request.form['family_size'])

    input_dict = {
        "Age": age,
        "Income": income,
        "Spent": spent,
        "Family_Size": family_size
    }

    # create full feature vector (22 features)
    X = np.array([[input_dict.get(col, 0) for col in feature_names]])

    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)

    cluster = kmeans.predict(X_pca)[0]
    persona = persona_map[cluster]

    return render_template(
        "result.html",
        cluster=cluster,
        persona=persona
    )

if __name__ == "__main__":
    app.run(debug=True)