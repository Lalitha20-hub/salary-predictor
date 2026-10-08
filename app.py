
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("salary_model.pkl")
scaler = joblib.load("scaler.pkl")



@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    experience = float(request.form["experience"])
    experience_scaled = scaler.transform([[experience]])
    prediction = model.predict(experience_scaled)

    return render_template(
        "index.html",
        prediction_text=f"Predicted Salary: ₹{prediction[0]:,.2f}"
    )

if __name__ == "__main__":
    app.run(debug=True)