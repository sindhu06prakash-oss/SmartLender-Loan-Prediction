from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# ===========================
# Load Model and Scaler
# ===========================
model = pickle.load(open("rdf.pkl", "rb"))
scaler = pickle.load(open("scale1.pkl", "rb"))

# ===========================
# Home Page
# ===========================
@app.route("/")
def home():
    return render_template("home.html")

# ===========================
# Prediction Page
# ===========================
@app.route("/predict")
def predict():
    return render_template("input.html")

# ===========================
# Submit Prediction
# ===========================
@app.route("/submit", methods=["POST"])
def submit():

    gender = int(request.form["Gender"])
    married = int(request.form["Married"])
    dependents = int(request.form["Dependents"])
    education = int(request.form["Education"])
    self_employed = int(request.form["Self_Employed"])
    applicant_income = float(request.form["ApplicantIncome"])
    coapplicant_income = float(request.form["CoapplicantIncome"])
    loan_amount = float(request.form["LoanAmount"])
    loan_term = float(request.form["Loan_Amount_Term"])
    credit_history = int(request.form["Credit_History"])
    property_area = int(request.form["Property_Area"])

    input_data = [[
        gender,
        married,
        dependents,
        education,
        self_employed,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_term,
        credit_history,
        property_area
    ]]

    columns = [
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self_Employed",
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
        "Property_Area"
    ]

    data = pd.DataFrame(input_data, columns=columns)

    # Scale Input
    data_scaled = scaler.transform(data)

    # Predict
    prediction = model.predict(data_scaled)
    probability = model.predict_proba(data_scaled)

    print("Prediction:", prediction)
    print("Probability:", probability)

    if prediction[0] == 1:
        return render_template(
            "output.html",
            result="Loan Approved",
            emoji="😊",
            color="green",
            probability=round(probability[0][1] * 100, 2)
        )

    else:
        return render_template(
            "output.html",
            result="Loan Not Approved",
            emoji="😞",
            color="red",
            probability=round(probability[0][0] * 100, 2)
        )


# ===========================
# Run App
# ===========================
if __name__ == "__main__":
    app.run(debug=True)