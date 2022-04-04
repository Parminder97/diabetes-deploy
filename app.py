from flask import Flask, render_template, request
import pickle
import numpy as np


app = Flask("app")
loaded_model = pickle.load(open('Modeldb.pkl', 'rb'))


@app.route("/")
def home():
    return render_template("form.html")

@app.route("/Predictions", methods=['POST'])
def predict():
    Glucose = request.form["Glucose"]
    BloodPressure = request.form["BloodPressure"]
    SkinThickness = request.form["SkinThickness"]
    Insulin = request.form["Insulin"]
    BMI = request.form["BMI"]
    Age = request.form["Age"]

    predictions = loaded_model.predict([[Glucose,BloodPressure,SkinThickness,Insulin,BMI,Age]])[0]
    probability = loaded_model.predict_proba([[Glucose,BloodPressure,SkinThickness,Insulin,BMI,Age]])
    probability = np.round((np.max(probability) * 100),2)
    output = ""
    probability = f"{probability}%"     

    if predictions == 0:
      output = "Not diabetic"
    else:
      output = "Diabetic"
    print(predictions,probability)
    return render_template("form.html", output_predictions=output, output_proba=probability)

#Main function
if __name__ == '__main__':
    app.run(debug=True)
