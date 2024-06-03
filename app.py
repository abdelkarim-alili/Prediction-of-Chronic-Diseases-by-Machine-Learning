from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

# Results of Heart Disease Prediction
@app.route('/heart_disease_results', methods=['POST'])
def submit():
    if request.method == 'POST':
        try:
            age = int(request.form['age'])
            sex = int(request.form['sex'])
            cp = int(request.form['cp'])
            trestbps = int(request.form['trestbps'])
            chol = int(request.form['chol'])
            fbs = int(request.form['fbs'])
            restecg = int(request.form['restecg'])
            thalach = int(request.form['thalach'])
            exang = int(request.form['exang'])
            oldpeak = float(request.form['oldpeak'])
            slope = int(request.form['slope'])
            ca = int(request.form['ca'])
            thal = int(request.form['thal'])

            user_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

            prediction = ValuePredictor("Heart-Disease-Prediction.pkl", user_data)

            if prediction == 1:
                result = 'According to the information provided, signs of heart disease are detected.'
            else:
                result = 'According to the information provided, heart disease is not indicated.'

            return render_template('result.html', prediction=result)
        except Exception as e:
            return f"An error occurred: {e}"

# Results of Kidney Disease Prediction
@app.route('/kidney_disease_results', methods=['POST'])
def submit_kidney():
    if request.method == 'POST':
        try:
            bloodPressure = float(request.form['bloodPressure'])
            specificGravity = float(request.form['specificGravity'])
            albumin = float(request.form['albumin'])
            sugar = float(request.form['sugar'])
            redBloodCell = float(request.form['redBloodCell'])
            bloodUrea = float(request.form['bloodUrea'])
            serumCreatine = float(request.form['serumCreatine'])
            sodium = float(request.form['sodium'])
            potassium = float(request.form['potassium'])
            hemoglobine = float(request.form['hemoglobine'])
            whiteBloodCellsCount = float(request.form['whiteBloodCellsCount'])
            redBloodCellsCount = float(request.form['redBloodCellsCount'])
            hypertension = float(request.form['hypertension']) 

            user_data = np.array([[bloodPressure, specificGravity, albumin, sugar, redBloodCell, bloodUrea, serumCreatine, sodium, potassium, hemoglobine, whiteBloodCellsCount, redBloodCellsCount, hypertension]])

            prediction = ValuePredictor("decision_tree_model.pkl", user_data)

            if prediction == 1:
                result = 'According to the information provided, signs of kidney disease are detected.'
            else:
                result = 'According to the information provided, kidney disease is not indicated.'

            return render_template('result.html', prediction=result)
        except Exception as e:
            return f"An error occurred: {e}"
    
def ValuePredictor(model, user_data):
    loaded_model = pickle.load(open(model, "rb"))
    result = loaded_model.predict(user_data)
    return result[0]

if __name__ == '__main__':
    app.run(debug=True)
