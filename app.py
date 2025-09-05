from flask import Flask, render_template, request
import os 
import numpy as np
import pandas as pd
from mlProject.pipeline.prediction import PredictionPipeline


app = Flask(__name__)  # initializing a flask app


@app.route('/', methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")


@app.route('/train', methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!"


@app.route('/predict', methods=['POST', 'GET'])  # route to show the predictions in a web UI
def index():
    if request.method == 'POST':
        try:
            # ✅ Reading the inputs given by the user
            no_of_dependents = int(request.form['no_of_dependents'])
            education = int(request.form['education'])
            self_employed = int(request.form['self_employed'])
            income_annum = int(request.form['income_annum'])
            loan_amount = int(request.form['loan_amount'])
            loan_term = int(request.form['loan_term'])
            cibil_score = int(request.form['cibil_score'])
            total_assets = int(request.form['total_assets'])

            # ✅ Put values into DataFrame
            data = pd.DataFrame([[
                no_of_dependents,
                education,
                self_employed,
                income_annum,
                loan_amount,
                loan_term,
                cibil_score,
                total_assets
            ]], columns=[
                'no_of_dependents', 'education', 'self_employed',
                'income_annum', 'loan_amount', 'loan_term',
                'cibil_score', 'total_assets'
            ])

            # ✅ Load pipeline and predict
            obj = PredictionPipeline()
            predict = obj.predict(data)[0]

            result_text = "✅ Loan Approved" if predict == 1 else "❌ Loan Rejected"

            return render_template('results.html', prediction=result_text)

        except Exception as e:
            print('The Exception message is: ', e)
            return 'Something went wrong!'
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
