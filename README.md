# Loan_Approval_Prediction

import dagshub
dagshub.init(repo_owner='Rupsa9647', repo_name='Loan_Approval_Prediction', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)
  
# Loan_Approval_Prediction

## Workflows

1. Update config.yaml
2. Update schema.yaml
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline 
8. Update the main.py
9. Update the app.py



# How to run?
### STEPS:

Clone the repository

```cmd
https://github.com/Rupsa9647/Loan_Approval_Prediction
```
### STEP 01- Create a conda environment after opening the repository

```cmd
python -m venv myenv
```

```cmd
myenv\Scripts\Activate
```


### STEP 02- install the requirements
```cmd
pip install -r requirements.txt
```


```cmd
# Finally run the following command
python app.py
```

Now,
```cmd
open up you local host and port
```



## MLflow

[Documentation](https://mlflow.org/docs/latest/index.html)


##### cmd
- mlflow ui

### dagshub
[dagshub](https://dagshub.com/)

MLFLOW_TRACKING_URI=https://dagshub.com/Rupsa9647/Loan_Approval_Prediction.mlflow \
MLFLOW_TRACKING_USERNAME=Rupsa9647\
MLFLOW_TRACKING_PASSWORD=e894e35cce552accce9e3052a72d090789973129 \
python script.py


Run this to export as env variables:

```cmd

set MLFLOW_TRACKING_URI=https://dagshub.com/Rupsa9647/Loan_Approval_Prediction.mlflow 
set MLFLOW_TRACKING_USERNAME=Rupsa9647
set MLFLOW_TRACKING_PASSWORD=e894e35cce552accce9e3052a72d090789973129 
python script.py

```

## About MLflow 
MLflow

 - Its Production Grade
 - Trace all of your expriements
 - Logging & tagging your model


