'''
Código para utilizar determinado modelo YOLO
'''

import os
import mlflow

# Set the Tracking URI to use the dagshub server
os.environ["MLFLOW_TRACKING_URI"]='remote_tracking_uri'
os.environ["MLFLOW_TRACKING_USERNAME"]='remote_tracking_username'
os.environ["MLFLOW_TRACKING_PASSWORD"]='remote_tracking_password'

# Set a remote URI
# remote_tracking_uri same used in mlflow_version.py
mlflow.set_tracking_uri(uri='remote_tracking_uri')

# Set an experiment name
# experiment_name same used in mlflow_version.py
mlflow.set_experiment(experiment_name='experiment_name')

# the path model information is avaliable in dagshub/mlflow UI/Artifacts
logged_model = 'runs:/a2a7dbfe24dd44e1b0d398d621f934a2/artifact_model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict
loaded_model.predict(img_path='./image.jpg')