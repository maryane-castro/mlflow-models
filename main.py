'''
Arquivo para versionar e enviar para o servidor Dagshub
'''
import mlflow_version
import configs.pdi as pdi
import configs.nlp as nlp

import os
import warnings
os.environ["PYTHONNOUSERSITE"] = "true"
warnings.filterwarnings("ignore", category=UserWarning, module="_distutils_hack")


model = nlp.model_Price("models/ofertas.pt") # modelo para registro
params = {"image": "img"}


ver = mlflow_version.Version_model(
    # informações do servidor dagshub
    remote_tracking_uri="https://dagshub.com/maryane-castro/mlflow-models.mlflow",
    remote_tracking_username="maryane-castro",
    remote_tracking_password="574991a1da221fee01b4cf229fdd5aa681e5d927",

    # parametros do modelo
    model_params=params,
    model_metrics={"accuracy": 1.0},
    tags={"Dataset": "PDI"},
    model_name="Model_Offer",
    experiment_name= "Teste",
    description="",
    run_name="Offer_Run",
    model=model
)

ver.versioning_model()
