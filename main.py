import mlflow_version
import pdi

ner = pdi.YOLO_OFERTA_SALE("models/ofertas.pt") 

params = {"image": "img"}

ver = mlflow_version.Version_model(
        remote_tracking_uri = "https://dagshub.com/maryane-castro/mlflow-models.mlflow",
        remote_tracking_username = "maryane-castro",
        remote_tracking_password = "574991a1da221fee01b4cf229fdd5aa681e5d927",
        model_params = params,
        model_metrics = {"accuracy": 1.0},
        tags = {"Dataset": "Offer", "Team": "PDI"},
        model_name = "Model_Offer",
        experiment_name = "experiment_Offer",
        description = "teste de versiona,emy.",
        run_name = "Offer_Run",
        model = ner
    )

ver.versioning_model()


# MLFLOW_TRACKING_URI=https://dagshub.com/maryane-castro/mlflow-models.mlflow \
# MLFLOW_TRACKING_USERNAME=maryane-castro \
# MLFLOW_TRACKING_PASSWORD=574991a1da221fee01b4cf229fdd5aa681e5d927 \
# python script.py