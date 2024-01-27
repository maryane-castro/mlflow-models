import os
import mlflow

class Version_model:
    def __init__(
        self,
        remote_tracking_uri: str,
        remote_tracking_username: str,
        remote_tracking_password: str,
        model_params: dict,
        model_metrics: dict,
        tags: dict,
        model_name: str,
        experiment_name: str,
        description: str,
        run_name: str,
        model
        ):
        
        """
        Example paramns:
            remote_tracking_uri = "https://dagshub.com/ErikJhones/mlflow-nuven-sale.mlflow" (dagshub repository link)
            remote_tracking_username = "ErikJhones" (dagshub username)
            remote_tracking_password = "bb55efe3a7a126041aa9d49dcbb0921016ba831d" (dagshub token)
            model_params = {"solver": "newton-cholesky", "max_iter": 1000, "multi_class": "auto", "random_state": None}
            model_metrics = {"acc": 0.98, "mse": 1.0, "f1-score": 0.97}
            tags = {"Dataset": "SALE train NLP 1", "Team": "NLP"}
            model_name = "NER_NLP_complex"
            experiment_name = "NER_experiment_SALE"
            description = "Versionamento do modelo NER do projeto SALE NUVEN."
            run_name = "NLP NUVEN SALE RUN"
            model = <any tensorflow, pytorch, yolo model>
        """
        
        os.environ["MLFLOW_TRACKING_URI"]=remote_tracking_uri
        os.environ["MLFLOW_TRACKING_USERNAME"]=remote_tracking_username
        os.environ["MLFLOW_TRACKING_PASSWORD"]=remote_tracking_password
        
        self.remote_tracking_uri = remote_tracking_uri
        self.model_params = model_params
        self.model_metrics = model_metrics
        self.tags = tags
        self.model_name = model_name
        self.experiment_name = experiment_name
        self.description = description
        self.run_name = run_name
        self.model = model
     
    def versioning_model(self,):
        # Set a remote URI to upload the data
        mlflow.set_tracking_uri(uri=self.remote_tracking_uri)
        
        # Set an experiment name
        mlflow.set_experiment(experiment_name=self.experiment_name)
        
        # Start an MLflow run
        with mlflow.start_run(run_name=self.run_name, 
                              description=self.description):
            
            # Log the hyperparameters
            mlflow.log_params(self.model_params)

            # Log the loss metric
            mlflow.log_metrics(self.model_metrics)

            # Set a tag that we can use to remind ourselves what this run was for
            mlflow.set_tags(self.tags)

            mlflow.pyfunc.log_model(python_model = self.model, 
                        artifact_path = "artifact_model", 
                        registered_model_name = self.model_name)