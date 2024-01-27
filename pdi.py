import mlflow
from ultralytics import YOLO
import sys

class YOLO_OFERTA_SALE(mlflow.pyfunc.PythonModel):
    def __init__(self, path_model):
        try:
            self.model_oferta = YOLO(path_model)
        except:
            print("\nERRO: Não foi possível encontrar os pesos para o modelo Ofertas em './Yolo Ofertas v1/best.pt'")
            sys.exit(1)

    def predict(self, img_path):
        results = self.model_oferta(img_path)
        return results
    

class YOLO_RECORTE_SALE(mlflow.pyfunc.PythonModel):
    def __init__(self, path_model):
        try:
            self.model_recorte = YOLO(path_model)
        except:
            print("\nERRO: Não foi possível encontrar os pesos para o modelo Recortes em './Yolo Recortes v1/best.pt'")
            sys.exit(1)

    def predict(self, img_path):
        results = self.model_recorte(img_path)
        return results