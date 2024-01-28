'''
Arquivo PDI com função de modelo de inicialização e de predição
'''
import mlflow

from ultralytics import YOLO
import sys

class model_Offer(mlflow.pyfunc.PythonModel):
    def __init__(self, path_model):
        try:
            self.model_oferta = YOLO(path_model)
            print("ok")
        except:
            print("\nERRO: Não foi possível encontrar os pesos para o modelo Ofertas")
            sys.exit(1)

    def predict(self, img_path):
        results = self.model_oferta(img_path)
        return results
    

class model_Snip(mlflow.pyfunc.PythonModel):
    def __init__(self, path_model):
        try:
            self.model_recorte = YOLO(path_model)
        except:
            print("\nERRO: Não foi possível encontrar os pesos para o modelo Recortes")
            sys.exit(1)

    def predict(self, img_path):
        results = self.model_recorte(img_path)
        return results