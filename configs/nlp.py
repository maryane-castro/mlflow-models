'''
Arquivo NLP com função de modelo de inicialização e de predição
'''
import mlflow

from torch import argmax, cuda
from transformers import BertTokenizerFast, BertForTokenClassification
import re

from paddleocr import PaddleOCR
from paddleocr.tools.infer.utility import init_args
from paddleocr.tools.infer.predict_rec import TextRecognizer



class model_NER(mlflow.pyfunc.PythonModel):
    def __init__(self,):
        self.device = 'cuda' if cuda.is_available() else 'cpu'
        self.MAX_LEN = 128
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

        self.labels_to_ids = {'B-bra': 1,
        'B-cat': 2,
        'B-tip': 5,
        'B-unit': 3,
        'I-bra': 4,
        'I-cat': 7,
        'I-tip': 6,
        'I-unit': 8,
        'B-sep':9,
        'O': 0}

        self.ids_to_labels = {1: 'B-bra',
        2:'B-cat',
        5:'B-tip',
        3:'B-unit',
        4:'I-bra',
        7:'I-cat',
        6:'I-tip',
        8:'I-unit',
        9:'B-sep',
        0:'O'}

        self.model = BertForTokenClassification.from_pretrained('models/model_ner', num_labels=len(self.labels_to_ids))
        self.model.to(self.device)
    def prepare_sentence(self, sentence: str) -> str:
        """
        Método para preparar a sentença, mudando para caixa baixa e dividindo as vírgulas e barras.

        Args:
            sentence (str): texto a ser processado.

        Returns:
            str: texto recebido após o processamento aplicado.
        """
        sentence = re.sub(r'(?<=\d),(?=\d)', '.', sentence)
        sentence = sentence.lower()
        sentence = sentence.replace(',', ' , ')
        if 'c/' not in sentence and 'c /' not in sentence:
            sentence = sentence.replace('/', ' / ')
        sentence = sentence.replace('\\', ' \ ')

        return sentence
    
    def predict(self, sentence):
        
        inputs = self.tokenizer(sentence.split(),
                    is_pretokenized=True,
                    return_offsets_mapping=True,
                    padding='max_length',
                    truncation=True,
                    max_length=self.MAX_LEN,
                    return_tensors="pt")

        # move to gpu
        ids = inputs["input_ids"].to(self.device)
        mask = inputs["attention_mask"].to(self.device)
        # forward pass
        outputs = self.model(ids, attention_mask=mask)
        logits = outputs[0]

        active_logits = logits.view(-1, self.model.num_labels) # shape (batch_size * seq_len, num_labels)
        flattened_predictions = argmax(active_logits, axis=1) # shape (batch_size*seq_len,) - predictions at the token level

        tokens = self.tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
        token_predictions = [self.ids_to_labels[i] for i in flattened_predictions.cpu().numpy()]
        wp_preds = list(zip(tokens, token_predictions)) # list of tuples. Each tuple = (wordpiece, prediction)

        prediction = []
        for token_pred, mapping in zip(wp_preds, inputs["offset_mapping"].squeeze().tolist()):
            #only predictions on first word pieces are important
            if mapping[0] == 0 and mapping[1] != 0:
                prediction.append(token_pred[1])
            else:
                continue
            
        return prediction
    

class model_Price(mlflow.pyfunc.PythonModel):
    def __init__(self, model_args):
        """
        Classe de um modelo local da Paddle para o reconhecimento de texto.

        Args:
            model_args (dict): dicionário contendo os argumentos necessário para o carregamento do modelo.
            CustomPaddleOcrRec({
            'rec_model_dir': 'models/price_model/ocr',  #local do modelo
            'rec_char_dict_path': 'models/price_model/price_dict.txt', #local do arquivo txt
            'rec_image_shape': '3, 32, 100'}) 

        """
        self.recognizer = self.__load_model(model_args)

    def __load_model(self, model_args):
        """
        Método privado para carregar o modelo da Paddle.

        Args:
            model_args (dict): dicionário contendo os argumentos necessário para o carregamento do modelo.

        Returns:
            TextRecognizer: modelo para OCR.
        """
        args = init_args()
        for arg, value in model_args.items():
            args.set_defaults(**{arg: value})
        rec_model = TextRecognizer(args.parse_args())
        return rec_model

    def predict(self, image):
        """
        Método para realizar a reconhecimento do texto de interesse em uma imagem.

        Args:
            image (ndarray): imagem para aplicar o OCR.

        Returns:
            str: texto detectado pelo OCR.
            float: pontuação de confiância da resposta do modelo.
        """
        result, _ = self.recognizer([image])
        text = result[0][0]
        confidence_score = result[0][1]
        return text, confidence_score

