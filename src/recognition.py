import onnxruntime
import cv2
import numpy as np
import string

from .detection import NumbersDetection

class NumpbersRecognition(NumbersDetection):
    
    def __init__(self):
        self.detector = NumbersDetection()
        self.classes = string.digits + ' ' 
        self.session_rec = onnxruntime.InferenceSession('models/recognizer_model.onnx',providers=['CUDAExecutionProvider'])
        model_inputs_rec = self.session_rec.get_inputs()
        self.input_names_rec = [model_inputs_rec[i].name for i in range(len(model_inputs_rec))] 
        model_outputs_rec = self.session_rec.get_outputs()
        self.output_names_rec = [model_outputs_rec[i].name for i in range(len(model_outputs_rec))]
    
    def adjust_gamma(self,image, gamma=1.5):
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    
    def pred(self,img_to_rec):
        img_to_rec = cv2.cvtColor(img_to_rec, cv2.COLOR_BGR2GRAY) 
        img_to_rec = self.adjust_gamma(img_to_rec, gamma=1.5)
        _, img_to_rec = cv2.threshold(img_to_rec, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        img_to_rec = cv2.morphologyEx(img_to_rec, cv2.MORPH_CLOSE, kernel)
        img_to_rec = cv2.resize(img_to_rec, (200, 31))  # Изменение размера изображения
        img_to_rec = img_to_rec.astype(np.float32) / 255.0  # Нормализация
        img_to_rec = np.expand_dims(img_to_rec, axis=0)  # Добавление размерности batch
        img_to_rec = np.expand_dims(img_to_rec, axis=-1)
        return img_to_rec
    
    def post(self,outputs):
        predicted_text = outputs[0]
        predicted_indices = np.argmax(predicted_text, axis=2)[0]  
        predicted_chars = [self.classes[index] for index in predicted_indices if index!=10]
        predicted_string = ''.join(predicted_chars).strip()
        return predicted_string
        

    def __call__(self,image_orig):
        results = []
        images = self.detector(image_orig)
        for img_to_rec in images:
            img_to_rec = self.pred(img_to_rec)
            outputs = self.session_rec.run(None, {self.input_names_rec[0]: img_to_rec})
            results.append(self.post(outputs))
        return results
