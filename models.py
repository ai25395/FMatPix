from pyexpat import model
from processor import load_processor
from transformers import VisionEncoderDecoderModel, dependency_versions_check
from utils import batch_inference
import torch
import os
import sys

class OcrModel:
    def __init__(self):
        if getattr(sys, "frozen", None):
            basedir = sys._MEIPASS
        else:
            basedir = os.path.dirname(__file__)
        modelPath = os.path.join(basedir,"models")
        self.model = VisionEncoderDecoderModel.from_pretrained(modelPath)
        print('ocrmodel')     
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #device = 'cpu'
        self.model.to(device)
        self.processor = load_processor(modelPath)
        print('ocr model init')
    def predict(self,img):
        print('ocr model pred')
        res = batch_inference(img,self.model,self.processor)
        return res


