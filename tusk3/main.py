from fastapi import FastAPI
import torch
from pydantic import BaseModel
from transformers import VitsModel, AutoTokenizer
import uvicorn


#Вау это что функция, это практически ООП какое то пошло
def load_model():
    model = VitsModel.from_pretrained("facebook/mms-tts-eng")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
    return tokenizer, model

#Штука для uvicorn'а
my_app = FastAPI()
#Ну тут короч модель гружу
tokenizer, model = load_model()

#Это для POST запросов
class Item(BaseModel):
    text: str

#GET для рута
@my_app.get("/")
async def root():
    return {"message": "Hello World"}

#GET для директории предикт
@my_app.get("/predict/")
async def predict():
    #Наша модель из торча
    text = 'Do you want a totally war?'
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        # Массив циферок
        output = model(**inputs).waveform.float().numpy().tolist()
    return {"music": [output]}

#Собственно сам POST
@my_app.post("/predict/")
async def predict(item: Item):
    # Наша модель из торча
    inputs = tokenizer(item.text, return_tensors="pt")
    with torch.no_grad():
        #Массив циферок
        output = model(**inputs).waveform.float().numpy().tolist()
    return {"music": output}





