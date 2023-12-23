from fastapi import FastAPI
import torch
from pydantic import BaseModel
from transformers import VitsModel, AutoTokenizer

class Item(BaseModel):
    text: str


app = FastAPI()

#загрузка модели обучения
model = VitsModel.from_pretrained("facebook/mms-tts-eng")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

#передача сообщения
text = 'It cost me 125 dollars and it really sucks!.'

#создание ответа при обращении (GET) к корню API
@app.get("/")
async def root():
    return {"message": "Runny nose and runny yolk Even if u have a cold still"}

#вызов метода predict API для обработки переданных (POST) данных
@app.post("/predict/")
async def predict(item: Item):

    #токенизация текста и возвращение тензора
    inputs = tokenizer(item.text, return_tensors="pt")

    #включение режима для ускорения вычислений и экономии памяти
    with torch.no_grad():
        #применение модели к тензору
        out = model(inputs["input_ids"])

        #результат формы звукового сигнала
        shape = out.waveform.shape

        #длинна последовательности звукового сигнала
        lengths = out.sequence_lengths.numpy().tolist()

    #возращение результатов в виде словаря
    return {"shape": shape,
            "lengths": lengths}
