# software_engineering
Practical lessons in software engineering number.

## Участники команды:

Луговых Владислав Витальевич РИМ-130908

Ахаимов Данила Игоревич РИМ-130907

Чудинова Алена Сергеевна РИМ-130907

Холоденко Мария Дмитриевна РИМ-130907
____

## Описание модели:
С помощью рассматриваемой модели выполняется перевод текста на английском языке в речь при помощи готовой библиотеки ml, установить которую можно по ссылке:
>https://huggingface.co/facebook/mms-tts-eng/tree/main

### Использование модели:
```
from transformers import VitsModel, AutoTokenizer
import torch

model = VitsModel.from_pretrained("facebook/mms-tts-eng")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

text = "some example text in the English language"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    output = model(**inputs).waveform
```
## Разработка Web-приложения:
#### Установка необходимых пакетов:

***Созданный файл requirements.txt содержит список всех необходимых библиотек***
```
pip install -r requirements.txt
```
***Запуск приложения***
```
streamlit run ./streamlit_app.py
```
## Развертывание Web приложения в облаке Streamlit:

Веб-приложение доступно по ссылке:
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://appapppy-yp3guae6xlaon7c8ausr4p.streamlit.app/)

## Тесты:
-Разработаны тесты (test_stream.py) модели машинного обучения(streamlit_app.py).


