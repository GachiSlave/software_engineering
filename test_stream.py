from streamlit.testing.v1 import AppTest
import streamlit as st
import streamlit_app
import torch


at = AppTest.from_file("streamlit_app.py", default_timeout=1000).run()


# Тест №1
def test_run_app():
    '''Проверка загрузки модели, заголовка и приветственной строчки для ввода текста'''
    at.run()
    at.title[0].value == "Streamlit текст-в-речь переводчик!" == True
    at.text_input[0].value == "Введите текст:" == True
    assert not at.exception


# Тест №2
def test_text_input_button():
    '''Проверка корректности распознавания текста и генерации речи'''
    at.text_input[0].input("i purrrrred you on thursday, but (i'm) there are seven fridays in the week").run()
    at.button[0].click().run()    
    assert not at.exception


# Тест №3
def test_audio_output():
    '''Проверка генерации речи и вывода аудио'''
    at.button[0].click().run()
    inputs = streamlit_app.tokenizer('everyday rain', return_tensors="pt")
    with torch.no_grad():
        output = streamlit_app.model(**inputs).waveform
        st.audio(output.float().numpy(), sample_rate=streamlit_app.model.config.sampling_rate)
        assert not at.exception


# Тест №4
def test_empty_text_input_button():
    '''Проверка распознавания пустой строчки при генерации речи (появляется ошибка в приложении)'''
    at.text_input[0].input("").run()
    at.button[0].click().run()
    assert at.exception

    
# Тест №5
def test_symbols_input_button():
    '''Проверка корректности введенного текста для генерации речи (появляется ошибка в приложении)'''
    at.text_input[0].input("(；⌣̀︹⌣́)").run()
    at.button[0].click().run()
    assert at.exception


# Тест №6
def test_leng_input_button():
    '''Проверка языка введенного текста для генерации речи (появляется ошибка в приложении)'''
    at.text_input[0].input("когда останется немного в этом медленном пути").run()
    at.button[0].click().run()
    assert at.exception
