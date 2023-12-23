from fastapi.testclient import TestClient
from main1 import app

#создание клиента для тестирования и передача ему объекта app из API 
client = TestClient(app)


#Тест №1
'''вывод ответа приложения при обращении к корню'''
def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message":
        "Runny nose and runny yolk Even if u have a cold still"}


#Тест №2
'''проверка константного размера пакета данных звукового сигнала waveform
при отправке запроса на предсказание модели'''
def test_batch_size():
    response = client.post("/predict/",
                            json={"text": "Memoria"})
    json_data = response.json()
    assert response.status_code == 200
    assert json_data['shape'][0] == 1

    
#Тест №3
'''проверка совпедания длины переданного текста с длиной полученной из пакета
данных звукового сигнала метода waveform'''
def test_lengths():
    response = client.post("/predict/",
                            json={"text": "Skin the sun"})
    json_data = response.json()
    assert response.status_code == 200
    assert json_data['lengths'][0] == json_data['shape'][1]
