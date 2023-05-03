import pickle
import numpy as np
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel

soil_type_dict = {'DRY': 0, 'HUMID': 1, 'WET': 2}
air_humidity_type_dict = {'DESERT': 0, 'HUMID': 1, 'SEMI ARID': 2, 'SEMI HUMID': 3}
weather_condition_type = {'NORMAL': 0, 'RAINY': 1, 'SUNNY': 2, 'WINDY': 3}
crop_type_list = ['BANANA', 'BEAN', 'CABBAGE', 'CITRUS', 'COTTON', 'MAIZE', 'MELON',
       'MUSTARD', 'ONION', 'POTATO', 'RICE', 'SOYABEAN', 'SUGARCANE', 'TOMATO',
       'WHEAT']



with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

app = FastAPI()

class InputData(BaseModel):
    soil_type: str
    air_humidity_type: str
    weather_condition: str
    crop_type: str
    tempreature: float

    def create_sample(self):
        crop_type_idx =crop_type_list.index(self.crop_type) 
        sample_crop_type = np.zeros(15)
        sample_crop_type[crop_type_idx] = 1
        sample = np.array([soil_type_dict[self.soil_type], air_humidity_type_dict[self.air_humidity_type], self.tempreature,
                  weather_condition_type[self.weather_condition]])
        sample = np.concatenate([sample, sample_crop_type])
        print(sample)
        return sample

@app.post('/predict')
def predict(input_data: InputData):
    X = [input_data.create_sample()]
    y_pred = model.predict(X)
    return {'predykcja': y_pred[0]}
