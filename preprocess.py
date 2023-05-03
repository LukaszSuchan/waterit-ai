import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

data = pd.read_csv('rowdata.csv')

onehot_encoder = OneHotEncoder()
crop_type_encoded = onehot_encoder.fit_transform(data[['CROP TYPE']])
label_encoder = LabelEncoder()
data['CROP TYPE'] = label_encoder.fit_transform(data['CROP TYPE'])

crop_type_encoded = pd.DataFrame(crop_type_encoded.toarray(), columns= label_encoder.classes_)

data = data.drop('CROP TYPE', axis=1)
data = pd.concat([data, crop_type_encoded], axis=1)

temperature_split = data['TEMPERATURE'].str.split('-', expand=True).astype(int)
temperature_mean = temperature_split.mean(axis=1)
data['TEMPERATURE'] = temperature_mean


label_encoder = LabelEncoder()
data['SOIL TYPE'] = label_encoder.fit_transform(data['SOIL TYPE'])

label_encoder = LabelEncoder()
data['REGION'] = label_encoder.fit_transform(data['REGION'])

label_encoder = LabelEncoder()
data['WEATHER CONDITION'] = label_encoder.fit_transform(data['WEATHER CONDITION'])

print(data)

data.to_csv('dataset.csv', index=False)
print('data saved successfully')