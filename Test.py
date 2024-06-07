# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 23:56:10 2023

@author: Elif KAR 20190203011
"""

import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Önceden eğitilmiş modeli yükleme
model = load_model('COP_AYIRAN_MODEL.h5')

# Sınıflar
classes = ['cam', 'kağıt']

# Test resimlerini içeren bir dizin
test_image_directory = 'veriseti/TEST DENEME/KARIŞIK'

# Test resimlerini yükleyip tahmin etme
for filename in os.listdir(test_image_directory):
    img_path = os.path.join(test_image_directory, filename)
    
    # Resmi modelin beklentilerine uygun şekilde yükleme
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalizasyon

    # Tahmin
    predictions = model.predict(img_array)

    # En yüksek tahmin olasılığına sahip sınıfı bulma
    predicted_class_index = np.argmax(predictions)
    predicted_class = classes[predicted_class_index]

    # Tahmin sonuçlarını yazdırma
    print(f"Resim: {filename}, Tahmin: {predicted_class}, Tahmin Olasılığı: {predictions[0][predicted_class_index]}")

