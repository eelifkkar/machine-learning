# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 23:58:30 2023

@author: Elif KAR 20190203011
"""

import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image

# Önceden eğitilmiş modeli yükleme
model = load_model('COP_AYIRAN_MODEL.h5')

# Sınıflar
classes = ['cam', 'kağıt']

# Test resimlerini içeren bir dizin
test_image_directory = 'veriseti/TEST DENEME/KARIŞIK'

# Bir figür oluştur
plt.figure(figsize=(15, 10))

# Test resimlerini yükleyip tahmin etme
for i, filename in enumerate(os.listdir(test_image_directory)):
    img_path = os.path.join(test_image_directory, filename)
    
    # Resmi modelin beklentilerine uygun şekilde yükleme ve boyutlandırma
    img = Image.open(img_path)
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalizasyon

    # Tahmin
    predictions = model.predict(img_array)

    # En yüksek tahmin olasılığına sahip sınıfı bulma
    predicted_class_index = np.argmax(predictions)
    predicted_class = classes[predicted_class_index]

    # Resmi gösterme
    plt.subplot(3, 5, i+1)
    plt.imshow(img)
    plt.axis('off')

    # Tahmin sonuçlarını yazdırma
    plt.title(f"Resim: {filename}\nTahmin: {predicted_class}\nTahmin Olasılığı: {predictions[0][predicted_class_index]}")

# Birleştirilmiş bir grafik penceresini gösterme
plt.show()
