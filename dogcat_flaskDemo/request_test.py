# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 16:39:18 2019

@author: a0922
"""

import requests

url = 'http://localhost:5000/predict_digit'
image_path = 'dogtest.jpg'
files = {
        'image':open(image_path,'rb'),
        'Content-Type': 'image/jpeg',
        }

response = requests.post(url, files=files)
print(response.json())