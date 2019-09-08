# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 13:04:53 2019

@author: a0922
"""

from keras.models import load_model

from PIL import Image
import numpy as np
from flasgger import Swagger
from flask import Flask, request
import json

#####
import tensorflow as tf
from keras import backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)
global graph
graph = tf.get_default_graph()
#####


app = Flask(__name__)
swagger = Swagger(app)

model = load_model('./catdog.h5')

@app.route('/predict_digit', methods=['POST'])
def predict_digit():
    """Example returning a predicton of mnist
    ---
    parameters:
        - name: image
          in: formData
          type: file
          required: True
    definitions:
        value:
            type: object
            properties:
                value_name:
                    type: string
                    items:
                        $ref: '#/definitions/Color'
        Color:
            type: string
    responses:
        200:
            description: OK
            schema:
                $ref: '#/definitions/value'
    """

    
    im = Image.open(request.files['image'])
    im = im.resize((224,224))
    im2arr = np.array(im).reshape(1,224,224,3)
    response = {}
    #####
    with graph.as_default():
        result = model.predict([im2arr])[0]
        response['cat'] = result[0].item()
        response['dog'] = result[1].item()
    #####
    return json.dumps([response])

if __name__ == '__main__':
    app.run()
    
