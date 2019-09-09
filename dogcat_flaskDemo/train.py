
# coding: utf-8

# In[1]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.applications.xception import Xception
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import cv2, numpy as np, os.path
from keras.models import load_model
import json

import tensorflow as tf
from keras import backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)


class Model(object):

    FILE_PATH = 'catdog.h5'

    def __init__(self):
        self.model = None

    def train(self, batch_size, classes,epochs):
        print (classes)
        self.batch_size=batch_size
        self.epochs=epochs
        resnet = Xception(include_top=False,pooling='max', weights='imagenet' , input_shape=(299,299,3))
        for layer in resnet.layers[:]:
            layer.trainable = False
        
        #RESNET_WEIGHTS_PATH = 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5' #importing a pretrained model
        self.model = Sequential()
        self.model.add(resnet)
        self.model.add(Dropout(0.25))
        self.model.add(Dense(classes))
        self.model.add(Activation('softmax'))
        #self.model.layers[0].trainable = False
        self.model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        self.model.summary()
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
        test_datagen = ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow_from_directory(
            'train2',  # this is the target directory
            target_size=(299, 299),  # all images will be resized to 150x150
            batch_size=batch_size,
            class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

        
        validation_generator = test_datagen.flow_from_directory(
            'validation',
            target_size=(299, 299),
            batch_size=batch_size,
            class_mode='categorical')

        steps_per_epoch=5000 // self.batch_size
        validation_steps=400 // self.batch_size
        
        
        self.model.fit_generator(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=self.epochs,
            validation_data=validation_generator,
            validation_steps=validation_steps)
        
  

    def save(self, file_path=FILE_PATH):
        print('Model Saved.')
        self.model.save(file_path)

    def load(self, file_path=FILE_PATH):
        print('Model Loaded.')
        self.model = load_model(file_path)

    def predict(self, image):
        
        image=cv2.resize(image,(299,299),interpolation=cv2.INTER_CUBIC)
        #print(image.shape)
        image=img_to_array(image)
        #print(image.shape)
        image = image.reshape(-1,299,299,3)
        #print(image.shape)
        image = image.astype('float32')
        image /= 255
        result = self.model.predict_proba(image)
        print(result)
        result = self.model.predict_classes(image)

        return result
    
    def predictDirClass(self, testdir):
        images = []
        results = []
        for dirname,subdir,files in os.walk(testdir):
            for file in files:
                path = os.path.join(testdir,file)
                image = cv2.imread(path)
                image=cv2.resize(image,(299,299),interpolation=cv2.INTER_CUBIC)
                image = image.reshape(299,299,3)
                image = image.astype('float32')
                image /= 255
                images.append(image)
        images = np.array(images)
        images = images.reshape(-1,299,299,3)
        results = self.model.predict_classes(images)
        return results
    
    def predictDirRes(self, testdir):
        images = []
        results = []
        for dirname,subdir,files in os.walk(testdir):
            for file in files:
                path = os.path.join(testdir,file)
                image = cv2.imread(path)
                image=cv2.resize(image,(299,299),interpolation=cv2.INTER_CUBIC)
                image = image.reshape(299,299,3)
                image = image.astype('float32')
                image /= 255
                images.append(image)
        images = np.array(images)
        images = images.reshape(-1,299,299,3)
        results = self.model.predict(images)
        classifier = {0:'cat',1:'dog'}
        prediction = []
        for result in results:
            response = {}
            response['cat'] = result[0]
            response['dog'] = result[1]
            prediction.append(response)
        return json.dumps(str(prediction))    


if __name__ == '__main__':
    model = Model()
    fname=model.FILE_PATH
    if os.path.isfile(fname) is True: 
        model.load()
        print("load model")
        image = cv2.imread('dogtest.jpg')
        
        print(model.predict(image))
        classifier = {0:'cat',1:'dog'}
        print(classifier[model.predict(image)[0]])
        
        print(model.predictDirClass('test2'))
        print(model.predictDirRes('test2'))
        print(json.loads(model.predictDirRes('test2')))
        print(eval(json.loads(model.predictDirRes('test2'))))
        print(eval(json.loads(model.predictDirRes('test2')))[0])
        print(eval(json.loads(model.predictDirRes('test2')))[0]['cat'])
        #print(model.predictDirRes('test2')[0][0]['cat'])
    else :
        model.train(batch_size=32, classes=2,epochs=15)
        model.save()
        model.load()
        print("load model")
        
        
        

    


# In[2]:



'''
model = Model()
image = cv2.imread('./data/train/5/8.jpg')

image = image[:,:,::-1]
model.load()
model.predict(image)
print(model.predict(image))

'''

