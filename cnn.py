#importting the Keras Libraries and packages
import keras
#from keras import models
#from sklearn.preprocessing import MinMaxScaler
from keras .models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.optimizers import Adam
#import tensorflow as tf
#Initializing the CNN
#scalar = MinMaxScaler()
classifier =Sequential()
#Convolution
classifier.add(Convolution2D(32,3,3, input_shape=(64,64,3),activation='relu')) #32 filters (3,3)Conv window, 64*64 imagedimention 3-RGB
#Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))#(2,2)feature detector
#Addition of Second convolutional layers and pooling to get more accuracy on the test_set
classifier.add(Convolution2D(32,3,3,activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
#Flattening
classifier.add(Flatten())#to bring the layer to 1D tensor

#Full connection
classifier.add(Dense(output_dim=128,activation='relu'))#dense works as nodes(example 0 and 1)
classifier.add(Dense(output_dim=1 ,activation='sigmoid'))
#classifier.add(Dense(output_dim=2,activation='softmax'))


#Compiling the CNN

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#classifier.compile(Adam(lr=.0001),loss='binary_crossentropy',metrics=['accuracy'])

#Image preprocessing-fitting the CNN to image
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)


training_set = train_datagen.flow_from_directory('F:\My Machine Learning files\Cataract\Train',
                                                   target_size=(64, 64),
                                                    batch_size=10,
                                                    #batch_size=855,
                                                    class_mode='binary')

test_set = test_datagen.flow_from_directory('F:\My Machine Learning files\Cataract\Test',
                                                target_size=(64, 64),
                                                 batch_size=10,
                                                 #batch_size=20,
                                                 class_mode='binary')

history =classifier.fit_generator(training_set,
                            #steps_per_epoch=4,#epochs=(total images divided by batch_size)
                             steps_per_epoch=8068,
                            
                            epochs=10,#generally 1 step more than steps_per_epoch
                            validation_data=test_set,
                            nb_val_samples=1600
                            )
#verbose=numebrs of output to be printed in the console

#fine tunning the model
vgg16_classifier= keras.applications.vgg16.VGG16()
vgg16_classifier.summary()
type(vgg16_classifier)
classifier=Sequential()
for layer in vgg16_classifier.layers:
    classifier.add(layer)
    
classifier.summary()
classifier.layers.pop()
classifier.summary()
for layer in classifier.layers:
    layer.trainable=False

#classifier.add(Dense(output_dim=80,activation='relu'))#dense works as nodes(example 0 and 1)
#classifier.add(Dense(output_dim=1 ,activation='sigmoid'))    classifier.add(Dense(output_dim=2,activation='softmax'))
classifier.summary()


classifier.compile(Adam(lr=.0001),loss='binary_crossentropy',metrics=['accuracy'])


#Image preprocessing-fitting the CNN to image
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)


training_set = train_datagen.flow_from_directory('F:\My Machine Learning files\Cataract\Train',
                                                   target_size=(64, 64),
                                                    #batch_size=9,
                                                    batch_size=855,
                                                    class_mode='binary')

test_set = test_datagen.flow_from_directory('F:\My Machine Learning files\Cataract\Test',
                                                target_size=(64, 64),
                                                 #batch_size=9,
                                                 batch_size=20,
                                                 class_mode='binary')

history =classifier.fit_generator(training_set,
                            steps_per_epoch=4,#epochs=(total images divided by batch_size)
                            #steps_per_epoch=3420,
                            
                            epochs=5,#generally 1 step more than steps_per_epoch
                            validation_data=test_set,
                            nb_val_samples=80
                            )

    

#classification predict
classifier.summary()

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('F:\My Machine Learning files\Cataract\Test\Cataract\cat_0_19.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'normal'
else:
    prediction = 'cataract'
    
print(prediction)
X = scalar.transform(test_image)
scalar.fit(X)
print(classifier.predict_proba(X))
print(classifier.layers[6].output)

#saving the file
classifier.save('CN_and_NN.h5')


#loading the saved file
from keras.models import load_model
new_model=load_model('C_and_N.h5')
new_model.summary()


#matplot

import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
