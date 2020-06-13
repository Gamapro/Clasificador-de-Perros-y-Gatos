import sys
import os
import numpy as np
import tensorflow
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers import Convolution2D, MaxPooling2D, Conv2D
from keras import backend as K
from keras.callbacks import EarlyStopping, TensorBoard 
from keras.optimizers import Adam
from time import time
import matplotlib.pyplot as plt

K.clear_session()
data_entrenamiento = "./data/entrenamiento"
data_validacion = "./data/validacion"

# Parametros

epocas = 20
altura, longitud = 100,100
batch_size = 32
batch_size_val = 16
pasos = 500
pasos_validacion = 100
filtrosConv1 = 8
filtrosConv2 = 16
filtrosConv3 = 32
tamano_filtro1 = (8,8)
tamano_filtro2 = (4,4)
tamano_filtro3 = (2,2)
pool = (2,2)
clases = 2
lr = 0.001

# Preprocesamiento

entrenamiento_datagen = ImageDataGenerator(
    rescale = 1./255,  # Cada pixel este entre 0 y 1
    shear_range = 0.2, # Generar las imagenes "inclinadas"
    #zoom_range = 0.2,  # Zoomearlas parahacer "enfasis"
    horizontal_flip = True   # Invertir para distinguir direccionalidad
)

validacion_datagen = ImageDataGenerator(
    rescale = 1./255  # Solo reescalado para las imagenes de test        
)

imagen_entrenamiento = entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    target_size = (altura, longitud),
    #classes = ['perro','gato'],
    batch_size = batch_size,
    class_mode = 'categorical'
)

imagen_validacion = validacion_datagen.flow_from_directory(
    data_validacion,
    target_size = (altura, longitud),
    #classes = ['perro','gato'],
    batch_size = batch_size_val,
    class_mode = 'categorical'
)

#print(imagen_entrenamiento.class_indices)
def plots(ims, figsize=(20,20), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if(ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims)%2==0 else len(ims)//rows+1
    for i in range(len(ims)):
        sp = f.add_subplot(rows,cols,i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i],interpolation=None if interp else "none")    
    plt.show()
imgs, labels = next(imagen_entrenamiento)
#plots(imgs,titles=labels)


# Crear la Red Neuronal Convolucional

cnn = Sequential()

# Primera capa
cnn.add(Conv2D(filtrosConv1, tamano_filtro1, padding='same', input_shape=(altura,longitud,3), activation = 'relu'))
cnn.add(MaxPooling2D(pool_size=pool))

# Segunda Capa
cnn.add(Conv2D(filtrosConv2, tamano_filtro2, padding='same', activation = 'relu'))
cnn.add(MaxPooling2D(pool_size=pool))

# Tercera Capa
cnn.add(Conv2D(filtrosConv3, tamano_filtro3, padding='same', activation = 'relu'))
cnn.add(MaxPooling2D(pool_size=pool))

# Volverla plana, Cuarta y Quinta capa
cnn.add(Flatten())
cnn.add(Dense(4,activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(clases, activation='softmax'))

log_dir= "logs/{}".format(time())
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=0)

cnn.compile(
    loss='categorical_crossentropy', 
    optimizer=Adam(lr=lr),
    metrics=['accuracy']
)

cnn.fit_generator(
    imagen_entrenamiento,
    steps_per_epoch=pasos,
    epochs=epocas, 
    validation_data=imagen_validacion, 
    validation_steps=pasos_validacion,
    callbacks=[tensorboard_callback]
)

# Guardar modelo
dir = './modelo/'
if not os.path.exists(dir):
    os.mkdir(dir)
cnn.save('./modelo/modelo_prueba.h5')
cnn.save_weights('./modelo/pesos_prueba.h5')


print("Proceso finalizado!")
