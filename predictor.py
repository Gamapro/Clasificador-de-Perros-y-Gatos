# PREDICTOR

import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
import h5py
import os
import time
import subprocess
import threading
import socket

longitud, altura = 128, 128
medidas = (longitud, altura)
modelo = './modelo/modelo.h5'
weights = './modelo/pesos.h5'

# Cargamos el modelo

cnn = tf.keras.models.load_model(modelo)
cnn.load_weights(weights)

def predict(file):
    x = load_img(file, target_size=medidas)
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    arreglo = cnn.predict(x)
    resultado = arreglo[0]
    respuesta = np.argmax(resultado)
    s=""
    if respuesta==0:
        s = "gato" 
    elif respuesta==1:
        s = "perro" 
    return s

def java():
    os.system("java -jar Seleccionador.jar")

host = "25.30.173.27"
port = 7777
port2 = 6666
print_lock = threading.Lock() 
lis = []

def enviar(pred):

    try: 
        s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
        print("Socket de envio successfully created")
    except socket.error as err: 
        print ("Socket de envio creation failed with error %s" %(err)) 
   
    s2.connect((host,port2))
    s2.send(pred.encode('UTF8'))
    s2.close()
    print("Prediccion mandada")

    return

tre = threading.Thread(target=java,name='Seleccionar')
tre.start()

"""
print("GATOS:")
print(predict("Imagenes prueba/cat1.jpg"))
print(predict("Imagenes prueba/cat2.jpg"))
print(predict("Imagenes prueba/cat3.jpg"))
print(predict("Imagenes prueba/cat4.jpg"))
print(predict("Imagenes prueba/cat5.jpg"))
print(predict("Imagenes prueba/cat6.jpg"))
print(predict("Imagenes prueba/cat7.jpg"))
print(predict("Imagenes prueba/cat8.jpg"))
print(predict("Imagenes prueba/cat9.jpg"))

print("PERROS:")
print(predict("Imagenes prueba/dog1.jpg"))
print(predict("Imagenes prueba/dog2.jpg"))
print(predict("Imagenes prueba/dog3.jpg"))
print(predict("Imagenes prueba/dog4.jpg"))
print(predict("Imagenes prueba/dog5.jpg"))
print(predict("Imagenes prueba/dog6.jpg"))
"""

try: 
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
    print("Socket successfully created")
except socket.error as err: 
    print ("socket creation failed with error %s" %(err)) 

s.bind((host,port))
print("socket binded to %s" %(port))
s.listen(5)  
print("socket is listening")

while True:  # Mientras este conectado
    c, addr = s.accept() 
    print('Got connection from', addr)
    threading.Lock().acquire() 
    data = c.recv(1024).decode("UTF8")
    print("PYTHON: " + data)
    aux = data.split(os.path.abspath(os.getcwd()))
    if aux != [""]:
        lis = aux
    print(lis)
    pred = predict(lis[1][1:len(lis[1])])
    print("Prediccion a mandar: "+pred)
    envio = threading.Thread(target=enviar,name='Enviar',args=(pred,))
    envio.start()

s.close() 
