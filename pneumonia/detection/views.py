from email.policy import default
from importlib import import_module
from msilib.schema import File
from django.shortcuts import render
from django.core.files.storage import default_storage
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
import numpy as np
import cv2
from PIL import Image
# Create your views here.

model = load_model('model/Pneumonia_model_vgg19.h5')


def detection(request):
    label = " "
    return render(request, 'index.html', {'msg': label})


def imgprocess(request):

    if request.method == 'POST':
        file = request.FILES["pic"]
        file_name = default_storage.save(file.name, file)
        file_url = default_storage.path(file_name)
        if(file_name.lower().endswith(('.png', '.jpg', '.jpeg'))):
            label = process(file_name)
        else:
            label = "enter proper format file"
    return render(request, 'index.html', {'msg': label})


def process(fn):
    path = "media/"+fn
    img = image.load_img(path, target_size=(224, 224))
    y = image.img_to_array(img)
    y = np.expand_dims(y, axis=0)
    img_data = preprocess_input(y)
    out = model.predict(img_data)
    if(out[0][0] > 0.0):
        label = "YOU ARE NOT AFFECTED BY PNEUMONIA"
    else:
        label = "YOU ARE  AFFECTED BY PNEUMONIA"
    return label
