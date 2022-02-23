## pipenv --three
## pipenv shell
## pip install streamlit keras
## python3 -m pip install tensorflow-macos

import io

import numpy as np
import plost
import requests
import streamlit as st
from PIL import Image

st.title("📷  Computer vision app")

with st.sidebar:
    st.title("Upload an image")
    upload_method = st.radio(
        label="Choose how to upload an image",
        options=("Webcam", "URL", "File upload"),
        index=1,
    )

    st.write(
        f"""You have chosen to upload using {upload_method.lower()}!  
        Let's do it :rocket:"""
    )

    camera_input = None
    url = None
    file = None

    if upload_method == "Webcam":
        camera_input = st.camera_input(label="Take a picture")

    elif upload_method == "URL":
        url = st.text_input(
            label="URL of the image",
            value="https://user-images.githubusercontent.com/63207451/141209252-a98cc392-8831-4fbe-af90-61cb7eee8264.png",
        )

    else:
        file = st.file_uploader("Upload the image file", type=(".png", ".img"))

st.write("## Uploaded image")

img_bytes = None

if url:
    img_bytes = requests.get(url).content

if file:
    img_bytes = file.getvalue()

if camera_input:
    img_bytes = camera_input.getvalue()

if img_bytes:
    st.write("Here's the image you uploaded:")
    st.image(img_bytes, width=200)
    img_bytes_io = io.BytesIO(img_bytes)
    img_array = np.array(Image.open(img_bytes_io))
else:
    st.warning("👈 Please upload an image!")
    st.stop()

# st.write("## Pre-processing")

# to_grayscale = st.checkbox("Set to grayscale")
# if to_grayscale:
#     st.write(img_array.shape)
#     st.write(img_array[0][1])
#     img_array[:, :, 1] = img_array[:, :, 0]
#     img_array[:, :, 2] = img_array[:, :, 0]
#     st.image(Image.fromarray(img_array))


st.write("## Prediction")

import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.mobilenet import (
    decode_predictions as mobilenet_decode_predictions,
)
from tensorflow.keras.applications.mobilenet import (
    preprocess_input as mobilenet_preprocess_input,
)
from tensorflow.keras.applications.vgg16 import (
    decode_predictions as vgg16_decode_predictions,
)
from tensorflow.keras.applications.vgg16 import (
    preprocess_input as vgg16_preprocess_input,
)
from tensorflow.keras.preprocessing import image


@st.experimental_singleton
def load_vgg16():
    model = tf.keras.applications.VGG16(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
    )

    return model


@st.experimental_singleton
def load_mobilenet():
    model = tf.keras.applications.MobileNet(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
    )

    return model


model_name = st.selectbox("Choose model", options=("VGG-16", "MobileNet"))
if model_name == "VGG-16":
    load_model = load_vgg16
    decode_predictions = vgg16_decode_predictions
    preprocess_input = vgg16_preprocess_input
if model_name == "MobileNet":
    load_model = load_mobilenet
    decode_predictions = mobilenet_decode_predictions
    preprocess_input = mobilenet_preprocess_input

model = load_model()
IMAGENET_INPUT_SIZE = (224, 224)
IMAGENET_INPUT_SHAPE = [224, 224, 3]


@st.experimental_memo
def pre_process_img(img_array):
    img = Image.fromarray(img_array)
    img = img.convert("RGB")
    img = img.resize(IMAGENET_INPUT_SIZE, Image.NEAREST)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    img = img.reshape(*([1] + IMAGENET_INPUT_SHAPE))
    return img


st.write(f"Here are the predictions for **{model_name}**!")
pre_processed_img = pre_process_img(img_array=img_array)
prediction = model.predict(pre_processed_img)
# n_rows = st.number_input("Display top-n rows", 5, 200)
n_rows = 5
decoded_prediction = pd.DataFrame(
    decode_predictions(prediction, n_rows)[0],
    columns=["label_id", "label", "probability"],
).sort_values(by="probability", ascending=False)

left, right = st.columns(2)
with left:
    st.dataframe(decoded_prediction)
with right:
    plost.bar_chart(
        data=decoded_prediction,
        bar="label",
        value="probability",
        direction="horizontal",
    )
