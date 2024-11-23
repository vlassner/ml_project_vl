#CS 3210 - Machine Learning - Project
#Victoria Lassner

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model from kaggle
model = tf.keras.models.load_model('resnet50_fer2013.h5')

# Function to convertthe image into the correct format
def preprocess_image(image):
    # Resize image
    image = image.resize((224, 224))
    # Convert the image to an array
    img_array = np.array(image)
    # Normalize the image
    img_array = img_array / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Description of the model
st.title("ResNet Model")
st.text("ResNet (Residual Network) is a deep learning architecture for neural networks that's commonly used for computer"
        " vision applications like image segmentation and object detection.")
st.image("resnet_image.png")
st.caption("Source: Nayan Chaure, 2024, Medium")

st.text("This model was created by Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun and trained by Joyeeta Dey with the FER-2013 dataset")

st.write("Upload an Image to Test the Model.")

# User uploads image
uploaded_image = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert image to correct format for model
    img_array = preprocess_image(image)

    # Get model prediction
    prediction = model.predict(img_array)

    # Emotion labels
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    predicted_emotion = emotion_labels[np.argmax(prediction)]

    # Display the prediction
    st.write(f"Prediction: {prediction[0]['label']} with a confidence of {prediction[0]['score']*100:.2f}%")

# Add ability to enter a folder of photos