# CS 3210 - Machine Learning - Project
# Victoria Lassner

import streamlit as st
import pandas as pd
import numpy as np
from transformers import pipeline
from PIL import Image

# Description of the model
st.title("ViT Model")
st.caption("(You can test the model below.)")
st.text("The ViT model uses the transformer architecture. This is how it learns by following patterns in sequential "
        "data with its self-attention mechanism. The mechanism is how it takes into account the importance of each "
        "input element while making predictions. One of the drawbacks to this is that it lacks the built-in inductive "
        "bias of locality which is where the model assumes pixels nearby have a high chance of being related, that "
        "CNN's like ResNet have. This makes it computationally expensive for larger/high resolution images. The ViT "
        "model does require large amounts of data in order to make good predictions. However, they do have any "
        "advantages like being able to handle different types of input better than CNN like images of different sizes "
        "or finding long-range dependencies which is how it traces the relationship between elements among the many layers.")

st.image("vision-transformer-vit.png")
st.caption("Source: Dosovitskiy, Alexey, et al. “An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale.” arXiv.Org, 3 June 2021, arxiv.org/abs/2010.11929.")

st.text("The ViT model is made up of five different layers: patch embedding, multi-head self-attention, the layer "
        "normalization, multi-layer perception and the classification head. The first layer, patch embedding, breaks "
        "up the image into different sections and converts it into a vector representation like tokens. The next three "
        "layers are part of the transformer encoder blocks. The multi-head self-attention layer calculates weights by "
        "comparing the different sections/patches, so that the model can focus on certain features. The next layer, the "
        "layer normalization, processes the numeric values from the MHSA layer into a common scale which is where the "
        "features will be further refined in the multi-layer perception layer. The final layer, classification head, will "
        "take all of the features that have been extracted to make a prediction using the SoftMax activation function.")


st.text("This model is created by Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, "
        "Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil "
        "Houlsby and trained by Dmytro Iakubovskyi on HuggingFace using the dataset Fer-2013 from Kaggle.")


# Load the model
pipe = pipeline("image-classification", model="dima806/facial_emotions_image_detection")

st.title("Upload Image(s) to Test the Model.")
st.text("You can upload one or multiple images at a time.")

# User uploads photo(s)
uploaded_files = st.file_uploader("Choose image(s) from folder...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

# Store prediction results
predictions = []
confidences = []

if uploaded_files is not None:
    # determine the number of columns based on the number of images uploaded
    num_files = len(uploaded_files)
    # Display max of 3 images side by side
    # use the number of images to calculate the number of rows
    columns_per_row = 3
    #Calculate the number of rows needed
    rows = (num_files + columns_per_row - 1) // columns_per_row 

    # Loop through each row
    for row in range(rows):
        # Create columns for this row
        columns = st.columns(columns_per_row)

        # Loop through the images in this row
        for col in range(columns_per_row):
            idx = row * columns_per_row + col
            
            #check if there is an image to display in this column
            if idx < num_files:  
                uploaded_photo = uploaded_files[idx]
                input_image = Image.open(uploaded_photo)

                # Use the pipeline for predictions
                prediction = pipe(input_image)

                # Find top 3 predictions and confidence scores
                predictions = sorted(prediction, key=lambda x: x['score'], reverse=True)[:3]

                # Display the image and predictions in the corresponding column
                with columns[col]:
                    # Display the image
                    st.image(input_image, caption=f"Uploaded Image: {uploaded_photo.name}", width=200)

                    # Display the top predictions
                    st.write("Top 3 Predictions:")
                    for i, pred in enumerate(predictions):
                        predicted_label = pred['label']
                        confidence_score = pred['score'] * 100
                        st.write(f"{i+1}. {predicted_label}: {confidence_score:.2f}%")

        # Add a separator
        st.write("---")
