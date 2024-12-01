# CS 3210 - Machine Learning - Project
# Victoria Lassner

import streamlit as st
import pandas as pd
import numpy as np
from transformers import pipeline
from PIL import Image

# Description of the model
st.title("ResNet Model")
st.caption("(You can test the model below.)")
st.text("The ResNet model, Residual Network, is a Convolutional Neural Network (CNN). This version is"
        " ResNet50 which uses 50 layers to process the images. ResNet is unique in that it can skip "
        "layers and go deeper into the network which allows it to learn better.")

st.image("resnet_image.png")
st.caption("Source: Nayan Chaure, 2024, Medium")

st.text("There are four main sections of this model: convolutional filters, residual blocks, bottleneck"
        " blocks, and the end layer. The first section, the convolutional filters, looks for features "
        "like colors and edges to focus on. The residual blocks are vital in preventing the accuracy from"
         " plateauing after a certain number of layers has been reached by. It does this by connecting the"
          " input of the image itself or the feature to the output which allows it to retain information "
          "learned previously better. The bottleneck blocks reduce the number of features that they then "
          "process in a convolution layer. It will then bring those features back after.  In the end, the "
          "image is converted to a vector numbers where it will make its prediction.")

st.text("This model was created by Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun and trained by "
        "Celal11 with the FER-2013 dataset")

# Initialize the model
pipe = pipeline("image-classification", model="Celal11/resnet-50-finetuned-FER2013-0.001")

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