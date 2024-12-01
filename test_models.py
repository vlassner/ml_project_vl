# CS 3210 - Machine Learning - Project
# Victoria Lassner

import streamlit as st
from transformers import pipeline
from PIL import Image

st.title("Test Both Models")

st.text("This is where you will be able to test both models against each other.")
st.image("vit_res.png")
st.caption("Nikolas Adaloglou, Tim Kaiseron, 2023, 'Understanding Vision Transformers (ViTs): Hidden properties, insights, and robustness of their representations'")

# Initialize the models
pipe_resnet = pipeline("image-classification", model="Celal11/resnet-50-finetuned-FER2013-0.001")
pipe_vit = pipeline("image-classification", model="dima806/facial_emotions_image_detection")

st.title("Upload Image(s) to Test the Models.")
st.text("You can upload one image or multiple at the time.")

# User uploads one or more images
uploaded_files = st.file_uploader("Choose image(s) from folder...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files is not None:
    # Loop through each uploaded file
    for uploaded_photo in uploaded_files:
        # Open the uploaded image
        input_image = Image.open(uploaded_photo)
        
        # Display the original image for user to see
        st.image(input_image, caption=f"Uploaded Image: {uploaded_photo.name}", width=300)

        # Predictions from ResNet model
        prediction_resnet = pipe_resnet(input_image)
        predictions_resnet = sorted(prediction_resnet, key=lambda x: x['score'], reverse=True)[:3]

        # Predictions from ViT model
        prediction_vit = pipe_vit(input_image)
        predictions_vit = sorted(prediction_vit, key=lambda x: x['score'], reverse=True)[:3]

        # Create columns for side-by-side display of predictions
        col1, col2 = st.columns(2)

        # Display the results for ResNet model in the first column
        with col1:
            st.write("**ResNet Model Predictions**:")
            for i, pred in enumerate(predictions_resnet):
                predicted_label = pred['label']
                confidence_score = pred['score'] * 100
                st.write(f"{i+1}. {predicted_label}: {confidence_score:.2f}%")

        # Display the results for ViT model in the second column
        with col2:
            st.write("**ViT Model Predictions**:")
            for i, pred in enumerate(predictions_vit):
                predicted_label = pred['label']
                confidence_score = pred['score'] * 100
                st.write(f"{i+1}. {predicted_label}: {confidence_score:.2f}%")

        # Add a separator
        st.write("---")
