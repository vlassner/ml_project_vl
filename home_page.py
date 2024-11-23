#CS 3210 - Machine Learning - Project
#Victoria Lassner

import streamlit as st
from PIL import Image


st.title("Human Emotion Recognition")

st.text("Neural networks are increasingly being employed to detect human emotions by analyzing various forms of data,"  
        " such as facial expressions, voice tone, and physiological signals. These networks are trained on large datasets "  
        "containing labeled examples of emotional states, enabling them to recognize subtle patterns that may be imperceptible"  
        " to the human eye or ear. For instance, convolutional neural networks (CNNs) can process images to detect changes in " 
        "facial expressions, while recurrent neural networks (RNNs) or transformers can analyze speech patterns and the rhythm of"  
        "voice to infer emotional states like happiness, anger, or sadness.")

st.text("This technology has broad applications, ranging from improving customer service experiences through sentiment analysis to" 
        " enhancing human-robot interactions and even diagnosing mental health conditions. However, challenges remain in ensuring the" 
        " accuracy and fairness of emotion detection, particularly in capturing the nuanced and culturally varied expressions of emotion"  
        " across different individuals.")

st.image("neural_net.jpg")

st.write("This web application will allow you to test the accuracy of different neural networks against a user inputed photo/folder of photos.")


