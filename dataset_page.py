#CS 3210 - Machine Learning - Project
#Victoria Lassner

import streamlit as st

st.title("Dataset")

st.write("The data consists of 48x48 pixel grayscale images of faces. The faces have been automatically "\
         "registered so that the face is more or less centred and occupies about the same amount of space"\
        " in each image.The task is to categorize each face based on the emotion shown in the facial expression"\
        " into one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral). The "\
        "training set consists of 28,709 examples and the public test set consists of 3,589 examples.")

st.image("dataset_pic.png")

st.caption("Source: Manas Sambare, 'Fer-2013', Kaggle.com")