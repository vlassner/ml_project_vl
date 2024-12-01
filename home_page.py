#CS 3210 - Machine Learning - Project
#Victoria Lassner

import streamlit as st
from PIL import Image


st.title("Human Emotion Recognition")

st.text("Neural networks are a form of machine learning that is somewhat similar to the human mind due to the various "
        "layers that communicate with each other. Neural networks can be computational demanding and take a long time to "
        "train. However, they are able to handle large datasets, non-linear relationships, and uncover hidden patterns "
        "other models would not be able to. They are able to handle complex unstructured data like images, speech and text "
        "where other models like logistic regression would fail.")

st.image("neural_net.jpg")
st.caption("Source: Jamilu Auwalu Adamu, 2019, 'Superintelligent Deep Learning Artificial Neural Networks'")

st.text("A neural network consists of a number of layers. The first layer, the input layer, takes in the raw data from the user."
        " The middle layers or also known as the hidden layers perform the calculations. This includes breaking down the data like "
        "into simple vectors and assigning weights to nodes. The data is then multiplied by this weight "
        "and if it meets a certain threshold it will continue to a new layer which is done using common activation functions "
        "like Sigmoid, ReLU, and Tanh. This allows the model to pick out certain features to make its prediction. Neural "
        "networks are trained using algorithms like backpropagation and gradient descent where the network will make a "
        "prediction and the difference between the predicted and actual outputs is computed. It is then fed back into the "
        "neural network to adjust the weights properly.")


st.text("The goal of this web application is to allow the user to test the accuracy of different neural networks against user "
        "inputed photo(s). You will be able to learn about and use two of the most well known neural network models: ViT and "
        "ResNet models to better understand neural networks and how they function. This model is limited to humans only and "
        "uses only seven emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral. Please avoid using images of places "
        "or objects as well as images that are not jpg, png, jpeg.")

st.text("This web app was created by Victoria Lassner")

