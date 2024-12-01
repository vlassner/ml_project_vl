CS 3210 - Machine Learning - Project
Victoria Lassner

Option A: Build a Web-App for a Machine Learning Model

Goal:

    Create an interactive web application that allows users to interact with a machine learning model

Deliverables:

    Web Application: A working web application that allows a user to interact with a trained machine learning model (e.g., make predictions, visualize outputs)
    Code Files: Separate files for the user interface and the machine learning model, with clear documentation on how each part works
    Documentation: A README file with instructions on how to run the application, and a summary of the model used and its limitations

Grading Criteria:

    (20 points) Web-App Functionality and User Interface:
        Web-app is functional and allows users to interact with the model without errors
        Clear and intuitive user interface, including meaningful visual feedback (e.g., charts or graphs)
    (5 points) Model Choice and Performance:
        Model is appropriately chosen for the problem and produces reasonable predictions
        Evidence of proper evaluation metrics to assess model performance
    (10 points) Organization and Documentation:
        Code is well-organized, modular, and includes appropriate comments
        README includes clear setup instructions, with a concise explanation of the model, data, and limitations


To setup this web app:

1. Run pip install -r requirements.txt
2. Run streamlit run app.py

Description:

There are two different models in this web app that the user is able to interact with. The user has the ability
to upload one or more photos in which the model will predict the emotion of the person in the photo(s). The 
model with display the confidence of the top three emotions for each photo. The two models, ResNet and ViT are 
pre-trained on the dataset Fer-2013 due to the computational demand of training the models personally.

The dataset, Fer-2013, is a set of 48x48 pixel grayscale images of faces. There are seven emotions, anger, disgust,
fear, happy, neutral, sad and surprise. This limits the number of emotions the models are trained to recongized, so
they won't be 100% accurate if it a different emotion from these.

The models are trained to recongize human emotions, so avoid using photos of random places or objects as the results
will be not be accurate and photos with muliptle people. The models are designed to output the top three emotions per
photo. The images can be in color or not and any size. The models will convert them into the correct format. The 
images must be in the format jpg, png, jpeg for the model to be able to make a prediction.