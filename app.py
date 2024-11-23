#CS 3210 - Machine Learning - Project
#Victoria Lassner

import streamlit as st
#pip install -r requirements.txt
#streamlit run app.py

pg = st.navigation([st.Page("home_page.py", title="Home"),st.Page("dataset_page.py", title="Dataset"), st.Page("vit_model_page.py",title="VIT Model"), st.Page("resnet_model.py",title="Resnet Model"), st.Page("geinitz_model.py", title="Geinitz Model")])
pg.run()