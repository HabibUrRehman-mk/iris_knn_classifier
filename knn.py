import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split



@st.cache_resource
def load_and_train_model():
    df = sns.load_dataset('iris')
    x = df.drop('species', axis=1)
    y = df['species']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train, y_train)
    return knn

knn = load_and_train_model()




st.title('Iris flower Classifier using KNN')
st.write("Enter the flower's measurement to predict the flower specie")

# st.sidebar.title("About")
# st.sidebar.write(
#     "This is an Iris flower specie predictor created by Habib Ur Rehman "
#     "using KNN and Streamlit."
# )
# st.sidebar.write(
#     "If you have any feedback, feel free to contact: "
#     "📧 mail.habiburrehman@gmail.com"
# )
s_l=st.number_input("Sepal lenght (cm)",min_value=0.0,format='%.2f')
s_w=st.number_input("Sepal width (cm)",min_value=0.0,format='%.2f')
p_l=st.number_input("Petal lenght (cm)",min_value=0.0,format='%.2f')
p_w=st.number_input("Petal width (cm)",min_value=0.0,format='%.2f')

y_pred=knn.predict([[s_l,s_w,p_l,p_w]])
if st.button("Predict"):
    if s_l == 0 and s_w == 0 and p_l == 0 and p_w == 0:
        st.error("Enter the measurements to predict")
    else:
        st.success(f"The species is {y_pred[0]}")

st.write(
    "If you have any feedback, feel free to contact: "
    "📧 mail.habiburrehman@gmail.com"
)







