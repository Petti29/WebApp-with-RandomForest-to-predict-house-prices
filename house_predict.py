#Importare il dataset
import pandas as pd
df = pd.read_excel('../data/Real estate valuation data set.xlsx')
print(df.dtypes, '\n\n\n', df, '\n\n\n', df.isnull().sum())

#Librerie per il lavoro
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
#Creare i subset
y = df['Y house price of unit area']
X = df[['X2 house age','X3 distance to the nearest MRT station','X4 number of convenience stores','X5 latitude','X6 longitude']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#Fittare il modello
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
#Valutare il modello
mse = mean_squared_error(y_test, y_pred)
print('MSE =', mse)
print('Il prezzo medio osservato è:', y_test.mean())

#Interfaccia utente
import streamlit as st
#Titolo
st.title("House price of unit area prediction")
# Text input
user_lat = st.text_area("Enter the latitude:", "")
user_long = st.text_area("Enter the longitude:", "")
user_age = st.text_area("Enter the house age:", "")
user_mrt = st.text_area("Enter the distance to the nearest MRT station:", "")
user_stores = st.text_area("Enter the number of convenience stores:", "")
#Ottenere la previsione
if st.button("Calculate"):
        if user_lat.strip() == "" or user_long.strip() == "" or user_age.strip() == "" or user_mrt.strip() == "" or user_stores.strip() == "":
            st.warning("Please enter some text")
        elif (float(user_lat)) < -180 or (float(user_lat) > 180):
             st.warning("Latitude value is out of range")
        elif (float(user_long) < -180) or (float(user_long) > 180):
             st.warning("Longitude value is out of range")
        else:
            #Mettere insieme gli input dell'utente
            X = np.array([[float(user_lat), float(user_long), float(user_age), float(user_mrt), float(user_stores)]])
            prediction = rf.predict(X)
            #Visualizzare l'output
            st.success(f'Prediction: {round(prediction[0],2)}   |   Note that mse is equal to {round(mse,2)}')
