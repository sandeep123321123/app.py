import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import streamlit as st

# Data collection
data = {
    'Location': ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Kolkata', 'Pune'],
    'Size': [2000, 2500, 3000, 3500, 4000, 4500, 5000],
    'Bedrooms': [3, 4, 4, 5, 5, 6, 6],
    'Bathrooms': [2, 2, 3, 3, 4, 4, 5],
    'Price': [4000000, 5000000, 6000000, 7000000, 8000000, 9000000, 10000000]
}
df = pd.DataFrame(data)

# Location ko numerical value mein convert karna
location_map = {'Mumbai': 1, 'Delhi': 2, 'Bangalore': 3, 'Hyderabad': 4, 'Chennai': 5, 'Kolkata': 6, 'Pune': 7}
df['Location'] = df['Location'].map(location_map)

# Data preprocessing
X = df[['Location','Size','Bedrooms','Bathrooms']]
y = df['Price']

# Model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit app
st.title("HOUSE PRICE PRIDECTION APPLICATION CREATED BY SANDEEP SONAKR")

location = st.selectbox("Select Location", ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Kolkata", "Pune"])
size = st.number_input("Enter Size (in sqft)", min_value=1000, max_value=10000)
bedrooms = st.number_input("Enter Number of Bedrooms", min_value=1, max_value=10)
bathrooms = st.number_input("Enter Number of Bathrooms", min_value=1, max_value=10)

location_value = location_map[location]
new_house = pd.DataFrame({'Location': [location_value], 'Size': [size], 'Bedrooms': [bedrooms], 'Bathrooms': [bathrooms]})

if st.button("Predict Price"):
    st.success("Successfully")
    st.balloons()
    predicted_price = model.predict(new_house)
    st.write(f"Predicted Price: {predicted_price[0]}-NIR")
    
else:  
    st.write("Please enter values above, then click **Predict!**")
    st.write("Thank You  All Everyone.....")
    

