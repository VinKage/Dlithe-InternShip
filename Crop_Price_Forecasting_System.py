import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import numpy as np
from PIL import Image
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load Image
image = Image.open("crop.jpeg")

# Define the Streamlit app
st.image(image, width=500, use_column_width=50)

st.write("""
#  Crop Price Prediction App

   This App predicts **Crop Price for a definite period!**
""")
st.write('---')

agree = st.checkbox("Ready to begin")

def Dataset_Checking():
    if agree == True:
        Dataset_Agree = st.sidebar.checkbox("**Choose Any Crop**")
        if Dataset_Agree == True:
            return True
        else:
            return False

def Avg_Price_BarChart(df):
    avg_prices = df.groupby('Month')['WPI'].mean()
    st.bar_chart(avg_prices)

# ...

def WPI_Graph(df):
    chart_option = st.radio("Select Graph Type", ["WPI Line Chart", "Average Price Bar Chart"])

    if chart_option == "WPI Line Chart":
        chart_data = df[['WPI']]
        st.line_chart(chart_data)
    elif chart_option == "Average Price Bar Chart":
        Avg_Price_BarChart(df)


def Dataset():
    Wheat_Agree = st.sidebar.checkbox("Wheat")
    Ragi_Agree = st.sidebar.checkbox("Ragi")
    Copra_Agree = st.sidebar.checkbox("Copra")
    Cotton_Agree = st.sidebar.checkbox("Cotton")
    if Wheat_Agree == True:
        st.header("Dataset of Wheat")
        df = pd.read_csv("Wheat.csv")
        st.write(df)
        wpi_agree = st.checkbox("WPI over the Total Months")
        if wpi_agree == True:
            WPI_Graph(df)
        df['Year'] = df['Year'].fillna(0).astype(np.float64)
        df['Month'] = df['Month'].fillna(0).astype(np.float64)
        return df
    elif Ragi_Agree == True:
        st.header("Dataset of Ragi")
        df = pd.read_csv("Ragi.csv")
        st.write(df)
        wpi_agree = st.checkbox("WPI over the Total Months")
        if wpi_agree == True:
            WPI_Graph(df)
        df['Year'] = df['Year'].fillna(0).astype(np.float64)
        df['Month'] = df['Month'].fillna(0).astype(np.float64)
        return df
    elif Cotton_Agree == True:
        st.header("Dataset of Cotton")
        df = pd.read_csv("Cotton.csv")
        st.write(df)
        wpi_agree = st.checkbox("WPI over the Total Months")
        if wpi_agree == True:
            WPI_Graph(df)
        df['Year'] = df['Year'].fillna(0).astype(np.float64)
        df['Month'] = df['Month'].fillna(0).astype(np.float64)
        return df
    elif Copra_Agree == True:
        st.header("Dataset of Copra")
        df = pd.read_csv("Copra.csv")
        st.write(df)
        wpi_agree = st.checkbox("WPI over the Total Months")
        if wpi_agree == True:
            WPI_Graph(df)
        df['Year'] = df['Year'].fillna(0).astype(np.float64)
        df['Month'] = df['Month'].fillna(0).astype(np.float64)
        return df

def ChooseModel():
    ML_Agree = st.sidebar.checkbox(""" **Please choose your ML Algorithm** """)
    if ML_Agree == True:
        RF_Agree = st.sidebar.checkbox("Random Forest Regressor-->(User's Best Choice)")
        GB_Agree = st.sidebar.checkbox("Gradient Boosting Regressor")  # Add this option
        if RF_Agree == True:
            model = RandomForestRegressor(n_estimators=500)
        elif GB_Agree == True:
            model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.1)  # Adjust hyperparameters as needed
        return model

def suggestMonth(month):
    if(month == 1):
        return 17.56
    elif(month == 2):
        return 25.6
    elif(month == 3):
        return 32.71
    elif(month == 4):
        return 40.71
    elif(month == 5):
        return 57.7
    elif(month == 6):
        return 161.02
    elif(month == 7):
        return 275.5
    elif(month == 8):
        return 246.65
    elif(month == 9):
        return 168.75
    elif(month == 10):
        return 74.4
    elif(month == 11):
        return 26.6
    elif(month == 12):
        return 12.58

if __name__ == "__main__":
    Flag = Dataset_Checking()
    if Flag == True:
        df = Dataset()
        st.sidebar.write("")
        st.sidebar.write("")
        model = ChooseModel()
        pred_agree = st.checkbox("Get Your Prediction Started")
        if pred_agree == True:
            x = df[['Month','Year','Rainfall']]
            y = df.WPI
            model.fit(x, y)
            st.markdown("***")
            st.write("**Please Specify Input Parameters**")
            Year = st.number_input('Enter the Year')
            Month = st.number_input("Enter the Month")
            suggestion = suggestMonth(Month)
            st.write("The Average Rainfall of your month is ")
            st.write(suggestion)
            Rainfall = st.number_input("Enter the Rainfall")
            variety = st.text_input("Enter the Variety")
            Year = float(Year)
            Month = float(Month)
            Y_test = [[Month, Year, Rainfall]]
            if Year and Month and Rainfall and variety:
                Y_test = [[Month, Year, Rainfall]]
                Result = model.predict(Y_test)
                st.write("**Predicted Values is :**", Result)
            else:
                st.write("Please fill in all input fields.")
