import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
import os

# Check if data file exists
DATA_FILE = "perrin-freres-monthly-champagne-.csv"
if not os.path.exists(DATA_FILE):
    st.error(f"Data file '{DATA_FILE}' not found in the current directory.")
    st.stop()

# Load the saved model
@st.cache_resource
def load_model():
    try:
        with open('arima_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("ARIMA model file 'arima_model.pkl' not found.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

model = load_model()

# Streamlit UI
st.title('Sales Forecasting App')

# Load and prepare data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_FILE)
        df.columns = ['Month', 'Sales']
        df.drop([105, 106], axis=0, inplace=True)
        df['Month'] = pd.to_datetime(df['Month'])
        df.set_index('Month', inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

df = load_data()

# Display historical data with more options
st.subheader('Historical Sales Data')

# Add date range selector
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start date", df.index.min())
with col2:
    end_date = st.date_input("End date", df.index.max())

# Filter data based on selection
filtered_df = df[(df.index >= pd.to_datetime(start_date)) & 
                (df.index <= pd.to_datetime(end_date))]

# Plot with more options
plot_type = st.radio("Plot type", ["Line", "Area"], horizontal=True)
if plot_type == "Line":
    st.line_chart(filtered_df)
else:
    st.area_chart(filtered_df)

# Show raw data if requested
if st.checkbox("Show raw data"):
    st.dataframe(filtered_df.style.format({"Sales": "{:.2f}"}))

# Forecast settings
st.subheader('Forecast Settings')
months = st.slider('Select number of months to forecast:', 1, 24, 12)
include_ci = st.checkbox("Include confidence intervals", value=True)

# Generate forecast
if st.button('Generate Forecast'):
    with st.spinner('Generating forecast...'):
        try:
            forecast = model.get_forecast(steps=months)
            forecast_index = pd.date_range(df.index[-1], periods=months+1, freq='MS')[1:]
            forecast_df = pd.DataFrame({
                'Forecast': forecast.predicted_mean,
                'Lower CI': forecast.conf_int()['lower Sales'],
                'Upper CI': forecast.conf_int()['upper Sales']
            }, index=forecast_index)

            # Plot forecast
            st.subheader('Sales Forecast')
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(filtered_df['Sales'], label='Historical Sales')
            ax.plot(forecast_df['Forecast'], label='Forecast', color='red')
            if include_ci:
                ax.fill_between(forecast_df.index, 
                                forecast_df['Lower CI'], 
                                forecast_df['Upper CI'], 
                                color='pink', alpha=0.3, label='95% Confidence Interval')
            ax.set_title('Sales Forecast with ARIMA')
            ax.set_xlabel('Date')
            ax.set_ylabel('Sales')
            ax.legend()
            st.pyplot(fig)
            
            # Show forecast values
            st.subheader('Forecast Values')
            st.dataframe(forecast_df.style.format("{:.2f}"))
            
            # Download forecast
            csv = forecast_df.to_csv().encode('utf-8')
            st.download_button(
                label="Download Forecast as CSV",
                data=csv,
                file_name='sales_forecast.csv',
                mime='text/csv'
            )
            
        except Exception as e:
            st.error(f"Error generating forecast: {str(e)}")

# Model evaluation section
st.subheader('Model Performance Metrics')
if st.button('Show Evaluation Metrics'):
    with st.spinner('Evaluating model...'):
        try:
            # For demo, use last 20% as test set
            test_size = int(len(df) * 0.2)
            train_data = df.iloc[:-test_size]
            test_data = df.iloc[-test_size:]
            
            # Fit model on training portion
            eval_model = ARIMA(train_data['Sales'], order=(1,1,1), seasonal_order=(1,1,1,12)).fit()
            forecast_test = eval_model.get_forecast(steps=len(test_data))
            
            # Calculate metrics
            def calculate_metrics(actual, predicted):
                mae = mean_absolute_error(actual, predicted)
                mse = mean_squared_error(actual, predicted)
                rmse = sqrt(mse)
                mape = np.mean(np.abs((actual - predicted) / actual)) * 100
                return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape}

            metrics = calculate_metrics(test_data['Sales'], forecast_test.predicted_mean)
            
            # Display metrics in columns
            col1, col2 = st.columns(2)
            with col1:
                st.metric("MAE (Mean Absolute Error)", f"{metrics['MAE']:.2f}")
                st.metric("MSE (Mean Squared Error)", f"{metrics['MSE']:.2f}")
            with col2:
                st.metric("RMSE (Root Mean Squared Error)", f"{metrics['RMSE']:.2f}")
                st.metric("MAPE (Mean Absolute Percentage Error)", f"{metrics['MAPE']:.2f}%")
            
            # Plot evaluation
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(train_data.index, train_data['Sales'], label='Training Data')
            ax.plot(test_data.index, test_data['Sales'], label='Actual Test Data', color='blue')
            ax.plot(test_data.index, forecast_test.predicted_mean, label='Predictions', color='red')
            ax.set_title('Model Evaluation on Test Data')
            ax.legend()
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error evaluating model: {str(e)}")

# Model information in sidebar
st.sidebar.subheader('Model Information')
st.sidebar.write("This app uses a Seasonal ARIMA model for sales forecasting.")
st.sidebar.write("**Model Parameters:** (1,1,1)(1,1,1,12)")
st.sidebar.write("**Dataset:** Monthly champagne sales data")

# Add app information in sidebar
st.sidebar.subheader('App Information')
st.sidebar.write("""
- **Purpose:** Sales forecasting using time series analysis
- **Data Source:** Local CSV file
- **Features:** Historical data visualization, forecasting, model evaluation
""")

# Add a footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Note:** Forecasts are based on historical patterns and may not account for unexpected events.
""")