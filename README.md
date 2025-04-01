# Demand-Forecasting
# Sales Forecasting App

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![ARIMA](https://img.shields.io/badge/ARIMA-009688?style=for-the-badge&logo=arima&logoColor=white)

A Streamlit web application for forecasting sales using Seasonal ARIMA time series modeling.

## 🚀 Features

- 📈 Visualize historical sales data with interactive charts
- 🔮 Generate sales forecasts for 1-24 months ahead
- 📊 View model performance metrics (MAE, MSE, RMSE, MAPE)
- 📥 Download forecast results as CSV
- ⚙️ Customizable forecast settings
- 📱 Responsive web interface

## 📌 Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.7+
- pip package manager
- Required Python packages (listed in `requirements.txt`)

## 🔧 Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/sales-forecasting-app.git
   cd sales-forecasting-app
   ```

2. Create and activate a virtual environment (recommended):
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

## 📊 Data Preparation

Place your sales data file (`perrin-freres-monthly-champagne-.csv`) in the project root directory.

## 🚀 Usage

Run the Streamlit app:
   ```sh
   streamlit run app.py
   ```

The app will automatically open in your default browser at [http://localhost:8501](http://localhost:8501)

### 🎯 Use the interface to:
- View historical sales data
- Adjust date ranges
- Select forecast period
- Generate and download forecasts
- Evaluate model performance

## 📂 File Structure
```
 sales-forecasting-app/
 ├── app.py                # Main application code
 ├── arima_model.pkl       # Pre-trained ARIMA model
 ├── perrin-freres-monthly-champagne-.csv  # Sample dataset
 ├── requirements.txt      # Python dependencies
 └── README.md             # This file
```

## 🏆 Model Details
The app uses a **Seasonal ARIMA** model with the following parameters:

- **Order:** (1,1,1)
- **Seasonal Order:** (1,1,1,12)

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the project
2. Create your feature branch:
   ```sh
   git checkout -b feature/AmazingFeature
   ```
3. Commit your changes:
   ```sh
   git commit -m 'Add some amazing feature'
   ```
4. Push to the branch:
   ```sh
   git push origin feature/AmazingFeature
   ```
5. Open a Pull Request

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for more information.

## Screenshots:

![image](https://github.com/user-attachments/assets/4d526815-e262-48a4-9f2f-83ce1fbfbf85)

![image](https://github.com/user-attachments/assets/44352e43-2b08-41af-a2f8-9161cdd44a6e)

