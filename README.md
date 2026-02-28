# FreshTrack Analyzer 🥬🚚

**Tackling supply chain food waste with Machine Learning.**

## The Problem
Roughly 30% of all food produced globally is wasted, and a huge chunk of that happens during transit because of "cold chain" failures (refrigeration breaking down, delays, etc.). Usually, logistics companies only find out the food is bad *after* they unload it, wasting time and money.

## The Solution
FreshTrack is a lightweight dashboard that uses route transit times and IoT sensor data (temperature & humidity) to predict if a shipment is at risk of spoiling before it even reaches the shelf. 

## The Dataset
We used the **Cold Supply Chain Data** by Syed Ali Haider Naqvi from Kaggle!
This is a massive dataset containing real IoT sensor readings for cold-chain monitoring.
**Dimensions**: Over 40,000+ real sensor readings!

## How we built it
We used **Python** and **Streamlit** for the frontend, and **Scikit-Learn** for the backend ML.

## 🌟 Key Features
- **Live Spoilage Prediction**: Dispatchers can input route sensor averages, and our AI predicts if the shipment will survive the journey with real-time confidence scores.
- **Exploratory Data Analysis (EDA)**: Fully interactive `Plotly` graphs showing the correlation between ambient temperature, humidity, and the risk of spoilage, as well as a 3D scatter plot for multivariate analysis.
- **Advanced Machine Learning**: Upgraded to a `RandomForestClassifier` (an ensemble method) to prevent overfitting on the massive 40,000+ row dataset.
- **Explainable AI (Feature Importance)**: The app displays exactly which scientific features cause the most spoilage, proving the model learned correct physical constraints (e.g., Object Temperature > Ambient Temperature for spoilage).
- **Robust Data Pipeline**: Cleaned raw string column names, dropped redundant Fahrenheit columns (to prevent multicollinearity), and imputed missing median temperatures.

## 🏗️ System Architecture & Flow

```mermaid
graph TD
    A[Raw Kaggle Dataset] -->|"kagglehub download"| B(Data Pipeline)
    
    subgraph Streamlit Backend (app.py)
        B -->|Feature Engineering| C{Data Cleaning}
        C -->|Impute NaN, Drop Cols| D[Cleaned DataFrame]
        D -->|Train/Test Split| E(Random Forest Classifier)
    end
    
    subgraph Streamlit Frontend
        E -.->|"Extract Insights"| F[EDA & Feature Importance Tabs]
        G[User Input Sliders] -->|Live Sensor Data| H[Prediction Engine]
        E -->|"Model Weights"| H
        H -->|High/Low Risk| I{Dashboard Alert}
    end
```

## 🚀 The Project Flow (How to Demo)
1. **The Hook**: Globally, 30% of food is wasted, and a massive portion spoils silently in cold-chain trucks.
2. **The Data**: We pulled 40,000+ real IoT sensor readings (Temperature, Humidity, Cooling Power) from Kaggle.
3. **The AI**: We trained an explainable Random Forest model to predict spoilage probability based on those sensor readings.
4. **The UI**: We built a Streamlit dashboard showing interactive 3D risk zones, feature correlations, and a live prediction simulator for dispatchers.

## Running Locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Make sure you load the dataset using:
   ```bash
   python -c "import kagglehub; kagglehub.dataset_download('syedalihaidernaqvi/cold-supply-chain-data')"
   ```
   *(Note: The data is already saved locally as `real_cold_chain_data.csv` for convenience).*
3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Live Deployment (Hackathon Ready)
To share this project with the judges via a public URL, deploy it for free using **Streamlit Community Cloud**:
1. Push this entire project folder (including `requirements.txt` and `app.py`) to a public **GitHub** repository.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and create a free account linked to your GitHub.
3. Click **"New App"** and select your FreshTrack repository.
4. Set the Main file path to `app.py`.
5. Click **Deploy!** Your app will be live on the internet in about 2 minutes.
