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

**Key Hackathon Features:**
- **Interactive UI with Tabs**: We separated the application into Live Prediction, Exploratory Data Analysis (EDA), and ML Insights using Streamlit tabs for a professional user experience.
- **Exploratory Data Analysis (EDA)**: Fully interactive `Plotly` graphs showing the correlation between ambient temperature, humidity, and the risk of spoilage, as well as class imbalance.
- **Advanced Machine Learning**: Upgraded to a `RandomForestClassifier` (an ensemble method) to prevent overfitting on the massive 40,000+ row dataset.
- **Feature Importance Tracking**: The app uses the Random Forest model to display exactly which scientific features cause the most spoilage (proving out the physics behind the data).
- **Data Pipeline Excellence**: Cleaned raw string column names, dropped redundant Fahrenheit columns (to prevent multicollinearity), and imputed missing median temperatures to prove robust data engineering.

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
