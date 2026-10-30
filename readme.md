# Industrial Carbon Emission Forecasting & Strategy System

An end-to-end machine learning system and interactive Streamlit web dashboard for forecasting industrial carbon emissions ($tCO_2e$) and simulating decarbonization strategies.

---

## Project Structure

```
carbon_app/
├── app.py                                     # Interactive Streamlit application (self-contained)
├── carbon_emission_dataset_with_Industry.csv  # Historical dataset (18,250 facility records)
├── carbon_emission_pipeline.ipynb             # End-to-end Jupyter Notebook (data to evaluation)
├── models/                                    # Serialized model artifacts (generated upon training)
│   ├── lstm_model.keras                       # Trained Keras stacked LSTM model
│   ├── feature_scaler.joblib                  # Fitted MinMaxScaler for 15 operational features
│   ├── target_scaler.joblib                   # Fitted MinMaxScaler for emission target
│   ├── metrics.json                           # Test evaluation metrics (MAE, RMSE, R²) and loss curves
│   └── test_data.npz                          # Held-out test sequences and ground-truth values
├── requirements.txt                           # Project dependencies
└── README.md                                  # Setup and execution guide
```

---

## Getting Started

### 1. Installation

Set up a virtual environment and install the required dependencies:

```bash
# Optional: create and activate virtual environment
python -m venv venv

# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

### 2. Model Training

You have two simple ways to train the model:

#### Option A: Run the Jupyter Notebook
Open and execute all cells in `carbon_emission_pipeline.ipynb`:
```bash
jupyter notebook carbon_emission_pipeline.ipynb
```
Running the notebook trains the stacked LSTM model, logs evaluation metrics, and automatically populates the `models/` directory.

#### Option B: In-App Training via Streamlit
Launch the app directly. If the `models/` directory is not yet generated, click the **"Train Model Now"** button in the sidebar to train and serialize the model directly within the browser interface.

---

### 3. Launching the Streamlit Application

Run the dashboard with:

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## Dashboard Features

The dashboard provides four interactive views accessible from the sidebar:

1. **Overview**:
   - Executive KPI cards displaying model performance: MAE, RMSE, and $R^2$ Score.
   - Interactive line chart tracking actual vs. predicted emissions across held-out test sequences.

2. **Model Performance**:
   - Epoch-by-epoch loss convergence curves (Training Loss vs. Validation Loss).
   - Actual vs. predicted regression plot for test observations.

3. **Forecast Prediction**:
   - Live operational input controls for energy consumption (Total, Renewable, Non-Renewable), production output, transport distance, and raw materials.
   - Instant real-time LSTM forecasting with input breakdown bar charts and summary tables.

4. **Decision Recommendations**:
   - Automated prescriptive scenario simulation evaluating operational adjustments (e.g., cutting non-renewable energy, increasing renewable share, improving process efficiency).
   - Highlighted best-action card showing projected $tCO_2e$ reduction and an interactive horizontal comparison chart.

---

## Model Architecture & Technical Details

- **Sequence Window**: 60-day sliding time-series windows per facility. Each window includes the current day's operational features to capture both historical trends and intra-day operational demand.
- **Data Scaling**: `MinMaxScaler(feature_range=(0, 1))` fitted independently on 15 operational inputs and the target emission variable.
- **Deep Learning Architecture**:
  - `LSTM(128, return_sequences=True)` $\rightarrow$ `Dropout(0.2)`
  - `LSTM(64)` $\rightarrow$ `Dropout(0.2)`
  - `Dense(32, activation="relu")` $\rightarrow$ `Dense(1, activation="relu")`
- **Optimization**: Adam optimizer with Mean Squared Error (MSE) loss, regularized using `EarlyStopping` (patience=10) and `ReduceLROnPlateau` (factor=0.5, patience=5).
