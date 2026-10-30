import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
load_model = tf.keras.models.load_model

@st.cache_resource
def load_lstm_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    return None

st.set_page_config(
    page_title="CO₂ Emission System",
    page_icon="CO2",
    layout="wide"
)

DATA_PATH = "carbon_emission_dataset_with_Industry.csv"
if not os.path.exists(DATA_PATH) and os.path.exists("datasets/carbon_emission_dataset_with_Industry.csv"):
    DATA_PATH = "datasets/carbon_emission_dataset_with_Industry.csv"

MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "lstm_model.keras")
FEATURE_SCALER_PATH = os.path.join(MODELS_DIR, "feature_scaler.joblib")
TARGET_SCALER_PATH = os.path.join(MODELS_DIR, "target_scaler.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
TEST_DATA_PATH = os.path.join(MODELS_DIR, "test_data.npz")

SEQUENCE_LENGTH = 60
TARGET_COLUMN = "Carbon_Emission_tCO2e_TARGET"

FEATURE_COLUMNS = [
    "Total_Energy_Consumption_kWh",
    "Renewable_Energy_Consumption_kWh",
    "NonRenewable_Energy_Consumption_kWh",
    "Production_Output_Units",
    "Supply_Chain_Transport_km",
    "Raw_Material_Usage_kg",
    "Energy_Cost_USD",
    "Carbon_Tax_USD",
    "Process_Efficiency_Percent",
    "Employment_Count",
    "Public_Acceptance_Index",
    "Strategy_Implementation_Cost_USD",
    "Expected_Carbon_Reduction_Percent",
    "Expected_Renewable_Share_Percent",
    "Social_Impact_Score"
]


def apply_theme():
    custom_css = """
    <style>
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    .metric-card-container {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
        transition: transform 0.15s ease-in-out, box-shadow 0.15s ease-in-out;
    }
    .metric-card-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-card-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-card-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0f172a;
    }
    .section-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1e293b;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .success-card {
        background-color: #f0fdf4;
        border: 1px solid #86efac;
        border-radius: 12px;
        padding: 20px 24px;
        margin: 1.25rem 0;
        color: #166534;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    .success-card h3 {
        margin-top: 0;
        margin-bottom: 12px;
        color: #15803d;
        font-weight: 700;
        font-size: 1.2rem;
    }
    .success-card p {
        margin: 6px 0;
        font-size: 0.95rem;
        color: #166534;
    }
    .success-card strong {
        color: #14532d;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


def metric_card(title, value):
    st.markdown(
        f"""
        <div class="metric-card-container">
            <div class="metric-card-title">{title}</div>
            <div class="metric-card-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def section_title(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def sidebar_navigation(pages):
    st.sidebar.markdown(
        """
        <div style="font-size: 1.1rem; font-weight: 700; color: #1e293b; margin-bottom: 0.75rem;">
            Dashboard Navigation
        </div>
        """,
        unsafe_allow_html=True
    )
    return st.sidebar.radio("Go to", pages, index=0, label_visibility="collapsed")


apply_theme()


def load_data(filepath):
    df = pd.read_csv(filepath)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    if "Company_ID" in df.columns and "Date" in df.columns:
        df = df.sort_values(["Company_ID", "Date"]).reset_index(drop=True)
    return df


def scale_features(df, feature_cols, target_col=TARGET_COLUMN):
    feat_scaler = MinMaxScaler(feature_range=(0, 1))
    tgt_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = df.copy()
    scaled[feature_cols] = feat_scaler.fit_transform(df[feature_cols].values)
    scaled[[target_col]] = tgt_scaler.fit_transform(df[[target_col]].values)
    return scaled, feat_scaler, tgt_scaler


def build_sequences(scaled_df, feature_cols, seq_len=SEQUENCE_LENGTH, target_col=TARGET_COLUMN):
    X_list, y_list, comp_list = [], [], []
    companies = scaled_df["Company_ID"].unique()
    for comp in companies:
        comp_df = scaled_df[scaled_df["Company_ID"] == comp].sort_values("Date")
        features_data = comp_df[feature_cols].values
        target_data = comp_df[target_col].values
        n_samples = len(comp_df)
        if n_samples < seq_len:
            continue
        for i in range(seq_len - 1, n_samples):
            window = features_data[i - seq_len + 1 : i + 1]
            target_val = target_data[i]
            X_list.append(window)
            y_list.append([target_val])
            comp_list.append(comp)
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32), np.array(comp_list)


def split_sequences(X, y, company_ids, test_size=0.2, random_state=42):
    return train_test_split(
        X, y, company_ids, test_size=test_size, random_state=random_state, shuffle=True
    )


def train_and_save_model(df):
    os.makedirs(MODELS_DIR, exist_ok=True)
    scaled_df, feat_scaler, tgt_scaler = scale_features(df, FEATURE_COLUMNS, TARGET_COLUMN)
    X, y, company_ids = build_sequences(scaled_df, FEATURE_COLUMNS, SEQUENCE_LENGTH, TARGET_COLUMN)
    X_train, X_test, y_train, y_test, _, companies_test = split_sequences(X, y, company_ids)

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="relu")
    ])

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)

    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )

    pred_scaled = model.predict(X_test, verbose=0)
    predictions = tgt_scaler.inverse_transform(pred_scaled)
    actual = tgt_scaler.inverse_transform(y_test)

    mae = float(mean_absolute_error(actual, predictions))
    rmse = float(np.sqrt(mean_squared_error(actual, predictions)))
    r2 = float(r2_score(actual, predictions))

    model.save(MODEL_PATH)
    joblib.dump(feat_scaler, FEATURE_SCALER_PATH)
    joblib.dump(tgt_scaler, TARGET_SCALER_PATH)

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "num_test_sequences": int(len(X_test)),
        "train_loss": [float(v) for v in history.history["loss"]],
        "val_loss": [float(v) for v in history.history["val_loss"]]
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    np.savez(
        TEST_DATA_PATH,
        X_test=X_test,
        actual=actual,
        predicted=predictions,
        companies_test=companies_test
    )

    test_data = {
        "X_test": X_test,
        "actual": actual,
        "predicted": predictions,
        "companies_test": companies_test
    }

    return model, feat_scaler, tgt_scaler, metrics, test_data


def generate_co2_suggestions(model, X_base, feature_cols, feat_scaler, tgt_scaler):
    strategies = [
        {"feature": "NonRenewable_Energy_Consumption_kWh", "action": "Reduce", "change_pct": -25.0},
        {"feature": "Renewable_Energy_Consumption_kWh", "action": "Increase", "change_pct": 30.0},
        {"feature": "Process_Efficiency_Percent", "action": "Improve", "change_pct": 15.0},
        {"feature": "Raw_Material_Usage_kg", "action": "Optimize", "change_pct": -15.0},
        {"feature": "Supply_Chain_Transport_km", "action": "Shorten", "change_pct": -20.0},
        {"feature": "Total_Energy_Consumption_kWh", "action": "Conserve", "change_pct": -15.0}
    ]

    sample_size = min(150, len(X_base))
    X_sample = X_base[:sample_size].copy()

    pred_base_scaled = model.predict(X_sample, verbose=0)
    pred_base = tgt_scaler.inverse_transform(pred_base_scaled) if tgt_scaler else pred_base_scaled
    baseline_avg = float(np.mean(pred_base))

    results = []
    for strat in strategies:
        feat_name = strat["feature"]
        if feat_name not in feature_cols:
            continue
        feat_idx = feature_cols.index(feat_name)
        change_pct = strat["change_pct"]

        X_scenario = X_sample.copy()
        if feat_scaler:
            min_val = feat_scaler.data_min_[feat_idx]
            max_val = feat_scaler.data_max_[feat_idx]
            val_range = max_val - min_val if max_val != min_val else 1.0
            unscaled = X_scenario[:, :, feat_idx] * val_range + min_val
            modified = unscaled * (1.0 + change_pct / 100.0)
            X_scenario[:, :, feat_idx] = np.clip((modified - min_val) / val_range, 0.0, 1.0)
        else:
            X_scenario[:, :, feat_idx] *= (1.0 + change_pct / 100.0)

        pred_scen_scaled = model.predict(X_scenario, verbose=0)
        pred_scen = tgt_scaler.inverse_transform(pred_scen_scaled) if tgt_scaler else pred_scen_scaled
        scenario_avg = float(np.mean(pred_scen))
        reduction = baseline_avg - scenario_avg

        results.append({
            "feature": feat_name,
            "action": strat["action"],
            "change_pct": change_pct,
            "baseline": baseline_avg,
            "scenario": scenario_avg,
            "reduction": reduction
        })

    return sorted(results, key=lambda x: x["reduction"], reverse=True)


@st.cache_data
def load_dataset():
    if not os.path.exists(DATA_PATH):
        return None
    return load_data(DATA_PATH)


@st.cache_resource
def load_lstm_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return load_model(MODEL_PATH)
    except Exception:
        return None


@st.cache_resource
def load_feature_scaler():
    if not os.path.exists(FEATURE_SCALER_PATH):
        return None
    try:
        return joblib.load(FEATURE_SCALER_PATH)
    except Exception:
        return None


@st.cache_resource
def load_target_scaler():
    if not os.path.exists(TARGET_SCALER_PATH):
        return None
    try:
        return joblib.load(TARGET_SCALER_PATH)
    except Exception:
        return None


@st.cache_data
def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    try:
        with open(METRICS_PATH, "r") as file:
            return json.load(file)
    except Exception:
        return None


@st.cache_data
def load_test_data():
    if not os.path.exists(TEST_DATA_PATH):
        return None
    try:
        data = np.load(TEST_DATA_PATH, allow_pickle=True)
        result = {
            "X_test": data["X_test"],
            "actual": data["actual"],
            "predicted": data["predicted"],
            "companies_test": data["companies_test"]
        }
        data.close()
        return result
    except Exception:
        return None


df = load_dataset()
model = load_lstm_model()
feature_scaler = load_feature_scaler()
target_scaler = load_target_scaler()
metrics = load_metrics()
test_data = load_test_data()

pages = [
    "Overview",
    "Model Performance",
    "Forecast Prediction",
    "Decision Recommendations"
]

selected_page = sidebar_navigation(pages)

st.sidebar.markdown("---")

if model is None or metrics is None or test_data is None:
    st.sidebar.warning("Model not trained yet.")
    if st.sidebar.button("Train Model Now", type="primary", use_container_width=True):
        if df is not None:
            with st.spinner("Training LSTM model on historical dataset..."):
                model, feature_scaler, target_scaler, metrics, test_data = train_and_save_model(df)
            st.sidebar.success("Training completed!")
            st.rerun()
        else:
            st.sidebar.error("Dataset not found to train model.")

if test_data is not None:
    X_TEST = test_data["X_test"]
    actual = test_data["actual"].flatten()
    predicted = test_data["predicted"].flatten()
    companies_test = test_data["companies_test"]
else:
    X_TEST = None
    actual = np.array([])
    predicted = np.array([])
    companies_test = np.array([])


if selected_page == "Overview":

    st.markdown(
        '<div class="main-title">CO₂ Emission System</div>',
        unsafe_allow_html=True
    )

    if metrics is None or len(actual) == 0:
        st.warning("⚠️ Model is not trained yet. Click 'Train Model Now' in the sidebar or run the training notebook.")
        st.stop()

    col1, col2, col3 = st.columns(3)

    with col1:
        metric_card("MAE", f'{metrics["mae"]:.2f}')

    with col2:
        metric_card("RMSE", f'{metrics["rmse"]:.2f}')

    with col3:
        metric_card("R² Score", f'{metrics["r2"]:.2f}')

    section_title("Actual vs Predicted Emissions")

    plot_df = pd.DataFrame(
        {
            "Sequence": range(1, len(actual) + 1),
            "Actual": actual,
            "Predicted": predicted
        }
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=plot_df["Sequence"],
            y=plot_df["Actual"],
            mode="lines",
            name="Actual"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=plot_df["Sequence"],
            y=plot_df["Predicted"],
            mode="lines",
            name="LSTM Predicted"
        )
    )

    fig.update_layout(
        height=450,
        xaxis_title="Test Sequence",
        yaxis_title="CO₂ Emission",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)


elif selected_page == "Model Performance":

    st.markdown(
        '<div class="main-title">LSTM Model Performance</div>',
        unsafe_allow_html=True
    )

    if metrics is None or len(actual) == 0:
        st.warning("⚠️ Model performance artifacts not found. Please train the model first.")
        st.stop()

    col1, col2, col3 = st.columns(3)

    with col1:
        metric_card("MAE", f'{metrics["mae"]:.2f}')

    with col2:
        metric_card("RMSE", f'{metrics["rmse"]:.2f}')

    with col3:
        metric_card("R² Score", f'{metrics["r2"]:.2f}')

    section_title("Actual vs Predicted")

    plot_df = pd.DataFrame(
        {
            "Sequence": range(1, len(actual) + 1),
            "Actual": actual,
            "Predicted": predicted
        }
    )

    fig = px.line(
        plot_df,
        markers=True,
        x="Sequence",
        y=["Actual", "Predicted"],
        labels={"value": "CO₂ Emission", "variable": "Type"}
    )

    fig.update_layout(
        template="plotly_white",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    section_title("Training and Validation Loss")

    loss_df = pd.DataFrame(
        {
            "Epoch": range(1, len(metrics["train_loss"]) + 1),
            "Training Loss": metrics["train_loss"],
            "Validation Loss": metrics["val_loss"]
        }
    )

    fig = px.line(
        loss_df,
        x="Epoch",
        y=["Training Loss", "Validation Loss"]
    )

    fig.update_layout(
        template="plotly_white",
        height=450
    )

    st.plotly_chart(fig, use_container_width=True)


elif selected_page == "Forecast Prediction":

    st.markdown(
        '<div class="main-title">Forecast Prediction</div>',
        unsafe_allow_html=True
    )

    st.write("Enter values in kWh in energy consumption: ")

    if model is None or feature_scaler is None or target_scaler is None:
        st.warning("⚠️ Please train the model first using the sidebar button.")
        st.stop()

    col1, col2, col3 = st.columns(3)

    with col1:
        total_energy = st.number_input(
            "Total Energy Consumption",
            min_value=0.0,
            value=0.0,
        )

        raw_material = st.number_input(
            "Raw Material Usage (kg)",
            min_value=0.0,
            value=0.0,
            step=1.0
        )

    with col2:
        renewable_energy = st.number_input(
            "Renewable Energy Consumption",
            min_value=0.0,
            value=0.0,
            step=1.0
        )

        supply_chain_transport = st.number_input(
            "Supply Chain Transport (km)",
            min_value=0.0,
            value=0.0,
            step=1.0
        )

    with col3:
        nonrenewable_energy = st.number_input(
            "Non-Renewable Energy Consumption",
            min_value=0.0,
            value=0.0,
            step=1.0
        )

        production_output = st.number_input(
            "Production Output (Units)",
            min_value=0.0,
            value=0.0,
            step=1.0
        )

    st.markdown("---")

    predict_button = st.button(
        "Predict CO₂ Emission",
        type="primary",
        use_container_width=True
    )

    if predict_button:

        main_values = {
            "Total_Energy_Consumption_kWh": total_energy,
            "Renewable_Energy_Consumption_kWh": renewable_energy,
            "NonRenewable_Energy_Consumption_kWh": nonrenewable_energy,
            "Production_Output_Units": production_output,
            "Supply_Chain_Transport_km": supply_chain_transport,
            "Raw_Material_Usage_kg": raw_material
        }

        default_values = {
            "Energy_Cost_USD": 5000.0,
            "Carbon_Tax_USD": 12.0,
            "Process_Efficiency_Percent": 50.0,
            "Employment_Count": 100.0,
            "Public_Acceptance_Index": 50.0,
            "Strategy_Implementation_Cost_USD": 7000.0,
            "Expected_Carbon_Reduction_Percent": 10.0,
            "Expected_Renewable_Share_Percent": 30.0,
            "Social_Impact_Score": 50.0
        }

        all_values = {
            **main_values,
            **default_values
        }

        input_values = np.array(
            [all_values[feature] for feature in FEATURE_COLUMNS],
            dtype=float
        )

        input_df = pd.DataFrame([input_values], columns=FEATURE_COLUMNS)

        scaled_input = feature_scaler.transform(input_df)

        sequence = np.repeat(
            scaled_input[np.newaxis, :, :],
            SEQUENCE_LENGTH,
            axis=1
        )

        prediction_scaled = model.predict(sequence, verbose=0)

        prediction = target_scaler.inverse_transform(prediction_scaled)

        predicted_co2 = float(prediction[0][0])

        st.session_state["live_prediction"] = predicted_co2
        st.session_state["live_input"] = main_values

    if "live_prediction" in st.session_state:

        predicted_co2 = st.session_state["live_prediction"]
        live_input = st.session_state["live_input"]

        section_title("Live Prediction")

        metric_col1, metric_col2, metric_col3 = st.columns(3)

        with metric_col1:
            metric_card("Predicted CO₂", f"{predicted_co2:,.2f}")

        with metric_col2:
            metric_card("Manual Features", "6")

        with metric_col3:
            metric_card("Model", "LSTM")

        st.markdown("---")

        section_title("Prediction Result")

        prediction_df = pd.DataFrame({
            "Prediction": ["Current Input"],
            "CO₂ Emission": [predicted_co2]
        })

        fig_prediction = px.line(
            prediction_df,
            x="Prediction",
            y="CO₂ Emission",
            markers=True,
            title="Live CO₂ Emission Prediction"
        )

        fig_prediction.update_traces(
            line=dict(width=3),
            marker=dict(size=12)
        )

        fig_prediction.update_layout(
            template="plotly_white",
            height=450,
            yaxis_title="CO₂ Emission",
            xaxis_title=""
        )

        st.plotly_chart(fig_prediction, use_container_width=True)

        input_chart_df = pd.DataFrame({
            "Feature": [
                "Total Energy",
                "Renewable Energy",
                "Non-Renewable Energy",
                "Production Output",
                "Supply Chain Transport",
                "Raw Material"
            ],
            "Value": [
                live_input["Total_Energy_Consumption_kWh"],
                live_input["Renewable_Energy_Consumption_kWh"],
                live_input["NonRenewable_Energy_Consumption_kWh"],
                live_input["Production_Output_Units"],
                live_input["Supply_Chain_Transport_km"],
                live_input["Raw_Material_Usage_kg"]
            ]
        })

        fig_inputs = px.bar(
            input_chart_df,
            x="Feature",
            y="Value",
            title="Live Operational Input Values"
        )

        fig_inputs.update_layout(
            template="plotly_white",
            height=450,
            xaxis_title="Operational Feature",
            yaxis_title="Input Value"
        )

        st.plotly_chart(fig_inputs, use_container_width=True)

        section_title("Prediction Input Summary")

        summary_df = pd.DataFrame({
            "Feature": list(live_input.keys()),
            "Value": list(live_input.values())
        })

        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True
        )

        st.success(f"Live prediction generated: {predicted_co2:,.2f}")


elif selected_page == "Decision Recommendations":

    st.markdown(
        '<div class="main-title">CO₂ Decision Recommendations</div>',
        unsafe_allow_html=True
    )

    if model is None or X_TEST is None or feature_scaler is None or target_scaler is None:
        st.warning("⚠️ Trained model not found. Please train the model first using the sidebar button.")
        st.stop()

    suggestions = generate_co2_suggestions(
            model,
            X_TEST,
            FEATURE_COLUMNS,
            feature_scaler,
            target_scaler
    )

    if len(suggestions) == 0:
        st.warning("No recommendations available.")
    else:
        section_title("Recommended Actions")

        recommendation_df = pd.DataFrame(suggestions)
        recommendation_df["reduction"] = recommendation_df["reduction"].round(2)

        st.dataframe(
            recommendation_df[
                [
                    "feature",
                    "action",
                    "change_pct",
                    "baseline",
                    "scenario",
                    "reduction"
                ]
            ],
            use_container_width=True
        )

        section_title("Best Emission Reduction Action")

        best = suggestions[0]

        st.markdown(
            f"""
            <div class="success-card">
            <h3>{best["action"]} {best["feature"]}</h3>
            <p>
            Recommended change:
            <strong>{best["change_pct"]}%</strong>
            </p>
            <p>
            Baseline emission:
            <strong>{best["baseline"]:.2f}</strong>
            </p>
            <p>
            Scenario emission:
            <strong>{best["scenario"]:.2f}</strong>
            </p>
            <p>
            Estimated reduction:
            <strong>{best["reduction"]:.2f}</strong>
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        chart_df = recommendation_df.copy()
        chart_df = chart_df.sort_values("reduction", ascending=True)

        fig = px.bar(
            chart_df,
            x="reduction",
            y="feature",
            orientation="h",
            labels={
                "reduction": "Estimated CO₂ Reduction",
                "feature": "Operational Feature"
            }
        )

        fig.update_layout(
            template="plotly_white",
            height=450
        )

        st.plotly_chart(fig, use_container_width=True)