import streamlit as st
import pandas as pd
import numpy as np

from simulation import OrganoidSimulator, SimulationParams, generate_synthetic_data
from ml_model import DrugResponsePredictor
from visualization import create_growth_plot, create_prediction_plot

st.set_page_config(page_title="Organoid Drug Response Simulator", layout="wide")

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = DrugResponsePredictor()
    synthetic_data = generate_synthetic_data()
    st.session_state.model_metrics = st.session_state.predictor.train(synthetic_data)

st.title("Organoid Drug Response Simulator")

# Sidebar for parameters
st.sidebar.header("Simulation Parameters")

drug_concentration = st.sidebar.slider(
    "Drug Concentration",
    min_value=0.0,
    max_value=2.0,
    value=1.0,
    step=0.1
)

growth_factor = st.sidebar.slider(
    "Growth Factor",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.1
)

time_period = st.sidebar.slider(
    "Time Period",
    min_value=10,
    max_value=50,
    value=30,
    step=5
)

# Create two columns for the main content
col1, col2 = st.columns(2)

# Simulation section
with col1:
    st.header("Organoid Growth Simulation")
    
    # Run simulation
    simulator = OrganoidSimulator()
    params = SimulationParams(
        drug_concentration=drug_concentration,
        time_period=time_period,
        growth_factor=growth_factor
    )
    
    simulation_df = simulator.simulate_growth(params)
    
    # Display growth plot
    st.plotly_chart(create_growth_plot(simulation_df), use_container_width=True)

# ML Prediction section
with col2:
    st.header("ML Model Predictions")
    
    # Display model metrics
    st.subheader("Model Performance")
    st.write(f"Training Score: {st.session_state.model_metrics['train_score']:.3f}")
    st.write(f"Test Score: {st.session_state.model_metrics['test_score']:.3f}")
    
    # Make prediction for current parameters
    current_params = pd.DataFrame([{
        'drug_concentration': drug_concentration,
        'growth_factor': growth_factor,
        'time_period': time_period
    }])
    
    predicted_effectiveness = st.session_state.predictor.predict(current_params)[0]
    
    st.subheader("Current Prediction")
    st.write(f"Predicted Drug Effectiveness: {predicted_effectiveness:.3f}")

# Export section
st.header("Export Results")
if st.button("Export Simulation Data"):
    csv = simulation_df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="simulation_results.csv",
        mime="text/csv"
    )
