import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def create_growth_plot(simulation_df):
    """Create interactive plot for growth curves"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=simulation_df['time'],
        y=simulation_df['growth'],
        name='Normal Growth',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=simulation_df['time'],
        y=simulation_df['drug_response'],
        name='With Drug',
        line=dict(color='red')
    ))
    
    fig.update_layout(
        title='Organoid Growth Curves',
        xaxis_title='Time',
        yaxis_title='Population',
        height=400
    )
    
    return fig

def create_prediction_plot(actual, predicted):
    """Create scatter plot for actual vs predicted values"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=actual,
        y=predicted,
        mode='markers',
        marker=dict(color='blue'),
        name='Predictions'
    ))
    
    # Add diagonal line for perfect predictions
    max_val = max(max(actual), max(predicted))
    min_val = min(min(actual), min(predicted))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Perfect Prediction'
    ))
    
    fig.update_layout(
        title='Actual vs Predicted Drug Effectiveness',
        xaxis_title='Actual Effectiveness',
        yaxis_title='Predicted Effectiveness',
        height=400
    )
    
    return fig
