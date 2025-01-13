# Organoid Drug Response Simulator

A cutting-edge scientific simulation platform for studying drug responses in organoids using advanced machine learning and interactive data visualization technologies.

## Overview

This project provides a web-based interface for simulating and analyzing how organoids respond to various drugs. It combines biological simulation with machine learning predictions to help researchers understand drug effectiveness.

## Features

- **Interactive Simulation**: Real-time visualization of organoid growth patterns
- **Drug Response Modeling**: Simulate effects of different drug concentrations
- **Machine Learning Integration**: Predictive modeling of drug effectiveness
- **Data Export**: Export simulation results for further analysis

## Technical Stack

### Core Technologies
- **Python 3.11**: Primary programming language
- **Streamlit**: Web interface framework
- **Plotly**: Interactive data visualization
- **Scikit-learn**: Machine learning implementation
- **NumPy/Pandas**: Data manipulation and analysis

### Components

1. **Simulation Engine** (`simulation.py`)
   - Implements `OrganoidSimulator` class
   - Uses logistic growth model for organoid simulation
   - Handles drug concentration effects
   - Generates synthetic data for ML training

2. **Machine Learning Model** (`ml_model.py`)
   - `DrugResponsePredictor` class using Random Forest
   - Feature scaling with StandardScaler
   - Handles model training and predictions
   - Includes performance metrics

3. **Visualization Module** (`visualization.py`)
   - Interactive Plotly graphs
   - Growth curve visualization
   - Prediction accuracy plots
   - Real-time data updates

4. **Web Interface** (`app.py`)
   - Streamlit-based dashboard
   - Parameter controls in sidebar
   - Split view for simulation and ML predictions
   - Data export functionality

## Getting Started

1. Install dependencies:
   ```bash
   pip install streamlit numpy pandas plotly scikit-learn
   ```

2. Run the application:
   ```bash
   streamlit run app.py
   ```

3. Access the interface at `http://localhost:5000`

## Usage

1. **Adjust Parameters**:
   - Drug Concentration (0.0-2.0)
   - Growth Factor (0.1-1.0)
   - Time Period (10-50)

2. **View Results**:
   - Left panel shows growth simulation
   - Right panel displays ML predictions
   - Export data using the download button

## Implementation Details

### Simulation Model
- Uses logistic growth equation for organoid population
- Incorporates drug concentration effects through inhibition factor
- Generates synthetic data with realistic variations

### Machine Learning Pipeline
- Random Forest Regressor for prediction
- StandardScaler for feature normalization
- Train/Test split for model validation
- Real-time prediction updates

### Visualization Features
- Interactive growth curves
- Comparison between normal and drug-affected growth
- Prediction accuracy visualization
- Real-time updates with parameter changes

## Future Enhancements
- Integration with real experimental data
- Additional ML models for comparison
- 3D visualization of organoid structures
- Advanced parameter optimization

## Technical Considerations
- Optimized for real-time interaction
- Scalable data processing pipeline
- Modular architecture for easy extension
- Comprehensive error handling

## License
MIT License

Created as part of a scientific research simulation platform.
