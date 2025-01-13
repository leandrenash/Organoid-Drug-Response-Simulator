import numpy as np
from dataclasses import dataclass
import pandas as pd

@dataclass
class SimulationParams:
    drug_concentration: float
    time_period: int
    growth_factor: float

class OrganoidSimulator:
    def __init__(self):
        self.time_points = None
        self.growth_curve = None
        self.drug_response = None
    
    def simulate_growth(self, params: SimulationParams):
        """Simulate organoid growth with given parameters"""
        self.time_points = np.linspace(0, params.time_period, 100)
        
        # Simulate growth curve with logistic growth
        carrying_capacity = 1.0
        growth_rate = params.growth_factor
        initial_population = 0.1
        
        self.growth_curve = carrying_capacity / (1 + ((carrying_capacity - initial_population) / 
                                                     initial_population) * np.exp(-growth_rate * self.time_points))
        
        # Simulate drug response
        drug_effect = 1 - (params.drug_concentration / (params.drug_concentration + 1))
        self.drug_response = self.growth_curve * drug_effect
        
        return self._create_simulation_df()
    
    def _create_simulation_df(self):
        """Create a DataFrame with simulation results"""
        df = pd.DataFrame({
            'time': self.time_points,
            'growth': self.growth_curve,
            'drug_response': self.drug_response
        })
        return df

def generate_synthetic_data(n_samples=100):
    """Generate synthetic data for ML model training"""
    np.random.seed(42)
    
    data = []
    for _ in range(n_samples):
        drug_conc = np.random.uniform(0, 2)
        growth_factor = np.random.uniform(0.1, 1)
        time_period = np.random.randint(10, 50)
        
        params = SimulationParams(
            drug_concentration=drug_conc,
            time_period=time_period,
            growth_factor=growth_factor
        )
        
        simulator = OrganoidSimulator()
        df = simulator.simulate_growth(params)
        
        # Calculate effectiveness metric
        effectiveness = np.mean(1 - df['drug_response'] / df['growth'])
        
        data.append({
            'drug_concentration': drug_conc,
            'growth_factor': growth_factor,
            'time_period': time_period,
            'effectiveness': effectiveness
        })
    
    return pd.DataFrame(data)
