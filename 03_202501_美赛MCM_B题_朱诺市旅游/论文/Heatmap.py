import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib import rcParams

# Ensure proper display of negative signs
rcParams['axes.unicode_minus'] = False

# ================== Base Parameters ==================
base_params = {
    'x_ship': 3000,
    'x_land': 1200,
    'eta_emission': 0.10,
    'eco_repair_ratio': 0.40,    # Eco repair cost ratio
    'tax_ship': 78,             
    'tax_land': 18,
    'ship_co2': 0.01913,         # Cruise carbon emission coefficient (tons/person)
    'land_co2': 0.01120          # Land tourist carbon emission coefficient (tons/person)
}

# ================== Optimization Model ==================
def multi_objective_model(x):
    """Multi-objective optimization model"""
    x_ship, x_land = x
    tax_ship_season = x_ship * base_params['tax_ship'] * 1.2 * 92
    tax_ship_off = x_ship * base_params['tax_ship'] * (365 - 92)
    tax_land_total = x_land * base_params['tax_land'] * 365
    tax_revenue = tax_ship_season + tax_ship_off + tax_land_total
    eco_cost = tax_revenue * base_params['eco_repair_ratio']
    housing_subsidy = tax_revenue * 0.10
    emission = (x_ship * 365 * base_params['ship_co2'] + x_land * 365 * base_params['land_co2']) * (1 - base_params['eta_emission'])
    satisfaction = 3.8 + 0.35 * (base_params['eco_repair_ratio'] / 0.4) + 0.25 - 0.1 * (x_ship / 3000 + x_land / 1200)
    return emission / 1e3  # Convert to kilotons

# ================== Data Generation ==================
x_ship_values = np.linspace(2500, 12000, 100)  # Range of cruise visitors
x_land_values = np.linspace(800, 1200, 100)    # Range of land visitors
X_ship, X_land = np.meshgrid(x_ship_values, x_land_values)

tax_revenue_matrix = np.zeros_like(X_ship)
emission_matrix = np.zeros_like(X_ship)
satisfaction_matrix = np.zeros_like(X_ship)

for i in range(X_ship.shape[0]):
    for j in range(X_ship.shape[1]):
        x_ship = X_ship[i, j]
        x_land = X_land[i, j]
        
        # Tax revenue calculation
        tax_ship_season = x_ship * base_params['tax_ship'] * 1.2 * 92
        tax_ship_off = x_ship * base_params['tax_ship'] * (365 - 92)
        tax_land_total = x_land * base_params['tax_land'] * 365
        tax_revenue = tax_ship_season + tax_ship_off + tax_land_total
        
        # Carbon emission calculation
        emission = (x_ship * 365 * base_params['ship_co2'] + x_land * 365 * base_params['land_co2']) * (1 - base_params['eta_emission'])
        
        # Satisfaction calculation
        satisfaction = 3.8 + 0.35 * (base_params['eco_repair_ratio'] / 0.4) + 0.25 - 0.1 * (x_ship / 3000 + x_land / 1200)
        
        tax_revenue_matrix[i, j] = tax_revenue / 1e6  # Million USD
        emission_matrix[i, j] = emission / 1e3       # Kilotons
        satisfaction_matrix[i, j] = satisfaction

# ================== Visualization: Heatmap ==================
plt.figure(figsize=(12, 8))

# Heatmap
heatmap = plt.pcolormesh(
    X_ship, X_land, satisfaction_matrix, cmap='coolwarm', shading='auto'
)
plt.colorbar(heatmap, label='Satisfaction Score')

# Contour lines
contour_tax = plt.contour(
    X_ship, X_land, tax_revenue_matrix, levels=5, colors='green', linestyles='--'
)
plt.clabel(contour_tax, inline=True, fontsize=10, fmt="%.1f M USD")

contour_emission = plt.contour(
    X_ship, X_land, emission_matrix, levels=[28.0], colors='red', linestyles='-'
)
plt.clabel(contour_emission, inline=True, fontsize=10, fmt="Emission Limit: %.1f kt")

# Plot aesthetics
plt.title('Multi-Objective Heatmap of Cruise and Land Visitors', fontsize=14)
plt.xlabel('Cruise Visitors (persons/day)', fontsize=12)
plt.ylabel('Land Visitors (persons/day)', fontsize=12)
plt.grid(visible=True, linestyle='--', alpha=0.5)

# Save or show the plot
plt.tight_layout()
plt.show()
