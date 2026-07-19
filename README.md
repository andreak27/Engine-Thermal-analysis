# Engine Thermal Analysis

This project is a numerical heat-transfer simulator for engine components such as rocket chamber walls or nozzle sections. It models conduction through a solid wall, convection at the hot-gas and coolant sides, and optional radiation at high temperature, then solves a steady-state or transient temperature field with a finite-difference approach.

## Features
- 2D solid-wall temperature field solver with a finite-difference formulation
- Steady-state and transient simulation modes
- Temperature-dependent material properties and optional radiation coupling
- Parametric sweeps over wall thickness, coolant-side convection, and material choice
- Design-margin tracking against a user-defined temperature limit
- Plot generation for temperature distributions, cooldown curves, and safety-margin summaries

## Quick start
```bash
python -m pip install -r requirements.txt
python main.py
```

The run produces a set of plots in the outputs directory and writes a CSV sweep summary there as well.

## Example usage
```python
from engine_thermal_simulator import simulate_component, sweep_parametric

result = simulate_component(
    thickness=0.01,
    height=0.005,
    material="Inconel 718",
    hot_gas_temperature=1200.0,
    coolant_temperature=300.0,
    steady_state=False,
    time_steps=60,
    dt=0.05,
    wall_temperature_limit=1150.0,
)

print(result["hot_wall_temperature"])
print(result["safety_margin"])
print(result["design_margin_percent"])

sweep = sweep_parametric(thickness_values=[0.008, 0.010], coolant_flow_values=[5000.0, 7000.0])
```
