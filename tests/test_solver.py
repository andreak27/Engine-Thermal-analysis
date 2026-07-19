import numpy as np

from engine_thermal_simulator import simulate_component


def test_simulator_returns_finite_temperature_field():
    result = simulate_component(
        thickness=0.01,
        height=0.005,
        nx=7,
        ny=5,
        material="Inconel 718",
        hot_gas_temperature=1000.0,
        coolant_temperature=300.0,
        hot_side_h=2500.0,
        coolant_side_h=5000.0,
        use_radiation=False,
        steady_state=True,
    )

    assert result["temperature"].shape == (7, 5)
    assert np.isfinite(result["temperature"]).all()
    assert result["temperature"].max() > result["temperature"].min()
    assert result["temperature"][0, 0] < result["temperature"][-1, -1]


def test_transient_simulation_exposes_history_and_margin():
    result = simulate_component(
        thickness=0.01,
        height=0.005,
        nx=7,
        ny=5,
        material="SS304",
        hot_gas_temperature=1100.0,
        coolant_temperature=300.0,
        steady_state=False,
        time_steps=10,
        dt=0.05,
    )

    assert len(result["time_history"]) == 10
    assert result["temperature_history"].shape[0] == 10
    assert result["safety_margin"] >= 0.0
    assert result["design_margin_percent"] >= 0.0
