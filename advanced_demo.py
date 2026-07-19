from engine_thermal_simulator import simulate_component, sweep_parametric

if __name__ == "__main__":
    result = simulate_component(
        thickness=0.012,
        height=0.006,
        nx=40,
        ny=24,
        material="Inconel 718",
        hot_gas_temperature=1400.0,
        coolant_temperature=300.0,
        hot_side_h=3500.0,
        coolant_side_h=9000.0,
        use_radiation=True,
        steady_state=False,
        time_steps=80,
        dt=0.05,
        wall_temperature_limit=1150.0,
        output_dir="outputs",
    )
    sweep_parametric(
        thickness_values=[0.008, 0.010, 0.012],
        coolant_flow_values=[4000.0, 7000.0, 10000.0],
        material_names=["Inconel 718", "SS304"],
        output_dir="outputs",
    )

    print("Hot-wall temperature:", round(result["hot_wall_temperature"], 1))
    print("Safety margin:", round(result["safety_margin"], 1))
    print("Design margin percent:", round(result["design_margin_percent"], 1))
