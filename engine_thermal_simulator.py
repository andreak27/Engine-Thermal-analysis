from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _temperature_dependent_properties(material: Material, temperature: float) -> tuple[float, float, float]:
    conductivity = material.conductivity * (1.0 + 1.5e-4 * (temperature - 300.0))
    specific_heat = material.specific_heat * (1.0 + 3.0e-4 * (temperature - 300.0))
    density = material.density
    return conductivity, specific_heat, density


@dataclass
class Material:
    name: str
    density: float
    conductivity: float
    specific_heat: float
    emissivity: float = 0.8


MATERIALS = {
    "Inconel 718": Material(name="Inconel 718", density=8220.0, conductivity=20.0, specific_heat=460.0, emissivity=0.8),
    "SS304": Material(name="SS304", density=8000.0, conductivity=16.2, specific_heat=500.0, emissivity=0.3),
    "Aluminum": Material(name="Aluminum", density=2700.0, conductivity=205.0, specific_heat=900.0, emissivity=0.2),
}


def _get_material(name: str) -> Material:
    return MATERIALS[name]


def simulate_component(
    thickness: float = 0.01,
    height: float = 0.005,
    nx: int = 25,
    ny: int = 15,
    material: str = "Inconel 718",
    hot_gas_temperature: float = 1200.0,
    coolant_temperature: float = 300.0,
    hot_side_h: float = 3000.0,
    coolant_side_h: float = 5000.0,
    use_radiation: bool = True,
    steady_state: bool = True,
    time_steps: int = 200,
    dt: float = 0.1,
    output_dir: Optional[str] = None,
    wall_temperature_limit: float = 1200.0,
) -> Dict[str, Any]:
    material_data = _get_material(material)
    x = np.linspace(0.0, thickness, nx)
    y = np.linspace(0.0, height, ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing="ij")

    T = np.full((nx, ny), coolant_temperature, dtype=float)
    T[:, -1] = hot_gas_temperature
    T[0, :] = coolant_temperature

    history = None
    if steady_state:
        T = _solve_steady_state(T, dx, dy, material_data, hot_gas_temperature, coolant_temperature, hot_side_h, coolant_side_h, use_radiation)
        history = np.array([T])
    else:
        T, history = _solve_transient(T, dx, dy, material_data, hot_gas_temperature, coolant_temperature, hot_side_h, coolant_side_h, use_radiation, time_steps, dt)

    hot_wall_temperature = float(T[-1, -1])
    coolant_wall_temperature = float(T[0, 0])
    safety_margin = max(0.0, wall_temperature_limit - hot_wall_temperature)
    design_margin_percent = max(0.0, 100.0 * (1.0 - hot_wall_temperature / wall_temperature_limit))

    result = {
        "temperature": T,
        "x": x,
        "y": y,
        "material": material,
        "max_temperature": float(T.max()),
        "min_temperature": float(T.min()),
        "hot_wall_temperature": hot_wall_temperature,
        "coolant_wall_temperature": coolant_wall_temperature,
        "safety_margin": safety_margin,
        "design_margin_percent": design_margin_percent,
        "time_history": [float(h[-1, -1]) for h in history],
        "temperature_history": history,
    }

    if output_dir is not None:
        _write_outputs(result, output_dir)

    return result


def _solve_steady_state(
    T: np.ndarray,
    dx: float,
    dy: float,
    material: Material,
    hot_gas_temperature: float,
    coolant_temperature: float,
    hot_side_h: float,
    coolant_side_h: float,
    use_radiation: bool,
    max_iter: int = 5000,
    tol: float = 1e-6,
) -> np.ndarray:
    emissivity = material.emissivity
    nx, ny = T.shape

    for _ in range(max_iter):
        T_old = T.copy()
        k_eff, cp_eff, rho_eff = _temperature_dependent_properties(material, T.mean())
        alpha = k_eff / (rho_eff * cp_eff)
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                T[i, j] = 0.25 * (
                    T[i + 1, j] + T[i - 1, j] + T[i, j + 1] + T[i, j - 1]
                )

        for i in range(1, nx - 1):
            biot_cool = coolant_side_h * dx / max(k_eff, 1e-12)
            biot_hot = hot_side_h * dx / max(k_eff, 1e-12)
            T[i, 0] = (T[i, 1] + biot_cool * coolant_temperature) / (1.0 + biot_cool)
            T[i, -1] = (T[i, -2] + biot_hot * hot_gas_temperature) / (1.0 + biot_hot)

        for j in range(1, ny - 1):
            T[0, j] = (T[1, j] + coolant_temperature) / 2.0
            T[-1, j] = (T[-2, j] + hot_gas_temperature) / 2.0

        if np.max(np.abs(T - T_old)) < tol:
            break

    if use_radiation:
        T = T + 0.02 * (hot_gas_temperature - T) * emissivity

    return T


def _solve_transient(
    T: np.ndarray,
    dx: float,
    dy: float,
    material: Material,
    hot_gas_temperature: float,
    coolant_temperature: float,
    hot_side_h: float,
    coolant_side_h: float,
    use_radiation: bool,
    time_steps: int,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    T_hist = np.empty((time_steps, *T.shape), dtype=float)
    T_hist[0] = T
    for n in range(1, time_steps):
        T_new = T.copy()
        k_eff, cp_eff, rho_eff = _temperature_dependent_properties(material, T.mean())
        alpha = k_eff / (rho_eff * cp_eff)
        dt_eff = min(dt, 0.25 * min(dx, dy) ** 2 / max(alpha, 1e-12))
        for i in range(1, T.shape[0] - 1):
            for j in range(1, T.shape[1] - 1):
                lap = (T[i + 1, j] - 2 * T[i, j] + T[i - 1, j]) / dx**2 + (T[i, j + 1] - 2 * T[i, j] + T[i, j - 1]) / dy**2
                T_new[i, j] = T[i, j] + alpha * dt_eff * lap
        T = T_new
        T_hist[n] = T
    return T, T_hist


def sweep_parametric(
    thickness_values: Optional[list[float]] = None,
    coolant_flow_values: Optional[list[float]] = None,
    material_names: Optional[list[str]] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    thickness_values = thickness_values or [0.008, 0.010, 0.012]
    coolant_flow_values = coolant_flow_values or [5000.0, 7000.0, 9000.0]
    material_names = material_names or ["Inconel 718", "SS304"]

    results = []
    for thickness in thickness_values:
        for flow in coolant_flow_values:
            for material in material_names:
                res = simulate_component(
                    thickness=thickness,
                    material=material,
                    coolant_side_h=flow,
                    output_dir=None,
                )
                results.append({"thickness": thickness, "coolant_flow": flow, "material": material, "max_temperature": res["max_temperature"]})

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with (out / "sweep_results.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["thickness", "coolant_flow", "material", "max_temperature"])
            writer.writeheader()
            writer.writerows(results)

    return {"results": results}


def _write_outputs(result: Dict[str, Any], output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    contour = ax.contourf(result["x"], result["y"], result["temperature"].T)
    ax.set_title("Temperature field")
    ax.set_xlabel("Thickness (m)")
    ax.set_ylabel("Height (m)")
    fig.colorbar(contour, ax=ax, label="Temperature (K)")
    fig.tight_layout()
    fig.savefig(out / "temperature_field.png", dpi=180)
    plt.close(fig)

    times = np.linspace(0.0, 1.0, 50)
    temps = 1200.0 - 400.0 * times
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(times, temps, label="Cooldown trend")
    ax.plot(np.linspace(0.0, 1.0, len(result["time_history"])), result["time_history"], label="Simulation history", linestyle="--")
    ax.set_title("Cooldown curve")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Hot-wall temperature (K)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "cooldown_curve.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Hot-side", "Coolant-side", "Design margin"], [result["hot_wall_temperature"], result["coolant_wall_temperature"], result["design_margin_percent"]])
    ax.set_title("Safety margin")
    ax.set_ylabel("Temperature (K) / Margin (%)")
    fig.tight_layout()
    fig.savefig(out / "safety_margin.png", dpi=180)
    plt.close(fig)


def run_demo(output_dir: str = "outputs") -> Dict[str, Any]:
    result = simulate_component(output_dir=output_dir)
    sweep = sweep_parametric(output_dir=output_dir)
    return {"steady_state": result, "parametric_sweep": sweep}


if __name__ == "__main__":
    run_demo()
