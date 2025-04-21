# 🛰️ Spacecraft GNC (Guidance, Navigation & Control)

**Spacecraft GNC** is a modular simulation toolkit for spacecraft orbital dynamics and attitude control. Ported from a validated MATLAB implementation and grounded in theory from a dedicated research paper, this project provides a reproducible and extensible Python platform for GNC simulation.

---

## 📌 Features

- ✅ Propagate orbits using classical orbital elements (COEs)
- ✅ Solve Kepler's equation via Newton-Raphson (with JIT acceleration)
- ✅ Integrate the two-body problem (TBP) using a high-order Runge-Kutta method
- ✅ Compare numerical and analytical solutions (energy, error diagnostics)
- ✅ Plot:
  - Mean, Eccentric, and True Anomalies over time
  - Velocity and Position error evolution
  - Specific orbital energy
  - 3D ECI orbital trajectory with Earth
- ✅ Toggle plots interactively

---

## 🛠️ Getting Started

### 1. Install Poetry

First, install [Poetry](https://python-poetry.org/):

```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

Then ensure Poetry's scripts are in your system PATH:

```powershell
$env:Path += ";$env:USERPROFILE\AppData\Roaming\Python\Scripts"
```

Or restart your terminal/editor (e.g., VS Code).

---

### 2. Clone the Repo and Install Dependencies

```bash
git clone https://github.com/your-username/spacecraft-gnc.git
cd spacecraft-gnc
poetry install
```

---

### 3. Activate the Virtual Environment *(optional but faster)*

```bash
poetry shell
```

Then run the simulation directly with:

```bash
python -m gnc.simulation.main
```

Or simply:

```bash
sim
```

---

## 📂 Project Structure

```text
spacecraft_gnc/
├── pyproject.toml            # Poetry config and scripts
├── src/
│   └── gnc/
│       ├── dynamics/         # Orbital mechanics
│       ├── integrators/      # Custom ode113-like solver
│       ├── attitude/         # Rigid body kinematics (quaternions)
│       ├── simulation/       # Main runtime logic (main.py)
│       └── visualisation/    # Modular plotting scripts
└── tests/                    # Unit tests for core functions
```

---

## 🧪 Running Tests

Run all tests with:

```bash
poetry run pytest
```

Tests cover:
- Quaternion conversion
- Angular velocity propagation
- Energy and angular momentum conservation
- Orbital state errors and convergence

---

## 📈 Plots Generated

- 📉 **Specific Energy vs Time**
- 📊 **Anomaly Evolution (Mean, Eccentric, True)**
- 🔁 **Velocity and Position Error over Time**
- 🌍 **3D ECI Trajectory around Earth**

> Plots are optional — you'll be prompted interactively to show or skip them.

---

## 📦 Dependencies

Managed via Poetry. Key packages:

- `numpy` – numerical operations
- `scipy` – integrators, constants
- `matplotlib` – plotting
- `numba` – high-performance Kepler solver
- `colorama` – coloured terminal feedback
- `pytest` – testing

Install all with:
```bash
poetry install
```

---

## 🧭 Future Features (Planned)

- [ ] B-dot & PD Attitude Control
- [ ] External disturbances (magnetic torque, drag)
- [ ] Sensor simulation (gyros, sun sensors)
- [ ] Quaternion-based animation of spacecraft body
- [ ] Exportable PDF/HTML reports with embedded plots

---

## 📚 Resources

- 📄 Research paper (TBD / to be linked)
- 📁 MATLAB reference implementation
- 📸 Screenshots and sample outputs

---

## 🤝 Acknowledgements

This project is maintained by Leif Tinwell.

---

## 📜 License

**MIT License** — you're free to use, adapt, and share.

> Pull requests welcome. Star the repo if you find this useful 🚀

