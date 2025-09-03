# Spacecraft GNC Terminal    
```
 ____  ____   __    ___  ____  ___  ____   __   ____  ____     ___  __ _   ___       
/ ___)(  _ \ / _\  / __)(  __)/ __)(  _ \ / _\ (  __)(_  _)   / __)(  ( \ / __)      
\___ \ ) __//    \( (__  ) _)( (__  )   //    \ ) _)   )(    ( (_ \/    /( (__       
(____/(__)  \_/\_/ \___)(____)\___)(__\_)\_/\_/(__)   (__)    \___/\_)__) \___)    
```
**Spacecraft GNC** is a modular simulation toolkit for spacecraft orbital dynamics and attitude control. Ported from a validated MATLAB implementation and grounded in theory from a dedicated research [paper](Leif2112/SpacecraftGNC/master/SDM_Report.pdf), this project provides a Python platform for GNC simulation.

---

## Features

- Propagate orbits using classical orbital elements (COEs)
- Solve Kepler's equation 
- Integrate the two-body problem (TBP) using a high-order Runge-Kutta method
- Quaternion propagation of attitude dynamics 
- Plots o' plenty:
  - Mean, Eccentric, and True Anomalies over time
  - Velocity and Position error evolution
  - Angular velocities, angular momentum, kinetic rotational energy
  - Polhode plot
  - 3D ECI / ECEF orbital trajectory
  - Ground track (In-progress)

---

## Getting Started

### 1. Install Poetry

First, install [Poetry](https://python-poetry.org/):

```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

Then ensure Poetry's scripts are in your system PATH:

```powershell
$env:Path += ";$env:USERPROFILE\AppData\Roaming\Python\Scripts"
```

Or restart your terminal / IDE.

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

## Project Structure

```text
SPACECRAFT_GNC
├── .pytest_cache/
├── .vscode/
├── data/
├── src/
│   ├── ascii_globe/
│   └── gnc/
│       ├── __init__.py
│       ├── attitude/
│       ├── dynamics/
│       ├── integrators/
│       ├── simulation/
│       ├── visualisation/
│       └── __pycache__/
├── tests/
├── poetry.lock
├── pyproject.toml
├── pytest.ini
└── README.md

```

---

## Dependencies

Managed via Poetry, check pyproject.toml

Install all with:
```bash
poetry install
```

---

## TODO

- [ ] B-dot & PD Attitude Control
- [ ] External disturbances (magnetic torque, drag)
- [ ] Sensor simulation (gyros, sun sensors)

---

## Resources

- 📄 Research paper (TBD / to be linked)

---

## Acknowledgements

This project is maintained by Leif Tinwell.

---

## 📜 License

**MIT License** — you're free to use, adapt, and share.

> Pull requests welcome. Star the repo if you find this useful 

