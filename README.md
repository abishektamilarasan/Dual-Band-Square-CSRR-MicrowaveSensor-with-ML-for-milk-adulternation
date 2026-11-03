# Milk Adulteration Microwave Sensor

![CST](https://img.shields.io/badge/simulation-CST%20Studio%20Suite%202026-blue)
![Python](https://img.shields.io/badge/python-scikit--learn%20%7C%20SciPy-green)
![Streamlit](https://img.shields.io/badge/app-Streamlit-red)

A two-ring microwave sensor, simulated in CST, that shows how water and sugar added to milk shift its resonance, plus a machine-learning app that tells you **what was added and how much**, and also works backwards (pick an adulterant and amount, see the expected sensor response).


[![Watch the demo](images/cst_top_view.png)](https://drive.google.com/drive/folders/1W3719tTcmsYZ4h3aJ749Bmdmj2b1m0iO?usp=drive_link)

**[Demo video and files on Google Drive](https://drive.google.com/drive/folders/1W3719tTcmsYZ4h3aJ749Bmdmj2b1m0iO?usp=drive_link)** · Full write-up: [https://github.com/abishektamilarasan/Dual-Band-Square-CSRR-MicrowaveSensor-with-ML-for-milk-adulternation/blob/main/Milk%20Adulteration%20Microwave%20Sensor%20-%20Project%20Report.pdf](docs)

## At a glance

- Two square split-ring slot resonators on one microstrip line (dual band: about 2.2 GHz and 4.5 GHz).
- 16 simulated samples: air, fresh and synthetic milk, water 10 to 100 ml, sugar 2 to 15 g, electrolyte mixtures.
- Readings use the **first ring (about 2.2 GHz)** only, where literature permittivity data for milk is plentiful.
- Water lowers the resonant frequency by about 0.66 MHz per ml; sugar barely moves it, so the two are distinguishable.
- A Random Forest and spline model inside a Streamlit app identifies the adulterant and estimates its amount.

## Gallery

| Sensor geometry (mm) | S21 with fresh milk |
| :---: | :---: |
| ![Geometry](images/geometry_annotated.png) | ![S21](images/s21_fresh_milk.png) |

| Resonance shift vs amount added | Streamlit app (forward mode) |
| :---: | :---: |
| ![f1 shift](images/f1_vs_amount.png) | ![App](images/app_forward.png) |

More figures (parametric sweeps, S11, inverse mode, far-field) are in [`images/`](images).

## How it works

1. **Design:** the two-ring sensor is tuned in CST with parameter sweeps (patch sizes, slot gap, ring spacing, feed width).
2. **Simulate:** each liquid is loaded over the first ring using literature permittivity values.
3. **Extract:** resonant frequency f1, notch depth and Q1 are read from S21.
4. **Predict:** the model names the adulterant and estimates the amount; the inverse mode generates f1, depth, Q1 and an S21 curve.

## Repository structure

```
.
├── README.md
├── GEOMETRY.md            all sensor coordinates and simulation set-up
├── sensor_geometry.csv    same coordinates, machine-readable
├── cst/                   CST project file
├── ml/                    model, Streamlit app and dataset
├── docs/                  project report
└── images/                figures used here
```

## Quick start

**Sensor:** open the project in `cst/` with CST Studio Suite (the Education edition is enough). Sensor dimensions are in [GEOMETRY.md](GEOMETRY.md).

**App:** put `app.py`, `milk_model.py` and `milk_sensor_dataset.csv` in `ml/`, then

```bash
pip install numpy pandas scipy scikit-learn streamlit plotly
cd ml
streamlit run app.py        # opens at http://localhost:8501
```

## Machine learning

*

## Notes

- Everything is simulated; no fabricated sensor has been measured yet.
- The original lab CST model is no longer available. The model in `cst/` is the one built in the student version.
- Limitations, the dielectric data sources and all results are in the report in `docs/`.

