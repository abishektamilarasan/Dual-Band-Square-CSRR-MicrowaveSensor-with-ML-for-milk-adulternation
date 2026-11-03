# Dual-Band Microwave Sensor with Machine Learning for Milk Adulteration Detection

A planar two-ring microwave sensor, simulated in CST, that shows how water and sugar added to milk shift its resonance. A machine-learning app then identifies the adulterant and estimates how much was added.

*BECE305L – Antenna and Microwave Engineering (Slot A2+tA2), VIT Chennai*

[![Watch the demo video](images/cst_top_view.png)](https://drive.google.com/drive/folders/1W3719tTcmsYZ4h3aJ749Bmdmj2b1m0iO?usp=drive_link)

**Demo video and project files:** [Google Drive folder](https://drive.google.com/drive/folders/1W3719tTcmsYZ4h3aJ749Bmdmj2b1m0iO?usp=drive_link)

---

## Contents

1. [Overview](#overview)
2. [What makes it different](#what-makes-it-different)
3. [Sensor design](#sensor-design)
4. [Simulation setup](#simulation-setup)
5. [Test samples and dielectric data](#test-samples-and-dielectric-data)
6. [Results](#results)
7. [Why only the first ring](#why-only-the-first-ring-about-22-ghz)
8. [Machine learning and Streamlit app](#machine-learning-and-streamlit-app)
9. [Repository structure](#repository-structure)
10. [Limitations and future work](#limitations-and-future-work)
11. [Team](#team)
12. [References](#references)

---

## Overview

Milk is often adulterated with water, sugar and other additives, and laboratory tests are slow and not portable. A microwave resonator senses a liquid through its permittivity: a higher real permittivity (εr′) adds capacitance to the resonator and lowers its resonant frequency, and a lossy liquid makes the dip shallower and broader.

This project has two parts:

1. **The sensor.** A microstrip line on FR-4 with two square split-ring slot resonators of different size, designed and tuned in CST Studio Suite 2026. Each ring gives a notch in the transmission curve S21.
2. **The analysis.** For each milk sample we record the resonant frequency f1, the notch depth and the quality factor Q1. A machine-learning model then names the adulterant and estimates its amount, and a Streamlit app lets you use it in both directions.

## What makes it different

In the papers we reviewed, most milk-adulteration sensors report a single number, the frequency shift of one resonator. That shows the milk has changed, but not what was added. Many are also validated on clean solvents rather than milk. This project differs in that it:

- **names the adulterant,** using three readings per sample (f1, notch depth, Q1) instead of one;
- **works in both directions:** readings in, adulterant and amount out; or adulterant and amount in, expected sensor response and S21 curve out;
- **uses milk-specific permittivity data** (milk, diluted milk, sugar-adulterated milk) rather than pure solvents;
- **reports its limits** with leave-one-out validation on a small dataset (see [Limitations](#limitations-and-future-work)).

## Sensor design

![Annotated sensor geometry](images/geometry_annotated.png)

*Sensor geometry with CST dimensions (mm). Dashed circles are the inner walls of the liquid cells.*

| Item | Value |
| --- | --- |
| Substrate | FR-4 (permittivity and thickness: see [GEOMETRY.md](GEOMETRY.md)) |
| Feed line | 1.6 mm wide, 50 Ω microstrip |
| Ring 1 (lower band) | patch p1 = 9.76 mm, slot width 0.2 mm, liquid cell inner diameter 13 mm |
| Ring 2 (upper band) | patch p2 = 5.26 mm, slot width g2 = 0.2 mm, liquid cell inner diameter 12 mm |
| Ring spacing | x2 − x1 = 30.46 mm |
| Unloaded response | f1 = 2.168 GHz (−29.7 dB, Q ≈ 34); f2 ≈ 4.5 GHz (−23 dB) |

All solid coordinates are in [GEOMETRY.md](GEOMETRY.md) and [sensor_geometry.csv](sensor_geometry.csv).

### Tuning with parametric sweeps

![Parametric sweeps](images/parametric_sweeps.png)

*S21 sweeps (unloaded): (a) feed width, (b) p1, (c) p2, (d) g2, (e) ring spacing x2.*

- p1 moves the lower notch (f1) and p2 moves the upper notch (f2), so each ring sets its own band.
- A larger g2 (0.3 mm) removes the upper notch, and x2 shifts it.
- Feed width has only a weak effect on the notches.
- The first design looked better than it was (S21 near −45 dB) because of a coarse mesh and a small port. After refining the mesh and enlarging the port, the notches settled at about −29 and −23 dB.

## Simulation setup

- **Software:** CST Studio Suite 2026, Education (student) edition.
- **Solver:** Frequency Domain, tetrahedral mesh with adaptive refinement, 1 to 6 GHz, open (add space) boundaries, waveguide ports at both ends of the feed.
- **Student-version limit:** the mesh size is capped, so a coarse mesh was used. Results carry about ±0.02 to 0.03 GHz of simulation noise, which is as large as the whole sugar shift.
- **Note on the CST file:** the original model made in the lab is no longer available. The CST model attached to this project is the model we built for DA1 in the student version.

## Test samples and dielectric data

16 samples: air (unloaded reference), fresh milk, synthetic milk, water added to milk (10, 20, 35, 50, 70, 85, 100 ml), sugar added to milk (2, 5, 10, 15 g) and a low and a high electrolyte mixture.

| Sample | εr′ | εr″ | tan δ |
| --- | --- | --- | --- |
| Fresh milk | 69.55 | 10.06 | 0.145 |
| Water 20 / 50 / 100 ml | 73.86 / 76.36 / 77.88 | 10.91 / 10.82 / 10.66 | 0.148 / 0.142 / 0.137 |
| Sugar 5 g / 10 g | 63.05 / 63.20 | 13.61 / 14.14 | 0.216 / 0.224 |
| Synthetic milk | 72.09 | 14.71 | 0.204 |
| Electrolyte mixture, low / high | 76.42 / 73.96 | 11.25 / 20.29 | 0.147 / 0.274 |

- Milk, water, sugar and synthetic milk (2.45 GHz): Malame et al. (2014).
- Electrolyte mixture: Eldamak et al., *Biosensors* (2020). This is a sweat-type mixture of several salts and urea, **not pure urea**.
- Amounts not in the papers (for example water 35, 70, 85 ml and sugar 2, 15 g) are linear interpolations between published points. Water 10 ml used an extrapolated value.
- An ethanol–water series was also tried, but its results did not follow permittivity, so it is not used.

## Results

All readings are Band 1 (first ring). Δf1 = f1 − 2.1677 GHz (air); Q1 = f1 / (−3 dB bandwidth).

| Sample | εr′ | f1 (GHz) | Δf1 (MHz) | Depth (dB) | Q1 |
| --- | --- | --- | --- | --- | --- |
| Air (unloaded) | 1.00 | 2.1677 | 0.0 | −29.68 | 34.4 |
| Fresh milk | 69.55 | 1.9928 | −174.9 | −27.87 | 17.8 |
| Synthetic milk | 72.09 | 1.9832 | −184.5 | −19.55 | 6.8 |
| Water 10 ml | 73.03 | 1.9814 | −186.3 | −22.82 | 9.4 |
| Water 20 ml | 73.86 | 1.9590 | −208.7 | −22.82 | 9.4 |
| Water 35 ml | 75.11 | 1.9567 | −211.0 | −22.52 | 8.8 |
| Water 50 ml | 76.36 | 1.9110 | −256.7 | −22.58 | 10.0 |
| Water 70 ml | 76.97 | 1.9432 | −224.5 | −20.54 | 6.9 |
| Water 85 ml | 77.42 | 1.9300 | −237.7 | −21.99 | 8.3 |
| Water 100 ml | 77.88 | 1.9220 | −245.7 | −21.81 | 8.3 |
| Sugar 2 g | 62.96 | 2.1000 | −67.7 | −23.24 | 10.4 |
| Sugar 5 g | 63.05 | 2.1100 | −57.7 | −23.31 | 10.2 |
| Sugar 10 g | 63.20 | 2.1087 | −59.0 | −22.64 | 9.6 |
| Sugar 15 g | 63.35 | 2.0770 | −90.7 | −23.08 | 11.3 |
| Electrolyte mixture, low | 76.42 | 1.9366 | −231.1 | −21.31 | 7.8 |
| Electrolyte mixture, high | 73.96 | 1.9639 | −203.8 | −16.22 | 4.6 |

<p>
  <img src="images/s21_fresh_milk.png" width="48%" alt="S21 with fresh milk">
  <img src="images/s11_unloaded.png" width="48%" alt="Unloaded S11">
</p>

*Left: S21 with fresh milk (f1 about 2.0 GHz, −27.9 dB). Right: unloaded S11, close to 0 dB at the notches because the resonators reflect power there.*

![f1 versus amount added](images/f1_vs_amount.png)

*f1 versus the amount of water or sugar added. Dashed line: air. Dotted line: fresh milk.*

Main observations:

- Loading with milk lowers f1 from 2.168 GHz to about 1.99 GHz and roughly halves Q1, because milk is lossy.
- Water lowers f1 further, by about 0.66 MHz for every ml added (1.981 to 1.922 GHz from 10 to 100 ml).
- Sugar leaves f1 close to the air value (about 2.08 to 2.11 GHz), so water and sugar move f1 in clearly different ways.
- A higher electrolyte level makes the notch shallower (−21.3 to −16.2 dB) and much broader.
- Normalised sensitivity S = (Δf / f0) / Δεr′ × 100 % is about 0.12 %/εr′ for milk, 0.15 for 100 ml water and 0.04 for 5 g sugar. This is lower than some published dual-band sensors; the aim here was to tell adulterants apart, not to reach the highest sensitivity.

<details>
<summary>Far-field note (not a resonance)</summary>

<img src="images/farfield_3p5GHz.png" width="300" alt="Far-field at 3.5 GHz">

Far-field at 3.5 GHz from CST (not at a resonance of the sensor): realized gain −27.1 dBi, radiation efficiency −25.0 dB, directivity 3.29 dBi. The sensor is a guided-wave S21 device, not an antenna, so low gain and efficiency are expected.

</details>

## Why only the first ring (about 2.2 GHz)

All readings in the dataset use the **first ring only**, at about 2.2 GHz. Published dielectric data for milk and its adulterants is plentiful around 2.45 GHz (the ISM band) and scarce near the second ring's 4.5 GHz. Using the first ring let every sample have a literature-backed permittivity. The second ring is part of the design and was tuned and simulated unloaded, but it is not used in the dataset or the model. Using it is future work.

## Machine learning and Streamlit app

> **To be added:** model description, validation results, how to run the app.

App screenshots:

<p>
  <img src="images/app_forward.png" width="48%" alt="Forward mode of the Streamlit app">
  <img src="images/app_inverse.png" width="48%" alt="Inverse mode of the Streamlit app">
</p>

*Left: forward mode (sensor readings in, adulterant and amount out). Right: inverse mode (adulterant and amount in, expected sensor response and S21 curve out).*

## Repository structure

```
.
├── README.md
├── GEOMETRY.md            # all sensor coordinates and set-up
├── sensor_geometry.csv    # same coordinates, machine-readable
├── images/                # figures used in this README
├── cst/                   # add the CST project file here
├── data/                  # add the dataset here
└── ml/                    # add the model and Streamlit app here
```

## Limitations and future work

- Everything is simulated; there is no VNA measurement of a fabricated sensor yet.
- The dataset is small (16 samples), the permittivities come from the literature, and some amounts are interpolated.
- Only the first ring is used; the second ring is not in the model.
- No published dielectric data was found for urea alone or for detergent, so those adulterants are not covered.
- Concentration estimates are approximate. More samples at different amounts, simulated with identical mesh settings, would help.
- Next steps: fabricate the sensor on FR-4 and measure it with a VNA, add the second ring to the model, and validate concentration on held-out samples.

## Team

Presented by:

- Abishek Tamilarasan (24BEC1087)
- N Nithya Shapthan (24BEC1153)
- Niraimathiyan K (24BEC1397)
- Jaya Aswin JV (24BEC1552)

Course: BECE305L – Antenna and Microwave Engineering, Slot A2+tA2, VIT Chennai.
Faculty guide: Dr. Prasad Jones Christy Dass S.

## References

1. H. Mao and G. Liu, "Dual-Mode Microwave Sensor Based on BF-CSRR for Liquid Permittivity Measurement," *IEEE Sensors Journal*, vol. 25, no. 19, 2025.
2. C. Wang, X. Liu, L. Gan and Q. Cai, "A Dual-Band Non-destructive Dielectric Measurement Sensor Based on Complementary Split-Ring Resonator," *Frontiers in Physics*, vol. 9, 669707, 2021.
3. Malame et al., "Microwave Reflectometry Based Electrical Characterization of Milk for Adulteration Detection," 2014.
4. Eldamak et al., "Study of the Dielectric Properties of Artificial Sweat Mixtures at Microwave Frequencies," *Biosensors*, vol. 10, no. 6, 2020.

## License

*Add a license file (for example MIT) before making the repository public.*
