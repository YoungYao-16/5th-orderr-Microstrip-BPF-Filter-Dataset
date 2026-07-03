# 5th-orderr-Microstrip-BPF-Filter-Dataset
Datasets of 5th-Order Microstrip BPF with HFSS
# Reproduced Dataset for Surrogate-Based EM Optimization (5th-Order Microstrip BPF)

This repository contains a custom-reproduced, high-quality electromagnetic (EM) simulation dataset based on the methodology proposed in the paper:
> **"Surrogate-Based EM Optimization Using Neural Networks for Microwave Filter Design"** (IEICE Transactions on Electronics, 2022)

## 📌 Dataset Overview

While the original paper utilized a dataset of 6,300 training samples and 2,000 validation samples, this reproduced dataset expands and adjusts the data volume to better suit deep learning pipelines:
* **Training Set (`train.csv`):** 6,850 samples
* **Test Set (`test.csv`):** 1,777 samples

The dataset bridges the gap between the physical geometry of a symmetric fifth-order microstrip bandpass filter (BPF) and its frequency-domain electrical responses ($S$-parameters).

## 📊 Data Structure & Preview

The data is saved in standard `.csv` format. Each row represents a specific frequency sampling point for a given physical structural configuration.

### Data Columns Breakdown
* **Inputs (Geometric & Material Parameters):**
  * `1`: T-feed line length lq(**Note: `lq` was kept as a fixed value of 11.185 mm during dataset construction**).
  * `w`: Microstrip resonator width (fixed at 2.0 mm)
  * `sub_t`: Substrate thickness (1.0 mm)
  * `h`: Dielectric constant / height characteristics
  * `l1`, `l2`, `l3`: Physical lengths of the coupled resonators
  * `g1`, `g2`: Coupling gaps between adjacent resonators
* **Outputs (EM Responses across Frequencies):**
  * `Freq`: Sampling frequency point (in GHz)
  * `S11_re` / `S11_img`: Real and imaginary parts of the reflection coefficient ($S_{11}$)
  * `S21_re` / `S21_img`: Real and imaginary parts of the transmission coefficient ($S_{21}$)
  * `S11_dB` / `S21_dB`: $S$-parameters expressed in decibels (dB)

### Dataset Screenshot
Below is a preview of the structured CSV data:

![Dataset Preview](dataset_preview.png)

## 🛠️ How to Generate the Dataset (Automated Script Guide)

We provide the core python automation script (`generate_data.py`) used to produce this dataset via Ansys HFSS. It leverages PyAEDT and Python's `multiprocessing` to implement a high-throughput simulation pipeline.

### Pipeline Features:
1. **Zombile Process Cleanup**: Automatically scans and terminates residual non-graphical `ansysedt` sessions before launching to prevent license or port conflicts.
2. **Project Duplication**: Automatically duplicates independent `.aedt` project copies for each worker process to eliminate file read/write deadlock.
3. **Memory Management**: Periodically restarts the HFSS desktop session (every 150 iterations) to purge background memory accumulation during large-scale batches.

### Prerequisites
Make sure your environment has Ansys EM (v2024 R1 or later) installed on Linux/Windows, and install the required library:
```bash
