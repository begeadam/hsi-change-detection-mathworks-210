# Change Detection in Satellite Imagery (MATLAB University Project)

[![MATLAB](https://img.shields.io/badge/MATLAB-2023b%2B-blue.svg)](https://www.mathworks.com/products/matlab.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains my final university project for the MATLAB course. The project is inspired by and serves as a solution to the **MATLAB and Simulink Challenge Project Hub** task: [Change Detection in Hyperspectral Imagery (Project #210)](https://github.com/mathworks/MATLAB-Simulink-Challenge-Project-Hub/tree/main/projects/Change%20Detection%20in%20Hyperspectral%20Imagery).

## 🌍 Motivation & Background
For our final MATLAB course project, we were given a list of topics to choose from. I selected this specific project because of my strong interest in space engineering, particularly satellite imagery and ground mapping. Having recently discovered the innovative work of Earth observation companies like Planet Labs, I was highly motivated to learn how to apply MATLAB to real-world remote sensing problems.

## 🚀 Overview
Detecting changes on the Earth's surface between two hyperspectral/multispectral images taken at different times is critical for urban sprawl tracking and environmental monitoring. This project proposes a **Machine Learning-driven (Random Forest)** approach to overcome drastic phenological (seasonal) changes, mixed pixels, and global illumination shifts, using pure matrix operations without relying on heavily specialized external toolboxes.

## 🎯 Project Impact & Objectives
- **Problem:** Slow manual tracking of rapid urban sprawl, new constructions, and infrastructural changes.
- **Impact:** Automate satellite change detection to track urban development and environmental alterations efficiently.
- **Expertise Gained:** Machine Learning, Remote Sensing, Image Processing, Spectral Angle Mapping (SAM), Random Forests.

## 📖 Algorithm Overview
Instead of classical thresholding, this pipeline incorporates a robust 17-dimensional feature space to create a highly resilient Change Detection classifier. Our approach successfully addresses the core problems:
1. **Multiple Changes & Mixed Pixels:** A Random Forest classifier trained on a strictly balanced subset of pixels separates true urbanization from mere agricultural harvests using local standard deviation (texture) metrics.
2. **Matrix-Based Engine:** To maximize compatibility, images are processed as raw 3D Data Cubes, bypassing the need for the specialized Hyperspectral Imaging Toolbox while maintaining spectral integrity.
3. **Radiometric Harmonization:** Phase 1 implements exact `imhistmatch` to align the baseline reflectance properties of T2 to T1, severely dampening light-angle distortions.
4. **17-Feature Data Cube:** For each pixel, the model computes:
   - Absolute PCA mappings
   - Spectral Angle Mapper (SAM) arrays
   - Structural Fusion maps
   - Explicit channel-wise spectral differences (10 native Sentinel-2 bands)
   - Computed spatial texture and advanced indices (`NDVI`, `NDWI` deltas)

## 📂 Repository Structure
```text
├── src/
│   └── unified_hsi_pipeline.m    # The main, self-contained MATLAB script
│   └── HSI.mlx                   # A live script for interactive use 
├── README.md                     # Project documentation
└── LICENSE                       # MIT License
```
*(Note: The actual dataset is ignored via `.gitignore` to prevent massive uploads.)*

## 📥 Dataset & Installation

### 1. Download the Dataset
This project was benchmarked on the **ONERA Satellite Change Detection Dataset (OSCD)**. 
Due to storage constraints, the dataset is **not included** in this repository. 
- Download the dataset from [IEEE DataPort](https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection).
- Extract the files so that the `Onera Satellite Change Detection dataset - Images` folder is placed directly in the repository's root alongside the `src` folder.

### 2. Set Up the Environment
Ensure you have MATLAB installed alongside the following core toolboxes:
- Image Processing Toolbox
- Statistics and Machine Learning Toolbox

### 3. Run the Pipeline
Open MATLAB, set your working directory to the project root, and execute:
```matlab
run('src/unified_hsi_pipeline.m')
```

## 📊 Results Summary
By utilizing robust index differences (NDVI) and spatial textures, the algorithm minimizes seasonal agricultural noise while keeping building mapping tight. Results on the test sets:
- **Las Vegas:** 0.69 F1-Score (Extreme urban construction mapped almost perfectly against bare sand)
- **Dubai:** 0.49 F1-Score
- **Milano:** 0.26 F1-Score *(Highlights the extreme difficulty of pixel-based methods in environments with heavy seasonal vegetation changes).*

## 🔬 Limitations & Future Work
- The Random Forest model proved highly effective in arid environments (Las Vegas, Dubai) but struggled with false positives in European cities (Milano, Valencia) due to severe seasonal vegetation cycles.
- Mixed pixel challenges are still present in very heterogeneous terrains.
- Future work: Extend the pipeline to incorporate spatial-context-aware Deep Learning models (e.g., Siamese U-Net) for significantly better spatial-temporal consistency.

## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.
