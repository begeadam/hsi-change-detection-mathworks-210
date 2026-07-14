# Change Detection in Satellite Imagery (MATLAB University Project)

[![MATLAB](https://img.shields.io/badge/MATLAB-2023b%2B-blue.svg)](https://www.mathworks.com/products/matlab.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains my final university project for the MATLAB course. The project is inspired by and serves as a solution to the **MATLAB and Simulink Challenge Project Hub** task: [Change Detection in Hyperspectral Imagery (Project #210)](https://github.com/mathworks/MATLAB-Simulink-Challenge-Project-Hub/tree/main/projects/Change%20Detection%20in%20Hyperspectral%20Imagery).

## 🌍 Motivation & Background
For our final MATLAB course project, we were given a list of topics to choose from. I selected this specific project because of my strong interest in space engineering, particularly satellite imagery and ground mapping. Having recently discovered the innovative work of Earth observation companies like Planet Labs, I was highly motivated to learn how to apply MATLAB to real-world remote sensing problems.

## 🚀 Overview (V2 Update)
Detecting changes on the Earth's surface between two hyperspectral/multispectral images taken at different times is critical for urban sprawl tracking and environmental monitoring. This project proposes a **Machine Learning-driven (Random Forest)** approach to overcome drastic phenological (seasonal) changes, mixed pixels, and global illumination shifts. 

**V2 Major Updates:** The pipeline now strictly adheres to MathWorks' official standards by integrating the **Hyperspectral Imaging Library**. It introduces Spectral Unmixing (N-FINDR + FCLS) to handle mixed pixels and features a **Hierarchical Spectral Analysis** module that automatically categorizes the detected changes into dynamic sub-types (e.g., vegetation loss, construction).

## 🎯 Project Impact & Objectives
- **Problem:** Slow manual tracking of rapid urban sprawl, new constructions, and infrastructural changes.
- **Impact:** Automate satellite change detection to track urban development and environmental alterations efficiently.
- **Expertise Gained:** Machine Learning, Remote Sensing, Image Processing, Spectral Angle Mapping (SAM), Random Forests.

## 📖 Algorithm Overview
Instead of classical thresholding, this pipeline incorporates a robust 22-dimensional feature space to create a highly resilient Change Detection classifier. Our approach successfully addresses the core problems:
1. **Multiple Changes & Mixed Pixels (Spectral Unmixing):** A Random Forest classifier trained on a strictly balanced subset of pixels separates true urbanization from mere agricultural harvests using local standard deviation (texture) metrics. We also extract 5 endmembers via N-FINDR and compute Fully Constrained Least Squares (FCLS) abundances to solve mixed-pixel distortions.
2. **Library-Native Engine:** Images are processed natively using the `hypercube` class. Native toolbox functions (`sam`, `ndvi`, `nfindr`, `estimateAbundanceLS`) are heavily utilized.
3. **Radiometric Harmonization:** Phase 1 implements exact `imhistmatch` to align the baseline reflectance properties of T2 to T1, severely dampening light-angle distortions.
4. **22-Feature Data Cube:** For each pixel, the model computes:
   - Absolute PCA mappings
   - Spectral Angle Mapper (SAM) difference map
   - Structural Fusion maps
   - Explicit channel-wise spectral differences (10 native Sentinel-2 bands)
   - Computed spatial texture and advanced indices (`NDVI`, `NDWI` deltas)
   - Endmember abundance differences (5 dimensions)
5. **Hierarchical Categorization:** Post-classification, the changed pixels undergo K-Means and Ward's Hierarchical Clustering. The `evalclusters` function dynamically selects the optimal number of change types using the Calinski-Harabasz index.

## 📂 Repository Structure
```text
├── unified_hsi_pipeline_V2.m     # The main V2 MATLAB script
├── unified_hsi_pipeline.m        # Legacy V1 pipeline
├── HSI.mlx                       # A live script for interactive use 
├── Review.md                     # Code review and task assignments
├── results/                      # Output directory for change category maps
├── README.md                     # Project documentation
└── LICENSE                       # MIT License
```
*(Note: The actual dataset is ignored via `.gitignore` to prevent massive uploads.)*

## 📥 Dataset & Installation

### 1. Download the Dataset
This project was benchmarked on the **ONERA Satellite Change Detection Dataset (OSCD)**. 
Due to storage constraints, the dataset is **not included** in this repository. 
- Download the dataset from [IEEE DataPort](https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection).
- Extract the files so that the following folder structure is created in the project root:

```text
Project_Root/
├── Onera Satellite Change Detection dataset - Images/
│   ├── train.txt
│   ├── test.txt
│   ├── aguasclaras/
│   │   ├── imgs_1_rect/    ← Contains B01.tif .. B09.tif (Time 1)
│   │   └── imgs_2_rect/    ← Contains B01.tif .. B09.tif (Time 2)
│   ├── dubai/
│   │   ├── imgs_1_rect/
│   │   └── imgs_2_rect/
│   ├── ... (other cities)
│   ├── Onera Satellite Change Detection dataset - Train Labels/
│   │   └── <city>/cm/cm.png
│   └── Onera Satellite Change Detection dataset - Test Labels/
│       └── <city>/cm/cm.png
├── unified_hsi_pipeline_V2.m
└── ...
```

### 2. Set Up the Environment
Ensure you have MATLAB (R2023b or later) installed alongside the following toolboxes:
- Image Processing Toolbox
- Statistics and Machine Learning Toolbox
- **Image Processing Toolbox Hyperspectral Imaging Library** (Install via MATLAB Add-On Explorer)

### 3. Run the Pipeline
Open MATLAB, set your working directory to the project root, and execute:
```matlab
run('unified_hsi_pipeline_V2.m')
```
*Note: The script caches the trained Random Forest into `RF_Model.mat` to drastically speed up future executions.*

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
