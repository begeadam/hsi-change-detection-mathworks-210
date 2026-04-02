# Change Detection in Hyperspectral Imagery (MathWorks Challenge #210)

[![MATLAB](https://img.shields.io/badge/MATLAB-2023b%2B-blue.svg)](https://www.mathworks.com/products/matlab.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the solution for the **MATLAB and Simulink Challenge Project Hub** task: [Change Detection in Hyperspectral Imagery (Project #210)](https://github.com/mathworks/MATLAB-Simulink-Challenge-Project-Hub/tree/main/projects/Change%20Detection%20in%20Hyperspectral%20Imagery).

## 🚀 Overview
Detecting changes on the Earth's surface between two hyperspectral/multispectral images taken at different times is critical for urban sprawl tracking and environmental monitoring. This project proposes a **Machine Learning-driven (Random Forest)** approach leveraging the native **Image Processing Toolbox™ Hyperspectral Imaging Library** to overcome drastic phenological (seasonal) changes, mixed pixels, and global illumination shifts.

## 🎯 Project Impact & Objectives
- **Problem:** Slow manual tracking of rapid urban sprawl, new constructions, and infrastructural changes.
- **Impact:** Automate hyperspectral change detection to track urban development, illegal constructions, and environmental alterations efficiently.
- **Expertise Gained:** Machine Learning, Hyperspectral Imaging, Image Processing, Spectral Angle Mapping (SAM), Random Forests.

## 📖 Algorithm Overview
Instead of classical thresholding, this pipeline incorporates a robust 17-dimensional feature space to create a highly resilient Change Detection classifier. Our approach successfully addresses the challenge's core problem:
1. **Multiple Changes & Mixed Pixels:** A Random Forest classifier trained on a strictly balanced subset of pixels separates true urbanization from mere agricultural harvests using local standard deviation (texture) metrics.
2. **Hyperspectral Engine:** Images are constructed as `hypercube` objects, matching the dataset wavelengths to strictly adhere to the Hyperspectral Imaging architecture recommendations.
3. **Radiometric Harmonization:** Phase 1 implements exact `imhistmatch` to align the baseline reflectance properties of T2 to T1, severely dampening light-angle distortions.
4. **17-Feature Data Cube:** For each pixel, the model computes:
   - Absolute PCA mappings
   - Spectral Angle Mapper (SAM) arrays
   - Structural Fusion maps
   - Explicit channel-wise spectral differences (10 native bands)
   - Computed spatial texture and advanced indices (`NDVI`, `NDWI` deltas)

## 📂 Repository Structure
```text
├── src/
│   └── unified_hsi_pipeline.m    # The main, self-contained MATLAB script
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
Ensure you have MATLAB installed (recommended R2023b or later) alongside the following toolboxes:
- Image Processing Toolbox
- Image Processing Toolbox™ Hyperspectral Imaging Library
- Statistics and Machine Learning Toolbox

### 3. Run the Pipeline
Open MATLAB, set your working directory to the project root, and execute:
```matlab
run('src/unified_hsi_pipeline.m')
```

## 📊 Results Summary
By utilizing robust index differences (NDVI) and spatial textures, the algorithm minimizes seasonal agricultural noise while keeping building mapping tight:
- **Las Vegas:** 0.69 F1-Score (Extreme urban construction mapped almost perfectly against bare sand)
- **Dubai:** 0.49 F1-Score
- **Milano:** 0.26 F1-Score *(A significant achievement over classical PCA/K-Means that severely failed due to cloud and localized Phenology).*

## 🔬 Limitations & Future Work
- Mixed pixel challenges are still present in very heterogeneous or agriculture-heavy terrains (e.g., Valencia).
- Future work: Extend the pipeline to 3D CNN models or Transformers for significantly better spatial-temporal consistency.

## 🤝 Contributing
Contributions are welcome! Feel free to open an Issue or submit a Pull Request if you have ideas on improving the feature extractor or adding new deep learning backends.

## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.
