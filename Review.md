https://github.com/mathworks/MATLAB-Simulink-Challenge-Project-Hub/tree/main/projects/Change%20Detection%20in%20Hyperspectral%20Imagery


MATLAB/Simulink Challenge Submission Review - Change Detection in Hyperspectral Imagery

Dear Ádám,

Thank you for your submission to the MATLAB/Simulink Challenge Project Hub for the project "Change Detection in Hyperspectral Imagery".

We apologize for the delayed response. We appreciate your patience while we carefully reviewed your work.

After reviewing your submission, we believe it shows promise but requires some revisions before it can be accepted.

Overview
The submission consists of two files: unifiedhsipipeline.m (the main automated pipeline) and HSI.mlx (an interactive Live Script with visualization). The pipeline loads Sentinel-2 bands, performs histogram matching for radiometric correction, PCA-based co-registration, extracts a 17-dimensional feature vector per pixel (PCA map, SAM map, fusion map, 10 spectral differences, NDVI/NDWI deltas, texture features), trains a 50-tree Random Forest classifier with stratified sampling, and evaluates on test cities. The Live Script adds an Otsu-thresholding approach and K-means clustering for change-type classification.

Items to Address

Critical: Does not use the Image Processing Toolbox Hyperspectral Imaging Library as required by the project description
Major: No spectral unmixing or endmember analysis to address mixed pixel problem (project requirement #3)
Major: No hierarchical spectral analysis as suggested in project steps
Minor: Array preallocation warnings (Xtrain, Ytrain grow in loop)
Minor: Hungarian-language comments in Live Script
Minor: No sample results/figures included in the repository
Minor: Dataset instructions could be clearer about the exact expected folder structure
Suggested Changes

Integrate the Hyperspectral Imaging Library: Replace manual data cube loading with hypercube objects. Use library functions like sam, ndvi, and spectral matching instead of manual implementations. This is explicitly required by the project description.
Add spectral unmixing for mixed pixels: Use endmember extraction functions (ppi, nfindr, or fippi) and linear/nonlinear unmixing from the library to address project requirement #3 about mixed pixel ambiguity.
Implement hierarchical change analysis: Add hierarchical clustering or spectral matching (as in reference [1] and [4]) to automatically categorize change types, rather than simple K-means.
Preallocate training arrays: Estimate total training samples and preallocate Xtrain and Ytrain before the city loop to improve performance.
Translate Hungarian comments to English in the Live Script for accessibility.
Include sample results: Add a results/ folder with example output figures or a pre-computed change map so reviewers can verify performance without downloading the 20GB dataset.
Add input validation: Verify that loaded bands match expected Sentinel-2 band order before computing NDVI/NDWI with hardcoded indices.
If you have any questions or would like to discuss this feedback, please don't hesitate to reach out.

Best regards,
MATLAB/Simulink Challenge Review Team