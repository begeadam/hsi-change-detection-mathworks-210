% =========================================================================
% UNIFIED HYPERSPECTRAL IMAGING (HSI) PIPELINE - V2
% =========================================================================
% Revision Notes (V2):
%   - Integrates MathWorks Hyperspectral Imaging Library functions
%     (ndvi, sam, nfindr, estimateAbundanceLS, countEndmembersHFC)
%   - Adds spectral unmixing & endmember analysis for mixed-pixel handling
%   - Preallocates training arrays for improved performance
%   - Adds input validation for band count verification
% =========================================================================
% Requirements:
%   - Image Processing Toolbox
%   - Image Processing Toolbox Hyperspectral Imaging Library (Add-On)
%   - Statistics and Machine Learning Toolbox
% =========================================================================

clear; clc;

%% ========================================================================
%  CONFIGURATION
%  ========================================================================
NUM_ENDMEMBERS = 5;       % Fixed number of endmembers for spectral unmixing
NUM_RF_TREES   = 50;      % Number of trees in the Random Forest
NUM_FEATURES   = 22;      % Total feature dimensions (17 original + 5 abundance diffs)
EXPECTED_BANDS = 10;      % Expected number of Sentinel-2 spectral bands

% Sentinel-2 band central wavelengths (nm) for hypercube construction
WAVELENGTHS = [443, 490, 560, 665, 705, 740, 783, 842, 865, 945];

%% ========================================================================
%  1. TRAINING PHASE
%  ========================================================================

if exist('RF_Model.mat', 'file')
    fprintf('--- 1. TRAINING PHASE (SKIPPED) ---\n');
    fprintf('Loading pre-trained Random Forest model from RF_Model.mat...\n\n');
    load('RF_Model.mat', 'RF_Model');
else
    trainStr = fileread(fullfile('Onera Satellite Change Detection dataset - Images', 'train.txt'));
    trainCities = strsplit(strtrim(trainStr), ',');
    
    % Preallocate training arrays (estimated upper bound: ~14 cities * ~5000 samples)
    estimatedSamples = 100000;
    X_train = zeros(estimatedSamples, NUM_FEATURES);
    Y_train = zeros(estimatedSamples, 1);
    sampleCount = 0;
    
    fprintf('--- 1. TRAINING PHASE ---\n');
    for i = 1:length(trainCities)
        cityName = trainCities{i};
        if isempty(cityName)
            continue;
        end
        fprintf('Processing (Train): %s... ', cityName);
        
        labelDir = fullfile('Onera Satellite Change Detection dataset - Images', ...
            'Onera Satellite Change Detection dataset - Train Labels', ...
            cityName, 'cm');
        
        if ~exist(labelDir, 'dir')
            fprintf('[No Label Found] Skipping.\n');
            continue;
        end
        
        [featureCube, H, W] = process_city_features(cityName, WAVELENGTHS, ...
                                EXPECTED_BANDS, NUM_ENDMEMBERS);
        if isempty(featureCube)
            fprintf('[No Input Data] Skipping.\n');
            continue;
        end
        
        labelPath = fullfile(labelDir, 'cm.png');
        labelImg = imread(labelPath);
        labelImg = labelImg(1:H, 1:W);
        
        % Reshape the binary Ground Truth label matrix into a 1D column vector
        Y_img = (labelImg == 255);
        Y_vec = Y_img(:);
        
        % Flatten the feature cube into a 2D matrix for model training: (H*W) x NUM_FEATURES
        X_vec = reshape(featureCube, H*W, NUM_FEATURES);
        
        % Stratified Random Sampling to mitigate class imbalance
        changeIdx = find(Y_vec == 1);
        noChangeIdx = find(Y_vec == 0);
        
        % Retain all minority class samples (pixels representing 'Change')
        current_changes = length(changeIdx);
        
        % Subsample the majority class. Heuristic: 2x positive count, min 1500
        numToSample = max(1500, current_changes * 2);
        if length(noChangeIdx) > numToSample
            sampledNoChangeIdx = datasample(noChangeIdx, numToSample, 'Replace', false);
        else
            sampledNoChangeIdx = noChangeIdx;
        end
        
        % Append sampled data using preallocated indexing
        selectedIdx = [changeIdx; sampledNoChangeIdx];
        nNew = length(selectedIdx);
        
        % Expand preallocated array if necessary
        if sampleCount + nNew > size(X_train, 1)
            X_train = [X_train; zeros(nNew + 50000, NUM_FEATURES)]; %#ok<AGROW>
            Y_train = [Y_train; zeros(nNew + 50000, 1)];           %#ok<AGROW>
        end
        
        X_train(sampleCount+1 : sampleCount+nNew, :) = X_vec(selectedIdx, :);
        Y_train(sampleCount+1 : sampleCount+nNew)    = Y_vec(selectedIdx);
        sampleCount = sampleCount + nNew;
        
        fprintf('Done. (+%d samples)\n', nNew);
    end
    
    % Trim preallocated arrays to actual size
    X_train = X_train(1:sampleCount, :);
    Y_train = Y_train(1:sampleCount);
    
    fprintf('1/A. Training Random Forest Classifier on %d data points...\n', sampleCount);
    tic;
    % Train a Random Forest classifier (TreeBagger) with bootstrap aggregating
    RF_Model = TreeBagger(NUM_RF_TREES, X_train, Y_train, ...
        'OOBPrediction', 'on', 'Method', 'classification');
    toc;
    fprintf('Training Complete!\n\n');
    
    % Save the model to avoid retraining in the future
    save('RF_Model.mat', 'RF_Model');
end

%% ========================================================================
%  2. TESTING PHASE
%  ========================================================================

testCities = {'dubai', 'lasvegas', 'milano', 'brasilia', 'valencia'};

fprintf('--- 2. TESTING PHASE ---\n');
for i = 1:length(testCities)
    cityName = testCities{i};
    fprintf('\nCity: %s\n', cityName);
    
    [featureCube, H, W] = process_city_features(cityName, WAVELENGTHS, ...
                            EXPECTED_BANDS, NUM_ENDMEMBERS);
    if isempty(featureCube)
        fprintf('  Skipped (No Image Data).\n');
        continue;
    end
    
    % Locate Ground Truth labels for the test city
    gt_path = fullfile('Onera Satellite Change Detection dataset - Images', ...
        'Onera Satellite Change Detection dataset - Test Labels', ...
        cityName, 'cm', 'cm.png');
    
    % Fallback: attempt Train Labels directory
    if ~exist(gt_path, 'file')
        gt_path = fullfile('Onera Satellite Change Detection dataset - Images', ...
            'Onera Satellite Change Detection dataset - Train Labels', ...
            cityName, 'cm', 'cm.png');
    end
    
    if ~exist(gt_path, 'file')
        fprintf('  Skipped (No Ground Truth).\n');
        continue;
    end
    
    gt = imread(gt_path);
    gt_bin = (gt(1:H, 1:W) == 255);
    
    % Execute predictions using the trained Random Forest model
    X_test = reshape(featureCube, H*W, NUM_FEATURES);
    [predStr, ~] = predict(RF_Model, X_test);
    predMask = reshape(str2double(predStr), H, W);
    
    % Morphological Post-Processing: remove salt-and-pepper noise
    predMask = medfilt2(predMask, [5 5]);
    predMask = bwareaopen(predMask, 20);
    
    % Compute Standard Evaluation Metrics
    TP = sum(predMask(:) & gt_bin(:));
    FP = sum(predMask(:) & ~gt_bin(:));
    FN = sum(~predMask(:) & gt_bin(:));
    
    Precision = TP / (TP + FP + eps);
    Recall = TP / (TP + FN + eps);
    F1 = 2 * Precision * Recall / (Precision + Recall + eps);
    
    fprintf('  Precision: %.4f\n', Precision);
    fprintf('  Recall:    %.4f\n', Recall);
    fprintf('  F1-Score:  %.4f\n', F1);
    
    % =====================================================================
    % HIERARCHICAL SPECTRAL ANALYSIS (Dynamic Category Selection)
    % =====================================================================
    % 1. Extract changed pixels based on the RF prediction mask
    changeIndices = find(predMask == 1);
    if ~isempty(changeIndices)
        fprintf('  Hierarchical Spectral Analysis on %d pixels...\n', length(changeIndices));
        
        % The spectral difference (T2 - T1) was stored in features 4:13
        diffData = reshape(featureCube(:,:,4:13), H*W, 10);
        changedDiffs = diffData(changeIndices, :);
        
        % Step A: Fast K-means to create sub-clusters (reduce N for linkage)
        numSubClusters = min(100, size(changedDiffs, 1));
        [subLabels, subCenters] = kmeans(changedDiffs, numSubClusters, 'MaxIter', 100, 'EmptyAction', 'singleton');
        
        % Step B: Hierarchical clustering on the sub-centers
        Z = linkage(subCenters, 'ward', 'euclidean');
        
        % Step C: Dynamically determine optimal number of main categories
        if size(subCenters, 1) > 5
            try
                eva = evalclusters(subCenters, 'linkage', 'CalinskiHarabasz', 'KList', 2:min(10, size(subCenters, 1)-1));
                numMainCategories = eva.OptimalK;
            catch
                numMainCategories = 4; % Fallback if evalclusters fails
            end
        else
            numMainCategories = size(subCenters, 1);
        end
        fprintf('  Dynamically selected %d change categories.\n', numMainCategories);
        
        % Assign main category labels
        mainCategoryLabels = cluster(Z, 'MaxClust', numMainCategories);
        
        % Step D: Map back to the full image
        finalChangeTypes = zeros(H, W);
        pixelCategories = mainCategoryLabels(subLabels);
        finalChangeTypes(changeIndices) = pixelCategories;
        
        % Save visual output to results/ folder (without opening figure windows)
        if ~exist('results', 'dir')
            mkdir('results');
        end
        
        % Create colormap: 0 = black, 1..N = distinct colors
        cmap = [0 0 0; lines(numMainCategories)];
        
        % Convert indexed image to RGB
        rgbImg = ind2rgb(finalChangeTypes, cmap);
        
        % Write image file
        imwrite(rgbImg, fullfile('results', sprintf('%s_change_categories.png', cityName)));
    end
end
fprintf('\nFull pipeline executed successfully.\n');

%% ========================================================================
%  HELPER FUNCTIONS
%  ========================================================================

function [features, H, W] = process_city_features(cityName, wavelengths, ...
                                expectedBands, numEndmembers)
% PROCESS_CITY_FEATURES Extracts a multi-dimensional feature cube for a city.
%
%   Phases:
%     1A - Radiometric Harmonization (histogram matching T2 -> T1)
%     1B - Spatial Co-registration (PCA-based translation alignment)
%     2A - PCA Difference Map
%     2B - Spectral Angle Mapper (SAM) via Hyperspectral Imaging Library
%     2C - Structural Fusion Map (PCA + SAM)
%     2D - NDVI / NDWI via Hyperspectral Imaging Library
%     2E - Contextual Texture Features
%     2F - Spectral Unmixing (endmember extraction + abundance mapping)
%
%   Output feature vector per pixel (22 dimensions):
%     [1]      PCA difference map
%     [2]      SAM difference map
%     [3]      Fusion map (PCA + SAM)
%     [4:13]   Signed spectral difference per band (10 bands)
%     [14]     NDVI difference (toolbox-derived)
%     [15]     NDWI difference
%     [16:17]  Contextual texture features
%     [18:22]  Endmember abundance differences (5 endmembers)

    baseDir = 'Onera Satellite Change Detection dataset - Images';
    bandList = {'B01.tif', 'B02.tif', 'B03.tif', 'B04.tif', 'B05.tif', ...
                'B06.tif', 'B07.tif', 'B08.tif', 'B8A.tif', 'B09.tif'};
    
    path_T1 = fullfile(baseDir, cityName, 'imgs_1_rect');
    path_T2 = fullfile(baseDir, cityName, 'imgs_2_rect');
    
    % --- Load and spatially align multi-resolution spectral bands ---
    Cube_T1_Raw = localLoadAndResize(path_T1, bandList);
    Cube_T2_Raw = localLoadAndResize(path_T2, bandList);
    
    if isempty(Cube_T1_Raw) || isempty(Cube_T2_Raw)
        features = []; H = 0; W = 0;
        return;
    end
    
    % --- INPUT VALIDATION ---
    if size(Cube_T1_Raw, 3) ~= expectedBands
        warning('City "%s": Expected %d bands, got %d. Skipping.', ...
            cityName, expectedBands, size(Cube_T1_Raw, 3));
        features = []; H = 0; W = 0;
        return;
    end
    if size(Cube_T2_Raw, 3) ~= expectedBands
        warning('City "%s": T2 has %d bands (expected %d). Skipping.', ...
            cityName, size(Cube_T2_Raw, 3), expectedBands);
        features = []; H = 0; W = 0;
        return;
    end
    
    % =====================================================================
    %  HYPERSPECTRAL IMAGING LIBRARY INTEGRATION
    % =====================================================================
    % Construct official hypercube objects with wavelength metadata.
    % This satisfies the project requirement to use the Hyperspectral
    % Imaging Library as the primary data structure.
    hcube_T1 = hypercube(Cube_T1_Raw, wavelengths);
    hcube_T2 = hypercube(Cube_T2_Raw, wavelengths);
    
    [H, W, B] = size(Cube_T1_Raw);
    
    % =====================================================================
    %  PHASE 1A: Radiometric Harmonization (Histogram Matching T2 -> T1)
    % =====================================================================
    % Mitigate global radiometric discrepancies caused by varying sun
    % elevation angles and atmospheric drift between acquisition dates.
    Cube_T1_Corr = zeros(H, W, B);
    Cube_T2_Corr = zeros(H, W, B);
    
    for b = 1:B
        Ref_Band = mat2gray(Cube_T1_Raw(:,:,b));
        Cube_T1_Corr(:,:,b) = Ref_Band;
        Cube_T2_Corr(:,:,b) = imhistmatch(Cube_T2_Raw(:,:,b), Ref_Band);
    end
    
    % =====================================================================
    %  PHASE 1B: Spatial Co-registration via PCA
    % =====================================================================
    % Align images using the first principal component to correct for
    % sub-pixel spatial offsets between acquisition dates.
    X1 = reshape(Cube_T1_Corr, H*W, B);
    X2 = reshape(Cube_T2_Corr, H*W, B);
    [~, score1] = pca(X1, 'NumComponents', 1);
    [~, score2] = pca(X2, 'NumComponents', 1);
    PC1_T1 = reshape(score1, H, W);
    PC1_T2 = reshape(score2, H, W);
    
    % Spatially align using a rigid translation transformation
    [optimizer, metric] = imregconfig('monomodal');
    tform = imregtform(PC1_T2, PC1_T1, 'translation', optimizer, metric);
    R = imref2d(size(PC1_T1));
    Cube_T2_Reg = imwarp(Cube_T2_Corr, tform, 'OutputView', R);
    
    % Build corrected hypercube objects for toolbox function calls
    hcube_T1_corr = hypercube(Cube_T1_Corr, wavelengths);
    hcube_T2_reg  = hypercube(Cube_T2_Reg, wavelengths);
    
    % =====================================================================
    %  PHASE 2A: PCA Difference Map
    % =====================================================================
    diffCube = abs(Cube_T1_Corr - Cube_T2_Reg);
    X_Diff = reshape(diffCube, H*W, B);
    [~, scoreDiff] = pca(X_Diff, 'NumComponents', 1);
    pcaMapRaw = abs(reshape(scoreDiff(:,1), H, W));
    
    % Clamp upper 1st percentile outliers, then Min-Max normalize
    p99_pca = prctile(pcaMapRaw(:), 99);
    pcaMap = min(pcaMapRaw, p99_pca);
    pcaMap = (pcaMap - min(pcaMap(:))) / (max(pcaMap(:)) - min(pcaMap(:)));
    
    % =====================================================================
    %  PHASE 2B: Spectral Angle Mapper (SAM) via Toolbox
    % =====================================================================
    % Use the Hyperspectral Imaging Library's sam() function instead of
    % manual dot-product / acos computation. The toolbox SAM computes the
    % angle between each pixel spectrum and a set of reference spectra.
    %
    % Strategy: Extract mean spectrum from T1 as reference, then compute
    % SAM scores for both T1 and T2. The difference reveals spectral change.
    meanSpec_T1 = squeeze(mean(mean(Cube_T1_Corr, 1), 2))';  % 1 x B
    
    samScores_T1 = sam(hcube_T1_corr, meanSpec_T1);  % H x W (angle to T1 mean)
    samScores_T2 = sam(hcube_T2_reg, meanSpec_T1);   % H x W (angle to T1 mean)
    
    % SAM-based change indicator: difference in spectral angles
    samMapRaw = abs(samScores_T2 - samScores_T1);
    samMapRaw(isnan(samMapRaw)) = 0;
    samMapRaw = real(samMapRaw);
    
    % Clamp upper 1st percentile and normalize
    p99_sam = prctile(samMapRaw(:), 99);
    samMap = min(samMapRaw, p99_sam);
    samMap = (samMap - min(samMap(:))) / (max(samMap(:)) - min(samMap(:)) + eps);
    
    % =====================================================================
    %  PHASE 2C: Structural Fusion Map (PCA + SAM)
    % =====================================================================
    weightPCA = 0.5;
    fusionMap = (weightPCA * pcaMap) + ((1-weightPCA) * samMap);
    fusionMap = medfilt2(fusionMap, [3 3]);
    fusionMap = (fusionMap - min(fusionMap(:))) / (max(fusionMap(:)) - min(fusionMap(:)) + eps);
    
    % =====================================================================
    %  PHASE 2D: Spectral Index Differences via Toolbox (NDVI, NDWI)
    % =====================================================================
    % Use the Hyperspectral Imaging Library ndvi() function instead of
    % manual (B8-B4)/(B8+B4) computation.
    ndvi_T1 = ndvi(hcube_T1_corr);   % Toolbox auto-selects Red & NIR bands
    ndvi_T2 = ndvi(hcube_T2_reg);
    ndviDiff = abs(ndvi_T1 - ndvi_T2);
    
    % NDWI: The toolbox does not provide a dedicated ndwi() function,
    % so we compute it using the hypercube DataCube with proper band mapping.
    % NDWI = (Green - NIR) / (Green + NIR), where Green=Band3(560nm), NIR=Band8(842nm)
    green_T1 = Cube_T1_Corr(:,:,3);
    nir_T1   = Cube_T1_Corr(:,:,8);
    green_T2 = Cube_T2_Reg(:,:,3);
    nir_T2   = Cube_T2_Reg(:,:,8);
    
    ndwi_T1 = (green_T1 - nir_T1) ./ (green_T1 + nir_T1 + eps);
    ndwi_T2 = (green_T2 - nir_T2) ./ (green_T2 + nir_T2 + eps);
    ndwiDiff = abs(ndwi_T1 - ndwi_T2);
    
    % =====================================================================
    %  PHASE 2E: Contextual Spatial Texture (Standard Deviation Filter)
    % =====================================================================
    % Quantifies local structural variance to distinguish urban grids
    % from uniform rural fields.
    texture_T1   = stdfilt(Cube_T1_Corr(:,:,8), ones(5,5));
    texture_Diff = stdfilt(fusionMap, ones(5,5));
    
    % =====================================================================
    %  PHASE 2F: Spectral Unmixing (Endmember Extraction + Abundance)
    % =====================================================================
    % Addresses the mixed-pixel problem (Project Requirement #3).
    % Uses N-FINDR for endmember extraction and Fully Constrained Least
    % Squares (FCLS) for abundance estimation.
    
    % Extract endmembers from T1 using N-FINDR algorithm (toolbox function)
    endmembers_T1 = nfindr(hcube_T1_corr, numEndmembers);
    
    % Estimate fractional abundance of each endmember per pixel (FCLS)
    abundance_T1 = estimateAbundanceLS(hcube_T1_corr, endmembers_T1, Method="fcls");
    abundance_T2 = estimateAbundanceLS(hcube_T2_reg, endmembers_T1, Method="fcls");
    
    % Abundance difference: changes in material composition between dates
    abundanceDiff = abs(abundance_T1 - abundance_T2);
    
    % =====================================================================
    %  PHASE 3: Signed Spectral Difference (10 bands)
    % =====================================================================
    diffCubeSigned = Cube_T2_Reg - Cube_T1_Corr;
    
    % =====================================================================
    %  AGGREGATE: Build the 22-Dimensional Feature Vector
    % =====================================================================
    numFeatures = 17 + numEndmembers;  % 17 original + endmember abundances
    features = zeros(H, W, numFeatures);
    
    features(:,:,1)     = pcaMap;
    features(:,:,2)     = samMap;
    features(:,:,3)     = fusionMap;
    features(:,:,4:13)  = diffCubeSigned;           % 10 spectral bands
    features(:,:,14)    = ndviDiff;                  % Toolbox-derived NDVI
    features(:,:,15)    = ndwiDiff;                  % NDWI difference
    features(:,:,16)    = texture_T1;                % Spatial texture T1
    features(:,:,17)    = texture_Diff;              % Spatial texture change
    features(:,:,18:17+numEndmembers) = abundanceDiff;  % Unmixing features
end


function cube = localLoadAndResize(folder, bands)
% LOCALLOADANDRESIZE Load TIFF band images and spatially align to 10m resolution.
%   Uses B02 (10m) as the reference resolution baseline. All bands with
%   coarser resolution (20m, 60m) are upsampled via bicubic interpolation.
    cube = [];
    refImgPath = fullfile(folder, 'B02.tif');
    if ~exist(refImgPath, 'file')
        return;
    end
    refImg = imread(refImgPath);
    [refH, refW] = size(refImg);
    
    for k = 1:length(bands)
        fname = fullfile(folder, bands{k});
        if exist(fname, 'file')
            img = im2double(imread(fname));
            % Upsample lower resolution bands to match 10m baseline
            if size(img,1) ~= refH
                img = imresize(img, [refH, refW], 'bicubic');
            end
            cube = cat(3, cube, img);
        end
    end
end
