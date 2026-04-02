% =========================================================================
% UNIFIED HYPERSPECTRAL IMAGING (HSI) PIPELINE
% -------------------------------------------------------------------------
% This script implements a full satellite change detection workflow:
% Phase 1: Radiometric Harmonization & Spatial Co-registration
% Phase 2: Feature Extraction (PCA, SAM, NDVI, NDWI, Contextual Texture)
% Phase 3: Supervised Machine Learning Classification (Random Forest)
% =========================================================================

clear; clc;

%% 1. Training Phase

% Load the training city list
trainStr = fileread(fullfile('Onera Satellite Change Detection dataset - Images', 'train.txt'));
trainCities = strsplit(strtrim(trainStr), ',');

X_train = [];
Y_train = [];

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
    
    [featureCube, H, W] = process_city_features(cityName);
    if isempty(featureCube)
        fprintf('[No Input Data] Skipping.\n');
        continue;
    end
    
    labelPath = fullfile(labelDir, 'cm.png');
    labelImg = imread(labelPath);
    labelImg = labelImg(1:H, 1:W);
    
    % Reshape the binary Ground Truth label matrix into a 1D column vector: (H*W) x 1
    Y_img = (labelImg == 255);
    Y_vec = Y_img(:);
    
    % Flatten the 17-dimensional feature cube into a 2D matrix for model training: (H*W) x 17
    X_vec = reshape(featureCube, H*W, 17);
    
    % Apply Stratified Random Sampling to mitigate class imbalance ('Change' vs 'No-Change')
    changeIdx = find(Y_vec == 1);
    noChangeIdx = find(Y_vec == 0);
    
    % Retain all minority class samples (pixels representing 'Change')
    current_changes = length(changeIdx);
    
    % Subsample the majority class ('No-Change') to prevent model bias.
    % Heuristic rule: Sample twice the number of positive pixels, with a baseline minimum of 1500.
    numToSample = max(1500, current_changes * 2);
    if length(noChangeIdx) > numToSample
        sampledNoChangeIdx = datasample(noChangeIdx, numToSample, 'Replace', false);
    else
        sampledNoChangeIdx = noChangeIdx;
    end
    
    % Append the sampled dataset to the global training arrays
    selectedIdx = [changeIdx; sampledNoChangeIdx];
    X_train = [X_train; X_vec(selectedIdx, :)];
    Y_train = [Y_train; Y_vec(selectedIdx)];
    
    fprintf('Done. (+%d samples)\n', length(selectedIdx));
end

fprintf('1/A. Training Random Forest Classifier on %d data points...\n', length(Y_train));
tic;
% Train a Random Forest classifier (TreeBagger) using 50 decision trees via bootstrap aggregating
RF_Model = TreeBagger(50, X_train, Y_train, 'OOBPrediction', 'on', 'Method', 'classification');
toc;
fprintf('Training Complete!\n\n');

%% 2. Testing Phase

testCities = {'dubai', 'lasvegas', 'milano', 'brasilia', 'valencia'};

fprintf('--- 2. TESTING PHASE ---\n');
for i = 1:length(testCities)
    cityName = testCities{i};
    fprintf('\nCity: %s\n', cityName);
    
    [featureCube, H, W] = process_city_features(cityName);
    if isempty(featureCube)
        fprintf('  Skipped (No Image Data).\n');
        continue;
    end
    
    % Locate Ground Truth labels for the validation/test city
    gt_path = fullfile('Onera Satellite Change Detection dataset - Images', ...
        'Onera Satellite Change Detection dataset - Test Labels', ...
        cityName, 'cm', 'cm.png');
    
    % Fallback mechanism: Attempt to load from the 'Train Labels' directory if 'Test Labels' are unavailable
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
    X_test = reshape(featureCube, H*W, 17);
    [predStr, ~] = predict(RF_Model, X_test);
    predMask = reshape(str2double(predStr), H, W);
    
    % Morphological Post-Processing: Mitigate salt-and-pepper noise and isolate significant changes
    predMask = medfilt2(predMask, [5 5]); % Apply median filtering for spatial smoothing
    predMask = bwareaopen(predMask, 20);  % Remove isolated micro-clusters (noise artifacts)
    
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
end
fprintf('\nFull pipeline executed successfully.\n');

%% Helper Functions

function [features, H, W] = process_city_features(cityName)
    baseDir = 'Onera Satellite Change Detection dataset - Images'; 
    bandList = {'B01.tif', 'B02.tif', 'B03.tif', 'B04.tif', 'B05.tif', ...
                'B06.tif', 'B07.tif', 'B08.tif', 'B8A.tif', 'B09.tif'};
                
    path_T1 = fullfile(baseDir, cityName, 'imgs_1_rect');
    path_T2 = fullfile(baseDir, cityName, 'imgs_2_rect');
    
    % Inline helper function: Load TIFF images and spatially align multi-resolution spectral bands
    function cube = localLoadAndResize(folder, bands)
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
                % Upsample lower resolution bands (20m/60m) to match the 10m baseline resolution using bicubic interpolation
                if size(img,1) ~= refH
                    img = imresize(img, [refH, refW], 'bicubic');
                end
                cube = cat(3, cube, img);
            end
        end
    end

    Cube_T1_Raw = localLoadAndResize(path_T1, bandList);
    Cube_T2_Raw = localLoadAndResize(path_T2, bandList);
    
    if isempty(Cube_T1_Raw) || isempty(Cube_T2_Raw)
        features = []; H = 0; W = 0;
        return;
    end
    
    % Phase 1A: Radiometric Harmonization via Histogram Matching (Mapping T2 to T1)
    % Objective: Mitigate global radiometric discrepancies caused by varying sun elevation angles and atmospheric drift.
    [H, W, B] = size(Cube_T1_Raw);
    Cube_T1_Corr = zeros(size(Cube_T1_Raw));
    Cube_T2_Corr = zeros(size(Cube_T2_Raw));
    
    for i = 1:B
        Ref_Band = mat2gray(Cube_T1_Raw(:,:,i));
        Cube_T1_Corr(:,:,i) = Ref_Band;
        Cube_T2_Corr(:,:,i) = imhistmatch(Cube_T2_Raw(:,:,i), Ref_Band);
    end
    
    % Phase 1B: Spatial Co-registration using Principal Component Analysis (PCA)
    X1 = reshape(Cube_T1_Corr, H*W, B);
    X2 = reshape(Cube_T2_Corr, H*W, B);
    [~, score1] = pca(X1, 'NumComponents', 1);
    [~, score2] = pca(X2, 'NumComponents', 1);
    PC1_T1 = reshape(score1, H, W);
    PC1_T2 = reshape(score2, H, W);
    
    % Spatially align the Principal Component maps using a rigid translation transformation
    [optimizer, metric] = imregconfig('monomodal');
    tform = imregtform(PC1_T2, PC1_T1, 'translation', optimizer, metric);
    R = imref2d(size(PC1_T1));
    Cube_T2_Reg = imwarp(Cube_T2_Corr, tform, 'OutputView', R);
    
    % Phase 2A: Absolute Dimensionality Reduction (PCA) Difference Map
    diffCube = abs(Cube_T1_Corr - Cube_T2_Reg);
    X_Diff = reshape(diffCube, H*W, B);
    [~, scoreDiff] = pca(X_Diff, 'NumComponents', 1);
    pcaMapRaw = abs(reshape(scoreDiff(:,1), H, W)); 
    
    % Clamp the upper 1st percentile to remove outliers, then apply Min-Max normalization
    p99_pca = prctile(pcaMapRaw(:), 99);
    pcaMap = min(pcaMapRaw, p99_pca);
    pcaMap = (pcaMap - min(pcaMap(:))) / (max(pcaMap(:)) - min(pcaMap(:)));
    
    % Phase 2B: Spectral Angle Mapper (SAM)
    dotProd = sum(Cube_T1_Corr .* Cube_T2_Reg, 3);
    norm1 = sqrt(sum(Cube_T1_Corr.^2, 3));
    norm2 = sqrt(sum(Cube_T2_Reg.^2, 3));
    samMapRaw = acos(dotProd ./ (norm1 .* norm2 + eps));
    samMapRaw(isnan(samMapRaw)) = 0; 
    samMapRaw = real(samMapRaw);
    
    % Clamp the upper 1st percentile to remove outliers, then apply Min-Max normalization
    p99_sam = prctile(samMapRaw(:), 99);
    samMap = min(samMapRaw, p99_sam);
    samMap = (samMap - min(samMap(:))) / (max(samMap(:)) - min(samMap(:)));
    
    % Phase 2C: Structural Fusion Map (Combining PCA-based Magnitude and SAM-based Angular differences)
    weightPCA = 0.5;
    fusionMap = (weightPCA * pcaMap) + ((1-weightPCA) * samMap);
    fusionMap = medfilt2(fusionMap, [3 3]);
    fusionMap = (fusionMap - min(fusionMap(:))) / (max(fusionMap(:)) - min(fusionMap(:)));
    
    % Phase 3: Compute the 10-dimensional signed spectral difference cube
    diffCubeSigned = Cube_T2_Reg - Cube_T1_Corr;
    
    % --- ROBUST SPATIAL & SPECTRAL FEATURES ---
    
    % Feature 14: Normalized Difference Vegetation Index (NDVI) Shift
    % Highlights shifts in vegetation health, identifying deforestation or agricultural harvesting
    ndvi_T1 = (Cube_T1_Corr(:,:,8) - Cube_T1_Corr(:,:,4)) ./ (Cube_T1_Corr(:,:,8) + Cube_T1_Corr(:,:,4) + eps);
    ndvi_T2 = (Cube_T2_Reg(:,:,8) - Cube_T2_Reg(:,:,4)) ./ (Cube_T2_Reg(:,:,8) + Cube_T2_Reg(:,:,4) + eps);
    ndviDiff = abs(ndvi_T1 - ndvi_T2);
    
    % Feature 15: Normalized Difference Water Index (NDWI) Shift
    % Tracks alterations in water bodies and isolates dark architectural shadows
    ndwi_T1 = (Cube_T1_Corr(:,:,3) - Cube_T1_Corr(:,:,8)) ./ (Cube_T1_Corr(:,:,3) + Cube_T1_Corr(:,:,8) + eps);
    ndwi_T2 = (Cube_T2_Reg(:,:,3) - Cube_T2_Reg(:,:,8)) ./ (Cube_T2_Reg(:,:,3) + Cube_T2_Reg(:,:,8) + eps);
    ndwiDiff = abs(ndwi_T1 - ndwi_T2);
    
    % Feature 16 & 17: Contextual Spatial Texture (Standard Deviation Filter)
    % Quantifies local structural variance, helping the model distinguish between complex urban grids and uniform rural fields
    texture_T1 = stdfilt(Cube_T1_Corr(:,:,8), ones(5,5)); 
    texture_Diff = stdfilt(fusionMap, ones(5,5)); 
    
    % Aggregate all components into the final 17-Dimensional Feature Vector
    features = zeros(H, W, 17);
    features(:,:,1) = pcaMap;
    features(:,:,2) = samMap;
    features(:,:,3) = fusionMap;
    features(:,:,4:13) = diffCubeSigned; % Spectral Bands 1 to 10
    features(:,:,14) = ndviDiff;
    features(:,:,15) = ndwiDiff;
    features(:,:,16) = texture_T1;
    features(:,:,17) = texture_Diff;
end
