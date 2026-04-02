% UNIFIED HSI PIPELINE
% Includes Phase 1, Phase 2, and limited Phase 3 + Machine Learning
clear; clc;

%% 1. Tanítási Fázis (Training Phase)

trainStr = fileread(fullfile('Onera Satellite Change Detection dataset - Images', 'train.txt'));
trainCities = strsplit(strtrim(trainStr), ',');

X_train = [];
Y_train = [];

fprintf('--- 1. TANÍTÁSI FÁZIS (Training) ---\n');
for i = 1:length(trainCities)
    cityName = trainCities{i};
    if isempty(cityName)
        continue;
    end
    fprintf('Feldolgozás (Train): %s... ', cityName);
    
    labelDir = fullfile('Onera Satellite Change Detection dataset - Images', ...
        'Onera Satellite Change Detection dataset - Train Labels', ...
        cityName, 'cm');
    
    if ~exist(labelDir, 'dir')
        fprintf('[Nincs címke] Kihagyva.\n');
        continue;
    end
    
    [featureCube, H, W] = process_city_features(cityName);
    if isempty(featureCube)
        fprintf('[Nincs Hagyományos bement] Kihagyva.\n');
        continue;
    end
    
    labelPath = fullfile(labelDir, 'cm.png');
    labelImg = imread(labelPath);
    labelImg = labelImg(1:H, 1:W);
    
    % Label vectorization
    Y_img = (labelImg == 255);
    Y_vec = Y_img(:);
    
    % Feature vectorization: (H*W) x 17
    X_vec = reshape(featureCube, H*W, 17);
    
    % Stratified random sampling
    changeIdx = find(Y_vec == 1);
    noChangeIdx = find(Y_vec == 0);
    
    % Take all change pixels
    current_changes = length(changeIdx);
    
    % Sample non-change pixels (e.g. 2x the amount of changes, or at least 1500)
    numToSample = max(1500, current_changes * 2);
    if length(noChangeIdx) > numToSample
        sampledNoChangeIdx = datasample(noChangeIdx, numToSample, 'Replace', false);
    else
        sampledNoChangeIdx = noChangeIdx;
    end
    
    % Append to Global dataset
    selectedIdx = [changeIdx; sampledNoChangeIdx];
    X_train = [X_train; X_vec(selectedIdx, :)];
    Y_train = [Y_train; Y_vec(selectedIdx)];
    
    fprintf('Kész. (+%d minta)\n', length(selectedIdx));
end

fprintf('1/A. Modell tanítása (Random Forest). %d adatpontból...\n', length(Y_train));
tic;
% Train a Random Forest with 50 Trees
RF_Model = TreeBagger(50, X_train, Y_train, 'OOBPrediction', 'on', 'Method', 'classification');
toc;
fprintf('Tanítás Befejezve!\n\n');

%% 2. Tesztelési Fázis (Testing Phase)

testCities = {'dubai', 'lasvegas', 'milano', 'brasilia', 'valencia'};

fprintf('--- 2. TESZTELÉSI FÁZIS (Testing) ---\n');
for i = 1:length(testCities)
    cityName = testCities{i};
    fprintf('\nVáros: %s\n', cityName);
    
    [featureCube, H, W] = process_city_features(cityName);
    if isempty(featureCube)
        fprintf('  Kihagyva (nincs képadat).\n');
        continue;
    end
    
    % Tesztcímke (vagy Train címkéből, ha nincs Teszt címkében)
    gt_path = fullfile('Onera Satellite Change Detection dataset - Images', ...
        'Onera Satellite Change Detection dataset - Test Labels', ...
        cityName, 'cm', 'cm.png');
    if ~exist(gt_path, 'file')
        gt_path = fullfile('Onera Satellite Change Detection dataset - Images', ...
            'Onera Satellite Change Detection dataset - Train Labels', ...
            cityName, 'cm', 'cm.png');
    end
    if ~exist(gt_path, 'file')
        fprintf('  Kihagyva (nincs GT címke).\n');
        continue;
    end
    
    gt = imread(gt_path);
    gt_bin = (gt(1:H, 1:W) == 255);
    
    % Predikció
    X_test = reshape(featureCube, H*W, 17);
    [predStr, ~] = predict(RF_Model, X_test);
    predMask = reshape(str2double(predStr), H, W);
    
    % Post-Processing szűrés (laza só-bors ellen)
    predMask = medfilt2(predMask, [5 5]);
    predMask = bwareaopen(predMask, 20);
    
    % Metrikák
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
fprintf('\nTeljes folyamat lefutott.\n');

%% Helper Functions

function [features, H, W] = process_city_features(cityName)
    baseDir = 'Onera Satellite Change Detection dataset - Images'; 
    bandList = {'B01.tif', 'B02.tif', 'B03.tif', 'B04.tif', 'B05.tif', ...
                'B06.tif', 'B07.tif', 'B08.tif', 'B8A.tif', 'B09.tif'};
                
    path_T1 = fullfile(baseDir, cityName, 'imgs_1_rect');
    path_T2 = fullfile(baseDir, cityName, 'imgs_2_rect');
    
    % Helper inline for loading cubes
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
    
    % --- MATHWORKS HYPERSPECTRAL IMAGING LIBRARY INTEGARCIÓ ---
    % Sentinel-2 kiválasztott sávjainak megfelelő központi hullámhosszak (nm)
    wavelengths = [443, 490, 560, 665, 705, 740, 783, 842, 865, 945];
    
    % Hypercube objektumok létrehozása a hivatalos library használatához
    hcube_T1 = hypercube(Cube_T1_Raw, wavelengths);
    hcube_T2 = hypercube(Cube_T2_Raw, wavelengths);
    
    % Ahhoz, hogy a manuális regisztrációink és pixeles számolásaink gyorsak maradjanak,
    % az adatokat kinyerjük a hitelesített objektumból a továbbiakhoz (opcionális, de biztonságos)
    Cube_T1_Raw = hcube_T1.DataCube;
    Cube_T2_Raw = hcube_T2.DataCube;
    % --------------------------------------------------------
    
    % Phase 1: Radiometric harmonizáció (T2 -> T1)
    [H, W, B] = size(Cube_T1_Raw);
    Cube_T1_Corr = zeros(size(Cube_T1_Raw));
    Cube_T2_Corr = zeros(size(Cube_T2_Raw));
    
    for i = 1:B
        Ref_Band = mat2gray(Cube_T1_Raw(:,:,i));
        Cube_T1_Corr(:,:,i) = Ref_Band;
        Cube_T2_Corr(:,:,i) = imhistmatch(Cube_T2_Raw(:,:,i), Ref_Band);
    end
    
    % Phase 1: PCA-Coregistration
    X1 = reshape(Cube_T1_Corr, H*W, B);
    X2 = reshape(Cube_T2_Corr, H*W, B);
    [~, score1] = pca(X1, 'NumComponents', 1);
    [~, score2] = pca(X2, 'NumComponents', 1);
    PC1_T1 = reshape(score1, H, W);
    PC1_T2 = reshape(score2, H, W);
    
    [optimizer, metric] = imregconfig('monomodal');
    tform = imregtform(PC1_T2, PC1_T1, 'translation', optimizer, metric);
    R = imref2d(size(PC1_T1));
    Cube_T2_Reg = imwarp(Cube_T2_Corr, tform, 'OutputView', R);
    
    % Phase 2: PCA Map (Absolute dif)
    diffCube = abs(Cube_T1_Corr - Cube_T2_Reg);
    X_Diff = reshape(diffCube, H*W, B);
    [~, scoreDiff] = pca(X_Diff, 'NumComponents', 1);
    pcaMapRaw = abs(reshape(scoreDiff(:,1), H, W)); 
    p99_pca = prctile(pcaMapRaw(:), 99);
    pcaMap = min(pcaMapRaw, p99_pca);
    pcaMap = (pcaMap - min(pcaMap(:))) / (max(pcaMap(:)) - min(pcaMap(:)));
    
    % Phase 2: SAM Map
    dotProd = sum(Cube_T1_Corr .* Cube_T2_Reg, 3);
    norm1 = sqrt(sum(Cube_T1_Corr.^2, 3));
    norm2 = sqrt(sum(Cube_T2_Reg.^2, 3));
    samMapRaw = acos(dotProd ./ (norm1 .* norm2 + eps));
    samMapRaw(isnan(samMapRaw)) = 0; 
    samMapRaw = real(samMapRaw);
    p99_sam = prctile(samMapRaw(:), 99);
    samMap = min(samMapRaw, p99_sam);
    samMap = (samMap - min(samMap(:))) / (max(samMap(:)) - min(samMap(:)));
    
    % Phase 2: Fusion Map
    weightPCA = 0.5;
    fusionMap = (weightPCA * pcaMap) + ((1-weightPCA) * samMap);
    fusionMap = medfilt2(fusionMap, [3 3]);
    fusionMap = (fusionMap - min(fusionMap(:))) / (max(fusionMap(:)) - min(fusionMap(:)));
    
    % Phase 3 rögzített eleme: Signed Difference Cube (10 dimenzió)
    diffCubeSigned = Cube_T2_Reg - Cube_T1_Corr;
    
    % --- ÚJ ROBUZTIKUS FEATURE-ÖK ---
    % 1. NDVI (Növényzet) különbség
    ndvi_T1 = (Cube_T1_Corr(:,:,8) - Cube_T1_Corr(:,:,4)) ./ (Cube_T1_Corr(:,:,8) + Cube_T1_Corr(:,:,4) + eps);
    ndvi_T2 = (Cube_T2_Reg(:,:,8) - Cube_T2_Reg(:,:,4)) ./ (Cube_T2_Reg(:,:,8) + Cube_T2_Reg(:,:,4) + eps);
    ndviDiff = abs(ndvi_T1 - ndvi_T2);
    
    % 2. NDWI (Víz és árnyék) különbség
    ndwi_T1 = (Cube_T1_Corr(:,:,3) - Cube_T1_Corr(:,:,8)) ./ (Cube_T1_Corr(:,:,3) + Cube_T1_Corr(:,:,8) + eps);
    ndwi_T2 = (Cube_T2_Reg(:,:,3) - Cube_T2_Reg(:,:,8)) ./ (Cube_T2_Reg(:,:,3) + Cube_T2_Reg(:,:,8) + eps);
    ndwiDiff = abs(ndwi_T1 - ndwi_T2);
    
    % 3. Kontextuális Textúra (Variancia)
    % Textúra a kiindulási képen (jelzi a modellnek az eleve "rücskös" városi területeket)
    texture_T1 = stdfilt(Cube_T1_Corr(:,:,8), ones(5,5)); 
    % Textúra magában a fúziós változástérképben
    texture_Diff = stdfilt(fusionMap, ones(5,5)); 
    
    % Összesítve (17 csatorna kimentése)
    features = zeros(H, W, 17);
    features(:,:,1) = pcaMap;
    features(:,:,2) = samMap;
    features(:,:,3) = fusionMap;
    features(:,:,4:13) = diffCubeSigned;
    features(:,:,14) = ndviDiff;
    features(:,:,15) = ndwiDiff;
    features(:,:,16) = texture_T1;
    features(:,:,17) = texture_Diff;
end
