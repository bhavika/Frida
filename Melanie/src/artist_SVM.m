artistDir = 'C:\Users\Melanie\Desktop\imp_dataset2\';
trainingSet = imageDatastore(artistDir,'IncludeSubfolders',true, 'LabelSource','foldernames');
countEachLabel(trainingSet)
artistTest = 'C:\Users\Melanie\Desktop\imp_testset2\';
testSet = imageDatastore(artistTest,'IncludeSubfolders',true, 'LabelSource','foldernames');
countEachLabel(testSet)
numImages = numel(trainingSet.Files);
cellSize = [4 4];
hogFeatureSize = length(hog_4x4);
trainingFeatures = zeros(numImages, hogFeatureSize, 'single');
for i = 1:numImages
    img = readimage(trainingSet,i);

    img = rgb2gray(img);
    img = imbinarize(img);
    trainingFeatures(i, :) = extractHOGFeatures(img, 'CellSize', cellSize);
end
trainingLabels = trainingSet.Labels;
classifier = fitcecoc(trainingFeatures, trainingLabels);
numImages2 = numel(testSet.Files);
testFeatures = zeros(numImages2, hogFeatureSize, 'single');
for j = 1:numImages2
    img2 = readimage(testSet,j);

    img2 = rgb2gray(img2);
    img2 = imbinarize(img2);
    testFeatures(j, :) = extractHOGFeatures(img2, 'CellSize', cellSize);
end
testLabels = testSet.Labels;
predictedLabels = predict(classifier, testFeatures);
conMat = confusionmat(testLabels, predictedLabels);
conMat
