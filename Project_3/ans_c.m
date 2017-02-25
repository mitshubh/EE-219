%%%
% Created on Thu Feb 24 12:04:24 2017
%
% @author: Shubham
%%%

%clean workspace
clear;
clc;
pause(1)

%load dataset
movie100K = 'C:\Users\Admin\Google Drive\Future\UCLA\Winter 2017\EE-219\Project_3\ml-100k\u.data';
inputData = dlmread(movie100K);
uniqueUsers = unique(inputData(:,1));
uniqueMovies = unique(inputData(:,2));
R = zeros(length(uniqueUsers), length(uniqueMovies));

k = {10, 50, 100};
option = struct('iter',100, 'dis', false);

% 10 fold Cross Validation
Indices = crossvalind('Kfold', length(inputData), 10);

upper_thresh=4;
lower_thresh=3;
threshold=0:0.1:5;
avgPrecision = zeros(1,length(threshold));
avgRecall = zeros(1,length(threshold));
for th = 1:length(threshold)
    precision=0;
    recall=0;
    for i=1:10
        likedMovies_pred=0;
        likedMovies_actual=0;
        trainData = NaN(size(R));
        testIndex = Indices==i;
        trainIndex = ~testIndex;
        testInput = inputData(testIndex,1:3);
        trainInput = inputData(trainIndex,1:3);  % Convert this to proper format
        for j= 1:size(trainInput,1)
            trainData(trainInput(j,1), trainInput(j,2)) = trainInput(j,3);
        end
        [A,Y]=wnmfrule(trainData,k{3},option);
        R_pred = A*Y;
        % Precision
        for j= 1:size(testInput,1)
            if R_pred(testInput(j,1), testInput(j,2))>=threshold(th) % System predicts that the user likes the movie
               likedMovies_pred = likedMovies_pred + 1;
               if testInput(j,3)>=4 % threshold(th) % How many such movies were actually liked ?
                    likedMovies_actual = likedMovies_actual + 1;
               end
            end
        end
        precision = precision + likedMovies_actual/likedMovies_pred;
        % Recall
        likedMovies_actual=0;
        likedMovies_pred=0;
        for j= 1:size(testInput,1)
            if testInput(j,3)>=4 %threshold(th)    
               likedMovies_actual = likedMovies_actual + 1;
               % How many such movies were predicted as liked ?
               if R_pred(testInput(j,1), testInput(j,2))>=threshold(th)
                    likedMovies_pred = likedMovies_pred + 1;
               end
            end
        end
        recall = recall + likedMovies_pred/likedMovies_actual; 
        fprintf('Threshold: %d\ti: %d\tPrecision: %d\tRecall: %d\n',threshold(th), i, precision, recall);
    end
    avgPrecision(th) = precision/10;
    avgRecall(th) = recall/10;
end
plot(avgRecall, avgPrecision, 'Marker', '.')
title('Precision v/s Recall for k=100')
xlabel('Recall')
ylabel('Precision')