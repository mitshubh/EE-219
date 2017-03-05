%%%
% Created on Thu Feb 26 09:29:22 2017
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
W = NaN(length(uniqueUsers), length(uniqueMovies));
k = {10, 50, 100};
option = struct('iter',10, 'disp', false);

% 10 fold Cross Validation
Indices = crossvalind('Kfold', length(inputData), 10);
lambda = [0.01, 0.1, 1];
threshold=0:0.1:5;
avgPrecision = zeros(length(lambda),length(threshold));
avgRecall = zeros(length(lambda),length(threshold));
upper_thresh=4;
lower_thresh=3;

for iter = 1:length(lambda)
    for th = 1:length(threshold)
        averageAbsErr = 0;
        maxErr=0;
        minErr=10000000;
        precision=0;
        recall=0;
        for i=1:1
            err=0;
            likedMovies_pred=0;
            likedMovies_actual=0;
            trainData = NaN(size(R));
            % testData = zeros(size(R));
            testIndex = Indices==i;
            trainIndex = ~testIndex;
            testInput = inputData(testIndex,1:3);
            trainInput = inputData(trainIndex,1:3);  % Convert this to proper format
            for j= 1:size(trainInput,1)
                trainData(trainInput(j,1), trainInput(j,2)) = trainInput(j,3);
                % W(trainInput(j,1), trainInput(j,2)) = trainInput(j,3);  % Using known ratings as weights
            end
    %         trainData=isnan(W);
    %         W(trainData)=0; % Unknown ratings have zero weights
    %         trainData=~trainData; % Using 1 for known ratings and 0 for unknown ratings
            % Using a customized wnmfrule for proper weights
            [A,Y]=regularized_wnmfrule(trainData,k{3},W,lambda(iter),option); % Before swapping, we don't care about W; so uncomment section from regularized_wnmfrule
            R_pred = A*Y;
            for j= 1:size(testInput,1);
                err = err + abs(R_pred(testInput(j,1), testInput(j,2)) - testInput(j,3));
            end
            err=err/size(testInput,1);
            maxErr = max(err,maxErr);
            minErr = min(err,minErr);
            averageAbsErr = averageAbsErr + err;
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
        averageAbsErr = averageAbsErr/1;
        avgPrecision(iter, th) = precision/1;
        avgRecall(iter, th) = recall/1;
    end
    fprintf('\nk: %d, Average Absolute Error : %d for lambda: %d\n', k{3}, averageAbsErr, lambda(iter));
end
fprintf('\nk: %d, Average Absolute Error : %d\n', k{3}, averageAbsErr);
plot(avgRecall(1,:), avgPrecision(1,:),avgRecall(2,:), avgPrecision(2,:), avgRecall(3,:), avgPrecision(3,:), 'Marker', '.')
legend('y = k=100 & lambda = 0.01','y = k=100 & lambda = 0.1','y = k=100 & lambda = 1')
title('Precision v/s Recall for k=100 and lambda=0.01,0.1,1')
xlabel('Recall')
ylabel('Precision')