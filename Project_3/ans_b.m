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
option = struct('iter',100);

% 10 fold Cross Validation
Indices = crossvalind('Kfold', length(inputData), 10);

for l= 3 % 1:3
    averageAbsErr = 0;
    maxErr=0;
    minErr=10000000;
    for i=1:10
        err=0;
        trainData = NaN(size(R));
        testData = NaN(size(R));
        testIndex = Indices==i;
        trainIndex = ~testIndex;
        testInput = inputData(testIndex,1:3);
        trainInput = inputData(trainIndex,1:3);  % Convert this to proper format
        for j= 1:size(trainInput,1);
            trainData(trainInput(j,1), trainInput(j,2)) = trainInput(j,3);
        end
        [A,Y,numIter,tElapsed,finalResidual]=wnmfrule(trainData,k{l},option);
        R_pred = A*Y;
        % for j= 1:size(testInput,1);
         %   testData(testInput(j,1), testInput(j,2)) = testInput(j,3);
        % end
        % Error = abs(predicted - actual)
        for j= 1:size(testInput,1);
            err = err + abs(R_pred(testInput(j,1), testInput(j,2)) - testInput(j,3));
        end
        err=err/size(testInput,1);
        maxErr = max(err,maxErr);
        minErr = min(err,minErr);
%         for j=1:size(R,1)
%             for m = 1:size(R,2)
%                 if isnan(testData(j,m))==0
%                     err = err + abs(testData(j,m) - R_pred(j,m));
%                 end
%             end
%         end
    averageAbsErr = averageAbsErr + err;
    end
    averageAbsErr = averageAbsErr/10;
end
fprintf('\nk: %d, Average Absolute Error : %d\n', k{3}, averageAbsErr);