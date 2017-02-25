%%%
% Created on Thu Feb 23 00:52:27 2017
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

% If we want to use ml-latest-small dataset
% ratingsFile = 'C:\Users\Admin\Google Drive\Future\UCLA\Winter 2017\EE-219\Project_3\ml-latest-small\ratings.csv';
% moviesFile = 'C:\Users\Admin\Google Drive\Future\UCLA\Winter 2017\EE-219\Project_3\ml-latest-small\movies.csv';
% Ratings = csvread(ratingsFile, 1, 0);
% fid = fopen(moviesFile, 'rt');
% dt = textscan(fgetl(fid), '%s');
% Movies = textscan(fid, '%d %q %s', 'Delimiter', ',');    % use %q to handle commas in csv files
% out = fclose(fid);
% uniqueUsers = unique(Ratings(:,1));
% R = zeros(length(uniqueUsers), length(Movies{2}));

for i= 1:length(uniqueUsers)
    ind1 = inputData(:,1) == i;
    subArr = inputData(ind1,2:3);
    for j= 1:length(subArr);
        movieId = subArr(j,1);
        rating = subArr(j,2);
        R(i,movieId) = rating;
    end
end
R(R==0) = NaN;
% Apply the weighted nmf. The wnmfrule function automatically converts all 
% NaN's to zero and computes the weight matrix keeping all non-NaN's as 1 
% and NaN's to zero
k = {10, 50, 100};
option = struct('iter',100);
for i=1:length(k)
   [A,Y,numIter,tElapsed,finalResidual]=wnmfrule(R,k{i},option);
   fprintf('\nk: %d, Residual : %d\n', k{i}, finalResidual);
end