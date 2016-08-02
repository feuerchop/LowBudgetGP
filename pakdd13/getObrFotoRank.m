function [top, bottom] = getObrFotoRank( uid, num )
%GETOBRFOTORANK Summary of this function goes here
%   Detailed explanation goes here
load('real-results/fotorating-s201-obs10withIDs-10cv-201trainset.mat');
[~, bestNLOBId] = max(foto_results.Ratios(4, :));
bestNLOBModel = foto_results.Models{4}{bestNLOBId};
nObr = length(names);
if uid > nObr || uid <= 0
    fprintf('user id is not valid!\n');
    return;
end
if num < 0
    fprintf('number of fotos can not be negative!\n');
    return;
end
load('real-test/datasets/fotoratings-unrated.mat');
A=XRest;
[ratings, ~] = predictCrowdGPLVM(A(:,3:end), bestNLOBModel, uid);
rank = sortrows([FIDRest, ratings], -2);
top = rank(1:num, :);
bottom = rank(end-num+1:end, :);
end

