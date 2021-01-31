clc;clear;close all;
addpath('~/Documents/lightspeed/');
addpath('~/Documents/lbfgs/');
addpath('~/libsvm-3.11/matlab/');

datasets = {'auto-mpg.data.mat', ...
    'communities.mat', ...
    'Concrete_Data.mat', ...
    'winequality-white.mat'};

for j=1:4
    dataname=datasets{j};
    datapath=sprintf('./result-semisyn/datasets/%s',dataname);
    [X,Z]=loadExpData(datapath,500);
    %plotAnyObvrs(outfile);
    
    maxObs=20;
    maxReps=20;
    probRObs=0.7;
    
    %% open parallel setting
%     if matlabpool('size') > 0
%         matlabpool close force local
%     end
%     
%     matlabpool;
    %%
    
    %results=paperexp2(X,Z,maxObs,testObs,maxReps,propTrain,probRObs);
    results=paperexp6(X,Z,maxObs,maxReps,probRObs);
    fname=sprintf('./exp2-results/GGonly-%s-N%d-M%d-F%d-R%.1f-%s.mat',dataname,size(X,1),maxObs,maxReps,probRObs,datestr(now));
    save(fname, 'results');
    
end