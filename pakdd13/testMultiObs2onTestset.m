clc;clear;close all;
% addpath('~/Documents/lightspeed/');
% addpath('~/Documents/lbfgs/');
% addpath('~/libsvm-3.11/matlab/');

datasets = {'auto-mpg.data.mat', ...
            'communities.mat', ...
            'Concrete_Data.mat', ...
            'winequality-white.mat'};
        
dataname=datasets{2};
datapath=sprintf('./result-semisyn/datasets/%s',dataname);
[X,Z,Xt,Zt]=loadExpData(datapath,50,100);
%plotAnyObvrs(outfile);

maxObs=2;
maxReps=2;
probRObs=0.7;


%% open parallel setting
if matlabpool('size') > 0
    matlabpool close force local
end

matlabpool;
%%

%results=paperexp2(X,Z,maxObs,testObs,maxReps,propTrain,probRObs);
results=paperexp3(X,Z,maxObs,maxReps,probRObs);
fname=sprintf('./exp2-results/new/%s-N%d-M%d-F%d-R%.1f-%s.mat',dataname,size(X,1),maxObs,maxReps,probRObs,datestr(now));
save(fname, 'results');

%% predict on testset
% select best model for each baseline
ratios = results.GroundtruthPCC./results.GroundtruthMANE;
[~,bestSVRId] = max(ratios(1,:));
[~,bestGPRId] = max(ratios(2,:));
[~,bestSVRAVGId] = max(ratios(3,:));
[~,bestGPRAVGId] = max(ratios(4,:));
[~,bestRAYKARId] = max(ratios(5,:));
[~,bestLOBId] = max(ratios(6,:));
[~,bestNLOBId] = max(ratios(7,:));
bestSVRModel = results.Models{1}{bestSVRId};
bestGPRModel = results.Models{2}{bestGPRId};
bestSVRAVGModel = results.Models{3}{bestSVRAVGId};
bestGPRAVGModel = results.Models{4}{bestGPRAVGId};
bestRAYKARModel = results.Models{5}{bestRAYKARId};
bestLOBModel = results.Models{6}{bestLOBId};
bestNLOBModel = results.Models{7}{bestNLOBId};


