clc;clear;close all;

addpath ~/Documents/lightspeed/
cd ~/Documents/crowdexp/CrowdGPLVM

% load necessary dependicies
GPNeilPackage(1);

datasets = {'auto-mpg.data.mat', ...
    'communities.mat', ...
    'Concrete_Data.mat', ...
    'winequality-white.mat'};

%% open parallel setting
if matlabpool('size') > 0
    matlabpool close force local
end

matlabpool;

for j=3:3
    dataname=datasets{j};
    datapath=sprintf('./result-semisyn/datasets/%s',dataname);
    [X,Z]=loadExpData(datapath,500);
    % test variable
    testname='ProbRegularObserver';
    testpool=0:0.1:1;
%     testname='LatentDimension';
%     testpool=unique(floor(linspace(2,size(X,2),10)));
%     testpool(end)=[];
    
    for k=1:length(testpool)
        %     load('./new-exp-results/synsinc100.mat');
        %plotAnyObvrs(outfile);
        %%
        expsetup.StartTime=datestr(now);
        expsetup.LatentDimension=2;
        expsetup.NumberOfObservers=20;
        expsetup.ProbRegularObserver=0.5;
        expsetup.ProbHoldOut=0.5;
        expsetup.NumberOfRepeats=10;
        expsetup.NumberOfIterations=2000;
        expsetup.ProbMissing=0.4;
        expsetup.DataSetName=dataname;
        expsetup.ObserverNoiseLevel=0.2;
        expsetup.Data.X=X;
        expsetup.Data.Z=Z;
        expsetup.Approximation=false;
        
        % build test string
        tstr=sprintf('expsetup.%s=%d',testname,testpool(k));
        eval(tstr);
        %results=paperexp2(X,Z,maxObs,testObs,maxReps,propTrain,probRObs);
        expsetup=paperexp8(expsetup);
        fname=sprintf('./new-exp-results/%s-result-%s-%s.mat',testname,dataname,datestr(now));
        save(fname, 'expsetup');
        
    end
end

matlabpool close
GPNeilPackage(0);
