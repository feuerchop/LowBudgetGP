function expsetup = paperexp8(expsetup)

% just for test
% orgX=rand(10,10);
% orgZ=rand(10,1);
% maxObs=10;
% testObs=1;
% maxReps=10;
% propTrain=0.5;
% probRObs=0.9;


P=expsetup.NumberOfRepeats;
M=expsetup.NumberOfObservers;
iters=expsetup.NumberOfIterations;
q=expsetup.LatentDimension;
prob_miss=expsetup.ProbMissing;
prob_regob=zeros(1,M);
prob_regob(1:floor(M*expsetup.ProbRegularObserver))=1;
level_noise=expsetup.ObserverNoiseLevel;
minZ=min(expsetup.Data.Z);
maxZ=max(expsetup.Data.Z);
useApprox=expsetup.Approximation;



%%
lvmob_result=cell(1,P);
avg_result=cell(1,P);
gt_result=cell(1,P);
gplvm_result=cell(1,P);
lvmob2_result=cell(1,P);
lvmob3_result=cell(1,P);

for p=1:P
    cvp = cvpartition(size(expsetup.Data.X,1), 'Holdout', expsetup.ProbHoldOut);
    % prepare training and testing data
    X{p}=expsetup.Data.X(cvp.training,:);
    Z{p}=expsetup.Data.Z(cvp.training);
    Xt{p}=expsetup.Data.X(cvp.test,:);
    Zt{p}=expsetup.Data.Z(cvp.test,:);
    expsetup.Data.CVIndex{p}.Train=cvp.training;
    expsetup.Data.CVIndex{p}.Test=cvp.test;
end


parfor p=1:P
    % prepare simulated observer data
    nseeds=2;
    Y=zeros(size(X{p},1),M);
    Yt=zeros(size(Xt{p},1),M);
    s_g=level_noise*abs(maxZ-minZ)*rand(1,M);
    for m=1:M
        [f,b] = genNLfunc2([minZ,maxZ],nseeds,prob_regob(m));
        Y(:,m)=randnorm(1,f(Z{p}),s_g(m));
        Yt(:,m)=f(Zt{p});
    end
    
    % create missing value
    Y(rand(size(Y))<prob_miss)=nan;
    
    % test lvmob
    model=lvmobCreate(X{p},Y,q,[],[],[],iters,useApprox);
    lvmob_result{p}=testModel(model,Xt{p},Zt{p},Yt,M);
    
    % test lvmob2
    model=lvmob2Create(X{p},Y,q,[],[],[],iters,useApprox);
    lvmob2_result{p}=testModel(model,Xt{p},Zt{p},Yt,M);
    
    % test avg model
    model=gpavgCreate(X{p},Y,[],[],iters,useApprox);
    avg_result{p}=testModel(model,Xt{p},Zt{p},Yt,M);
    
    % test gplvm model
    model=mgpCreate(X{p},Y,[],[],iters,useApprox);
    gplvm_result{p}=testModel(model,Xt{p},Zt{p},Yt,M);
    
    % test lvmob3
    model=lvmob3Create(X{p},Y,[],[],iters,useApprox);
    lvmob3_result{p}=testModel(model,Xt{p},Zt{p},Yt,M);
    
    % test ground truth
    model=gpGtCreate(X{p},Y,Z{p},[],[],iters,useApprox);
    gt_result{p}=testModel(model,Xt{p},Zt{p},Yt,M);
    
end
%%
expsetup.LVMOB=lvmob_result;
expsetup.LVMOB2=lvmob2_result;
expsetup.LVMOB3=lvmob3_result;
expsetup.AVG=avg_result;
expsetup.LVM=gplvm_result;
expsetup.GT=gt_result;

expsetup.FinishTime=datestr(now);
end

function result=testModel(model,Xt,Zt,Yt,M)
[pred,~]=lvmobPredict(model,Xt,0);

result.gt.PCC=corr(pred,Zt);
result.gt.SCC=corr(pred,Zt,'type','Spearman');
if result.gt.PCC>0
    result.gt.NMAE=calcNMAE(pred,Zt);
else
    result.gt.NMAE=calcNMAE(-pred,Zt);
end

result.obs.PCC=0;
result.obs.SCC=0;
result.obs.NMAE=0;

for m=1:M
    [pred,~]=lvmobPredict(model,Xt,m);
    result.obs.PCC=result.obs.PCC+corr(pred,Yt(:,m));
    result.obs.SCC=result.obs.SCC+corr(pred,Yt(:,m),'type','Spearman');
    result.obs.NMAE=result.obs.NMAE+calcNMAE(pred,Yt(:,m));
end
result.obs.PCC=result.obs.PCC/M;
result.obs.SCC=result.obs.SCC/M;
result.obs.NMAE=result.obs.NMAE/M;
end