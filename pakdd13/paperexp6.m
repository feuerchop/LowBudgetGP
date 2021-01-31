function results= paperexp6(orgX,orgZ,nrObs,nrReps,probRObs)

% just for test
% orgX=rand(10,10);
% orgZ=rand(10,1);
% maxObs=10;
% testObs=1;
% maxReps=10;
% propTrain=0.5;
% probRObs=0.9;

GGPCC=zeros(1,nrReps);
GGMANE=zeros(1,nrReps);
%%

N=size(orgX,1);
cvp = cvpartition(N, 'kfold', nrReps);

for p=1:nrReps
    % prepare training and testing data
    X=orgX(cvp.training(p),:);
    Z=orgZ(cvp.training(p));
    Xt=orgX(cvp.test(p),:);
    Zt=orgZ(cvp.test(p),:);
    
    
    % prepare simulated observer data
    nseeds=2;
    Y=zeros(size(X,1),nrObs);
    Yt=zeros(size(Xt,1),nrObs);
    s_g=2*rand(1,nrObs);
    for m=1:nrObs
        [f,b] = genNLfunc2([min(orgZ),max(orgZ)],nseeds,probRObs);
        Y(:,m)=randnorm(1,f(Z),s_g(m));
        Yt(:,m)=f(Zt);
    end
    
    [w,prec]=raykarBaseline(X,Y,1000);
    
    mean_y=w*Xt';
    GGPCC(p)=corr(mean_y',Zt);
    GGMANE(p)=calcNMAE(mean_y',Zt);
    
end

%%
results.GroundtruthPCC=GGPCC;
results.GroundtruthMANE=GGMANE;%[SVRMANE;GPRMANE;SVRAVGMANE;GPRAVGMANE;GGMANE;LOBMANE;NLOBMANE];
% results.ObserverPCC=GGOBSPCC;%[GGOBSPCC;LOBOBSPCC; NLOBOBSPCC];
% results.ObserverMANE=GGOBSMANE;%[GGOBSMANE;LOBOBSMANE; NLOBOBSMANE];
end

function [pcc,mane] = calcObsErr(mymodel,Xt,Yt)
M=size(Yt,3);
pcc_all=zeros(1,M);
mane_all=zeros(1,M);
for m=1:M
    pYt=predictCrowdGPLVM(Xt,mymodel,m);
    pcc_all(m)=corr(pYt,Yt(:,1,m));
    mane_all(m)=calcNMAE(pYt,Yt(:,1,m));
end
pcc=mean(pcc_all);
mane=mean(mane_all);
end