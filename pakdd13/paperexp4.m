function results= paperexp4(orgX,orgZ,nrObs,nrReps,probRObs)

% just for test
% orgX=rand(10,10);
% orgZ=rand(10,1);
% maxObs=10;
% testObs=1;
% maxReps=10;
% propTrain=0.5;
% probRObs=0.9;
SVRPCC=zeros(1,nrReps);
GPRPCC=zeros(1,nrReps);
SVRAVGPCC=zeros(1,nrReps);
GPRAVGPCC=zeros(1,nrReps);


SVRMANE=zeros(1,nrReps);
GPRMANE=zeros(1,nrReps);
SVRAVGMANE=zeros(1,nrReps);
GPRAVGMANE=zeros(1,nrReps);

%%

N=size(orgX,1);
cvp = cvpartition(N, 'kfold', nrReps);

parfor p=1:nrReps
    % prepare training and testing data
    X=orgX(cvp.training(p),:);
    Z=orgZ(cvp.training(p));
    Xt=orgX(cvp.test(p),:);
    Zt=orgZ(cvp.test(p),:);
    
    
    % prepare simulated observer data
    nseeds=2;
    exp_info=[];
    Y=zeros(size(X,1),1,nrObs);
    Yt=zeros(size(Xt,1),1,nrObs);
    s_g=2*rand(1,nrObs);
    for m=1:nrObs
        [f,b] = genNLfunc2([min(orgZ),max(orgZ)],nseeds,probRObs);
        exp_info=[exp_info;b];
        Y(:,1,m)=randnorm(1,f(Z),s_g(m));
        Yt(:,1,m)=f(Zt);
    end
    
    % do SVR and GP on ground truth
    % SVR
    model0=svmtrain(Z,X,'-s 3');
    mean_y=svmpredict(Zt,Xt,model0);
    SVRMANE(p)=calcNMAE(mean_y,Zt);
    SVRPCC(p)=corr(mean_y,Zt);
    % GP
    mean_y=gpbaseline(X,Z,Xt);
    GPRMANE(p)=calcNMAE(mean_y,Zt);
    GPRPCC(p)=corr(mean_y,Zt);
    
    % SVR-AVG
    model0=svmtrain(mean(Y,3),X,'-s 3');
    mean_y=svmpredict(Zt,Xt,model0);
    SVRAVGMANE(p)=calcNMAE(mean_y,Zt);
    SVRAVGPCC(p)=corr(mean_y,Zt);
    % GPR-AVG
    mean_y=gpbaseline(X,mean(Y,3),Xt);
    GPRAVGMANE(p)=calcNMAE(mean_y,Zt);
    GPRAVGPCC(p)=corr(mean_y,Zt);
    
end

%%
results.GroundtruthPCC=[SVRPCC;GPRPCC;SVRAVGPCC;GPRAVGPCC];
results.GroundtruthMANE=[SVRMANE;GPRMANE;SVRAVGMANE;GPRAVGMANE];
end