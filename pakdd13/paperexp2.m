function results= paperexp2(orgX,orgZ,maxObs,testObs,maxReps,propTrain,probRObs)

% just for test
% orgX=rand(10,10);
% orgZ=rand(10,1);
% maxObs=10;
% testObs=1;
% maxReps=10;
% propTrain=0.5;
% probRObs=0.9;

% prepare training and testing data
N=size(orgX,1);
idx=randsample(N,floor(N*propTrain));
X=orgX(idx,:);
Z=orgZ(idx);
Xt=orgX(setdiff(1:N,idx),:);
Zt=orgZ(setdiff(1:N,idx));

% prepare simulated observer data
nseeds=2;
exp_info=[];
Y=zeros(length(idx),1,maxObs);
s_g=2*rand(1,maxObs);
for m=1:maxObs
    [f,b] = genNLfunc2([min(orgZ),max(orgZ)],nseeds,probRObs);
    exp_info=[exp_info;b];
    Y(:,1,m)=randnorm(1,f(Z),s_g(m));
end

% do SVR and GP on ground truth
% SVR
model0=svmtrain(Z,X,'-s 3');
mean_y=svmpredict(Zt,Xt,model0);
results.SVR.NMAE=calcNMAE(mean_y,Zt);
results.SVR.PCC=corr(mean_y,Zt);
% GP
mean_y=gpbaseline(X,Z,Xt);
results.GPR.NMAE=calcNMAE(mean_y,Zt);
results.GPR.PCC=corr(mean_y,Zt);


% do SVR-AVG and GPR-AVG
for m=2:testObs:maxObs
    % SVR-AVG
    model0=svmtrain(mean(Y(:,:,1:m),3),X,'-s 3');
    mean_y=svmpredict(Zt,Xt,model0);
    results.SVRAVG.NMAE(m)=calcNMAE(mean_y,Zt);
    results.SVRAVG.PCC(m)=corr(mean_y,Zt);
    % GPR-AVG
    mean_y=gpbaseline(X,mean(Y(:,:,1:m),3),Xt);
    results.GPRAVG.NMAE(m)=calcNMAE(mean_y,Zt);
    results.GPRAVG.PCC(m)=corr(mean_y,Zt);
end



end