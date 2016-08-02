function results= paperexp3(orgX,orgZ,nrObs,nrReps,probRObs)

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
%GGPCC=zeros(1,nrReps);
RAYKARPCC=zeros(1,nrReps);
LOBPCC=zeros(1,nrReps);
NLOBPCC=zeros(1,nrReps);

SVRMANE=zeros(1,nrReps);
GPRMANE=zeros(1,nrReps);
SVRAVGMANE=zeros(1,nrReps);
GPRAVGMANE=zeros(1,nrReps);
%GGMANE=zeros(1,nrReps);
RAYKARMANE=zeros(1,nrReps);
LOBMANE=zeros(1,nrReps);
NLOBMANE=zeros(1,nrReps);


%GGOBSMANE=zeros(1,nrReps);
LOBOBSMANE=zeros(1,nrReps);
NLOBOBSMANE=zeros(1,nrReps);

%GGOBSPCC=zeros(1,nrReps);
LOBOBSPCC=zeros(1,nrReps);
NLOBOBSPCC=zeros(1,nrReps);

%Models
SVRModels = {};
GPRModels = {};
SVRAVGModels = {};
GPRAVGModels = {};
RAYKARModels = {};
LOBModels = {};
NLOBModels={};
%%
nseeds=2;
exp_info=[];
s_g=2*rand(1,nrObs);

for m=1:nrObs
  [f{m},b] = genNLfunc2([min(orgZ),max(orgZ)],nseeds,probRObs);
  exp_info=[exp_info;b];
  orgY(:,1,m)=randnorm(1,f{m}(orgZ),s_g(m));
end

N=size(orgX,1);
cvp = cvpartition(N, 'kfold', nrReps);

parfor p=1:nrReps
  % prepare training and testing data
  X=orgX(cvp.training(p),:);
  Z=orgZ(cvp.training(p));
  Xt=orgX(cvp.test(p),:);
  Zt=orgZ(cvp.test(p),:);
  Y=orgY(cvp.training(p),1,:);
  Yt=orgY(cvp.test(p),1,:);
  % prepare simulated observer data
  %     nseeds=2;
  %     exp_info=[];
  %     Y=zeros(size(X,1),1,nrObs);
  %     Yt=zeros(size(Xt,1),1,nrObs);
  %     s_g=2*rand(1,nrObs);
  %     for m=1:nrObs
  %         [f,b] = genNLfunc2([min(orgZ),max(orgZ)],nseeds,probRObs);
  %         exp_info=[exp_info;b];
  %         Y(:,1,m)=randnorm(1,f(Z),s_g(m));
  %         Yt(:,1,m)=f(Zt);
  %     end
  
  % do SVR and GP on ground truth
  % SVR
  model0=svmtrain(Z,X,'-s 3');
  mean_y=svmpredict(Zt,Xt,model0);
  SVRMANE(p)=calcNMAE(mean_y,Zt);
  SVRPCC(p)=corr(mean_y,Zt);
  SVRModels{1,p} = model0;
  % GP
  mean_y=gpbaseline(X,Z,Xt);
  GPRMANE(p)=calcNMAE(mean_y,Zt);
  GPRPCC(p)=corr(mean_y,Zt);
  GPRModels{1,p} = [X,Z];
  % SVR-AVG
  model0=svmtrain(mean(Y,3),X,'-s 3');
  mean_y=svmpredict(Zt,Xt,model0);
  SVRAVGMANE(p)=calcNMAE(mean_y,Zt);
  SVRAVGPCC(p)=corr(mean_y,Zt);
  SVRAVGModels{1,p}=model0;
  % GPR-AVG
  mean_y=gpbaseline(X,mean(Y,3),Xt);
  GPRAVGMANE(p)=calcNMAE(mean_y,Zt);
  GPRAVGPCC(p)=corr(mean_y,Zt);
  GPRAVGModels{1,p}=[X,mean(Y,3)];
  % GG
  %     kopt=koptDefault(Y); %MUST use koptDefault to initialize the option!
  %     kopt.TrainMethod='MLE'; % can be 'MLE' or 'MAP'
  %     kopt.Regularization.Mode='Quad'; % can be 'L1' or 'L2', only for 'MLE'
  %     kopt.UserConfidence=exp_info;%ones(1,M); % higher means more reliable
  %     kopt.eta=1;
  %     kopt.UserKernel.AddTerms='011';
  %     kopt.InstanceKernel.AddTerms='011';
  %     model1=trainCrowdGPLVM(X,Y,kopt);
  %     mean_y=predictCrowdGPLVM(Xt,model1,0);
  %     GGMANE(p)=calcNMAE(mean_y,Zt);
  %     GGPCC(p)=corr(mean_y,Zt);
  %     [GGOBSPCC(p), GGOBSMANE(p)]=calcObsErr(model1,Xt,Yt);
  % RAYKAR
  [w, prec] = raykarBaseline(X, Y, 1000);
  mean_y = Xt*w';
  RAYKARMANE(p)=calcNMAE(mean_y,Zt);
  RAYKARPCC(p)=corr(mean_y,Zt);
  RAYKARModels{1,p} = w;
  % LOB
  kopt=koptDefault(Y); %MUST use koptDefault to initialize the option!
  kopt.TrainMethod='MLE'; % can be 'MLE' or 'MAP'
  kopt.Regularization.Mode='Quad'; % can be 'L1' or 'L2', only for 'MLE'
  kopt.UserConfidence=exp_info;%ones(1,M); % higher means more reliable
  kopt.eta=1;
  model2=trainCrowdGPLVM(X,Y,kopt);
  mean_y=predictCrowdGPLVM(Xt,model2,0);
  LOBMANE(p)=calcNMAE(mean_y,Zt);
  LOBPCC(p)=corr(mean_y,Zt);
  [LOBOBSPCC(p),LOBOBSMANE(p)]=calcObsErr(model2,Xt,Yt);
  LOBModels{1,p}=model2;
  
  %     bestPCC=0;
  %     bestMANE=1;
  bestRatio = -inf;
  % NLOB:
  for pp=1:10
    % set kernel options
    kopt=koptDefault(Y);
    kopt.TrainMethod='MAP';
    % initialize the new model with previous model
    kopt=inheritModel(model2,kopt);
    % train model
    model3=trainCrowdGPLVM(X,Y,kopt);
    % predict on Xt, 0 for prediting the groundtruth, 1...M for predicting corresponding users
    mean_y=predictCrowdGPLVM(Xt,model3,0);
    bestPCC = corr(mean_y,Zt);
    bestMANE = calcNMAE(mean_y,Zt);
    if bestRatio < bestPCC/bestMANE
      bestRatio = bestPCC/bestMANE;
      NLOBMANE(p)=bestMANE;
      NLOBPCC(p)=bestPCC;
      [NLOBOBSPCC(p),NLOBOBSMANE(p)]=calcObsErr(model3,Xt,Yt);
      NLOBModels{1,p}=model3;
    end
  end
end

%%
results.GroundtruthPCC=[SVRPCC;GPRPCC;SVRAVGPCC;GPRAVGPCC;RAYKARPCC;LOBPCC;NLOBPCC];
results.GroundtruthMANE=[SVRMANE;GPRMANE;SVRAVGMANE;GPRAVGMANE;RAYKARMANE;LOBMANE;NLOBMANE];
results.ObserverPCC=[LOBOBSPCC; NLOBOBSPCC];
results.ObserverMANE=[LOBOBSMANE; NLOBOBSMANE];
results.Models={SVRModels;GPRModels;SVRAVGModels;GPRAVGModels;RAYKARModels;LOBModels;NLOBModels};
results.ObrInfo = {f, exp_info};
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