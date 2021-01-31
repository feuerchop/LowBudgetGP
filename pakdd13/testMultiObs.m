clc;clear;close all;
% also need libsvm for SVR
addpath('~/Documents/lightspeed/');
addpath('~/Documents/lbfgs/');
N=50;
D=1;

xrange=[0,2*pi];
X=sort(rand(N,1))*(xrange(2)-xrange(1))+xrange(1);
Xt=(0:0.05:2*pi)';
Xt=setdiff(Xt,X);

% the groundtruth function (instance -> groundtruth)
% g=@(x) 10*sinc(x);
g=@(x) 10*sin(6*x).*sin(x/2);
% g=@(x) sin(x);

Z=g(X);
% s_g(4)=0;

pccCurve=[];
maneCurve=[];

% generate piecewise nonlinear function
nseeds=2;
reg_prob=1; % probabilty of a regular user;
exp_info=[];
maxObs=10;
repTest=10;

repPCC=zeros(maxObs-1,repTest);
repMANE=zeros(maxObs-1,repTest);

for m=1:maxObs
    [f,b] = genNLfunc2([min(Z),max(Z)],nseeds,reg_prob);
    exp_info=[exp_info;b];
    expert_func{m}=@(a) f(a); %w(1)*a; %abs(a);
end
%%

for M=2:maxObs
    Y=zeros(N,D,M);
    s_g=2*rand(1,M);
    
    disp(['#observers',num2str(M)]);
    
    
    for m=1:M
        Y(:,1,m)=randnorm(1,expert_func{m}(Z),s_g(m));
    end
    
    %% Linear Observer Model
    
    for p=1:repTest
        % set kernel options
        kopt=koptDefault(Y); %MUST use koptDefault to initialize the option!
        kopt.TrainMethod='MLE'; % can be 'MLE' or 'MAP'
        kopt.Regularization.Mode='Quad'; % can be 'L1' or 'L2', only for 'MLE'
        kopt.UserConfidence=exp_info(1:M);%ones(1,M); % higher means more reliable
        kopt.eta=1;
        % train model
        model1=trainCrowdGPLVM(X,Y,kopt);
        % set kernel options
        kopt=koptDefault(Y);
        kopt.TrainMethod='MAP';
        % initialize the new model with previous model
        kopt=inheritModel(model1,kopt);
        % train model
        model2=trainCrowdGPLVM(X,Y,kopt);
        % predict on Xt, 0 for prediting the groundtruth, 1...M for predicting corresponding users
        [mean_y,var_y]=predictCrowdGPLVM(Xt,model2,0);
        repPCC(M,p)=corr(mean_y,g(Xt));
        repMANE(M,p)=calcNMAE(mean_y,g(Xt));
    end
end

return

%% Nonlinear Observer Model


% draw plot
figure(1);subplot(5,1,4);
shadedErrorBar(Xt',normaltoUnit(mean_y),normaltoUnit(var_y),{'k--'});
axis tight
set(gca,'XTick',[],'YTick',[]);
box on;
model_name='NonlinearGP (LinearGP ini.)';
title(sprintf('%s: %.3f',model_name, calcNMAE(mean_y,g(Xt))));
plotExpertPrediction(g, expert_func, s_g, Xt, model2, mycolor,model_name);


%% Multiple Kernel Learning
% set kernel options
kopt=koptDefault(Y);
kopt.TrainMethod='MKL';

% must >=0, controls the number of kernels
kopt.InstanceKernel.Width=10;
kopt.UserKernel.Width=10;

% higher value indicates a more sparse solution.
kopt.InstanceKernel.RegC=10;
kopt.UserKernel.RegC=10;

% initialize the new model with previous model
kopt=inheritModel(model1,kopt);

% train model
model3=trainCrowdGPLVM(X,Y,kopt);

% predict on Xt, 0 for prediting the groundtruth, 1...M for predicting corresponding users
[mean_y,var_y]=predictCrowdGPLVM(Xt,model3,0);

% draw plot
figure(1);subplot(5,1,5);
shadedErrorBar(Xt',normaltoUnit(mean_y),normaltoUnit(var_y),{'k--'});
axis tight
set(gca,'XTick',[],'YTick',[]);
box on;
model_name='MKL-GP (LinearGP ini.)';
title(sprintf('%s: %.3f',model_name, calcNMAE(mean_y,g(Xt))));
plotExpertPrediction(g, expert_func, s_g, Xt, model3, mycolor,model_name);


%%
figure;

% draw response curve of each expert
for m=1:M
    subplot(4,M,m)
    Zt=[min(Z):0.1:max(Z)]';
    mean_y=expert_func{m}(Zt);
    var_y=repmat(s_g(m),[length(Zt),1]);
    shadedErrorBar(Zt,mean_y,var_y,{'color',mycolor(m,:)'});
    axis ([min(Zt), max(Zt), min(mean_y), max(mean_y)]);
    axis square;
    set(gca,'XTick',[],'YTick',[]);
    xlabel('groundtruth')
    ylabel(strcat(num2str(m), '-response'));
end

for m=1:M
    subplot(4,M,M+m)
    [mean_y,var_y, Zt]=predictCrowdGPLVM(Xt, model1, m);
    shadedErrorBar(Zt,mean_y,var_y,{'color',mycolor(m,:)'});
    axis ([min(Zt), max(Zt), min(mean_y), max(mean_y)]);
    axis square;
    set(gca,'XTick',[],'YTick',[]);
    xlabel('groundtruth')
    ylabel(strcat(num2str(m), '-response'));
end

for m=1:M
    subplot(4,M,2*M+m)
    [mean_y,var_y, Zt]=predictCrowdGPLVM(Xt, model2, m);
    shadedErrorBar(Zt,mean_y,var_y,{'color',mycolor(m,:)'});
    axis ([min(Zt), max(Zt), min(mean_y), max(mean_y)]);
    axis square;
    set(gca,'XTick',[],'YTick',[]);
    xlabel('groundtruth')
    ylabel(strcat(num2str(m), '-response'));
end

for m=1:M
    subplot(4,M,3*M+m)
    [mean_y,var_y, Zt]=predictCrowdGPLVM(Xt, model3, m);
    shadedErrorBar(Zt,mean_y,var_y,{'color',mycolor(m,:)'});
    axis ([min(Zt), max(Zt), min(mean_y), max(mean_y)]);
    axis square;
    set(gca,'XTick',[],'YTick',[]);
    xlabel('groundtruth')
    ylabel(strcat(num2str(m), '-response'));
end


