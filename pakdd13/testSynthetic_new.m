clc;clear;close all;
% also need libsvm for SVR
%addpath('~/Documents/lightspeed/');
%addpath('~/Documents/lbfgs/');
N=50;
D=1;
M=21;

xrange=[0,2*pi];
X=sort(rand(N,1))*(xrange(2)-xrange(1))+xrange(1);
Xt=(0:0.01:2*pi)';
Xt=setdiff(Xt,X);

% the groundtruth function (instance -> groundtruth)
% g=@(x) 10*sinc(x);
g=@(x) 10*sin(6*x).*sin(x/2);
% g=@(x) sin(x);

Z=g(X);
Zt = g(Xt);
Y = zeros(N,D,M);
Yt=zeros(size(Zt,1),D,M);
Y_sigmoid = zeros(size(Zt,1),D,M);
s_g=2*rand(1,M);
% s_g(4)=0;

%mycolor=distinguishable_colors(M);

% generate piecewise nonlinear function
nseeds=2;
reg_prob=0.5; % probability of a regular user;
exp_info=[];
for m=1:M
    [f,b] = genNLfunc2([min(Z),max(Z)],nseeds,reg_prob);
    exp_info=[exp_info;b];
    expert_func{m}=@(a) f(a); %w(1)*a; %abs(a);
end
['figure']
figure(1)
hold on
subplot(5,1,1);
% draw syn data
plot(Xt,normaltoUnit(g(Xt)),'k-','linewidth',2)
Z_sigmoid = sigmoid(normaltoUnit(g(Xt)));
hold all;
label_equality = []
for m=1:M
    Yt(:,1,m)=randnorm(1,expert_func{m}(Zt),s_g(m));
    Y_sigmoid(:,1,m)=sigmoid(Yt(:,1,m));
    %shadedErrorBar(x',(1+w(m))*y',sqrt(s_g(m))*ones(1,N),{'-','color',mycolor(m,:)},1);
    %plot(X,normaltoUnit(Y(:,1,m)),'.')%,'color',mycolor(m,:));
    %plot(X,sigmoid(expert_func{m}(Z)), '-')
    %normaltoUnit(Y(:,1,m))
    %sigmoid(normaltoUnit(Y(:,1,m)))
    plot(Xt,normaltoUnit(Yt(:,1,m)), '.-')
    equality = sum(Z_sigmoid==Y_sigmoid(:,1,m))/size(Z_sigmoid,1);
    label_equality = [label_equality, equality];
    pause()
    
end
hold off;
axis tight;
box on;
set(gca,'XTick',[],'YTick',[]);
title('groundtruth and observations from four experts')

% subplot(5,1,2)
% % train SVR using avg as baseline
% model0=svmtrain(mean(Y,3),X,'-s 3');
% mean_y=svmpredict(g(Xt),Xt,model0);
% %plot(X,normaltoUnit(mean(Y,3)),'m.');hold on;
% hold on;plot(Xt,normaltoUnit(mean_y),'k--','linewidth',1);hold off;
% axis tight;
% set(gca,'XTick',[],'YTick',[]);
% box on;
% model_name='SVR (trained on avg.)';
% title(sprintf('%s: %.3f',model_name, calcNMAE(mean_y,g(Xt))));

Y_labels = Y_sigmoid;
Z_labels = Z_sigmoid;

Y_labels(Y_labels>0.5) = 1;
Y_labels(Y_labels<=0.5)= 0;
Z_labels(Z_labels>0.5) = 1;
Z_labels(Z_labels<=0.5)= 0;

save('/home/bojan/research/papers/communication-efficient-learning/LowBudgetGP/data/data_in', 'Xt','Y_sigmoid','Z_sigmoid', 'Y_labels', 'Z_labels')
%%
% Set up model
% Train using the full training conditional (i.e. no approximation.)
options = fgplvmOptions('ftc');
latentDim = 1;

mY=reshape(Y,N,M);

model = fgplvmCreate(latentDim, M, mY, options);

% Add dynamics model.
options = gpOptions('ftc');
options.kern = kernCreate(model.X, {'rbf', 'white'});
options.kern.comp{1}.inverseWidth = 0.01;
% This gives signal to noise of 0.1:1e-3 or 100:1.
options.kern.comp{1}.variance = 1;
options.kern.comp{2}.variance = 1e-3^2;
model = fgplvmAddDynamics(model, 'gpTime', options);

% Optimise the model.
iters = 1000;
display = 1;

model = fgplvmOptimise(model, display, iters);



%%
% train a model, return the groundtruth a


% model1=trainCrowdGPLVM(X,Y);
% % Xt is the test input
% [mean_y,var_y]=predictCrowdGPLVM(Xt,model1,0);
% figure(1);subplot(5,1,3);
% shadedErrorBar(Xt',mean_y,var_y,{'k--'});
% axis tight
% model_name='NonlinearGP';
% title(sprintf('%s: %.3f',model_name, calcNMAE(mean_y,g(Xt))));
% plotExpertPrediction(g, expert_func, s_g, Xt, model1, mycolor,model_name);



%% Linear Observer Model
% set kernel options
kopt=koptDefault(Y); %MUST use koptDefault to initialize the option!
kopt.TrainMethod='MLE'; % can be 'MLE' or 'MAP'
kopt.Regularization.Mode='Ad'; % can be 'L1' or 'L2', only for 'MLE'
kopt.UserConfidence=exp_info;%ones(1,M); % higher means more reliable
kopt.eta=0.5;
% train model
model1=trainCrowdGPLVM(X,Y,kopt);

% predict on Xt, 0 for prediting the groundtruth, 1...M for predicting corresponding users
[mean_y,var_y]=predictCrowdGPLVM(Xt,model1,0);

% draw plot
figure(1);subplot(5,1,3);
shadedErrorBar(Xt',normaltoUnit(mean_y),normaltoUnit(var_y),{'k--'});
axis tight
set(gca,'XTick',[],'YTick',[]);
box on;
model_name='LinearGP';
title(sprintf('%s: %.3f',model_name, calcNMAE(mean_y,g(Xt))));
plotExpertPrediction(g, expert_func, s_g, Xt, model1, mycolor,model_name);


%% Nonlinear Observer Model
% set kernel options
kopt=koptDefault(Y);
kopt.TrainMethod='MAP';
kopt.UserKernel.ARD=0;
% initialize the new model with previous model
kopt=inheritModel(model1,kopt);

% train model
model2=trainCrowdGPLVM(X,Y,kopt);

% predict on Xt, 0 for prediting the groundtruth, 1...M for predicting corresponding users
[mean_y,var_y]=predictCrowdGPLVM(Xt,model2,0);

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


