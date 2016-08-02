clc;clear;close all;
addpath('~/Documents/lightspeed/');

M=2; % nb. of experts
N=100; % nb. of instances
D=1; % dim. of response (currently we only have 1 dim response, i.e. time interval)
L=100; % dim. of instance (i.e. nb. of features of an instance)
X=rand(N,L); % instance matrix N x L
Y=rand(N,D,M); % response matrix from all experts  N x D x M

% set kernel options
koption.InstanceKernel.AddTerms='111';
koption.InstanceKernel.ReturnKernel=0;
koption.InstanceKernel.RandInit=0;
koption.InstanceKernel.LogTheta=0;
koption.UserKernel.AddTerms='111';
koption.UserKernel.ReturnKernel=0;
koption.UserKernel.RandInit=0;
koption.UserKernel.LogTheta=0;
%% test old lbfgs
addpath('~/Documents/lbfgsb-for-matlab/');
printBlock('lbfgs1');
[Z, theta, kappa]=CrowdGPLVM(N,M,D,X,Y,koption);
pause;
%% test new lbfgs
rmpath('~/Documents/lbfgsb-for-matlab/');
addpath('~/Documents/lbfgs/');
printBlock('lbfgs2');
% ini
[Z0, theta0, kappa0, lb, ub, koption] = iniCrowdGPLVM(N, M, D, [0,1], koption);
auxdata={N M D X Y sqdist(X',X') koption}; 
x0=[Z0(:);theta0(:);kappa0(:)];
opts    = struct( 'x0', x0 );
opts.printEvery     = 100;
opts.m  = 20;
opts.maxIts = 1e4;
opts.maxTotalIts =1e6;
opts.pgtol      = 1e-10;
opts.factr      = 1e3;
f = @(x) objCrowdGPLVM(x, auxdata);
g = @(x) gradCrowdGPLVM(x, auxdata);
[x,f,info] = lbfgsb( {f,g} , lb, ub, opts );
pause;
%% test matlab internal
printBlock('fmincon')
x0=[Z0(:);theta0(:);kappa0(:)];
options = optimset('GradObj','on','Display','iter','FunValCheck','on','DerivativeCheck','on','Diagnostics','on'); % indicate gradient is provided 
%options = optimset('GradObj','on','Display','iter'); % indicate gradient is provided 
x = fmincon(@(x) wrapperCrowdGPLVM(x,auxdata),x0,[],[],[],[],lb,ub,[],options);
[Z, theta, kappa]=reshapePars(x, N, M, D, koption.UserKernel.NumPars, koption.InstanceKernel.NumPars);

