clc;clear;close all;
addpath('~/Documents/lightspeed/');

M=3; % nb. of experts
N=10; % nb. of instances
D=1; % dim. of response (currently we only have 1 dim response, i.e. time interval)
L=4; % dim. of instance (i.e. nb. of features of an instance)
X=rand(N,L); % instance matrix N x L
Y=rand(N,D,M); % response matrix from all experts  N x D x M

kopt=koptDefault(Y);
kopt.Regularization.Mode='Ad'; % can be 'L1' or 'L2', only for 'MLE'
kopt.eta=0.5;
kopt.UserKernel.ARD=1;
kopt.UserKernel.RegC=1e4;
kopt.UserConfidence=ones(M,1);
kopt.TrainMethod='MLE'; % can be 'MLE' or 'MAP'

% train the model
model1=trainCrowdGPLVM(X,Y,kopt);


