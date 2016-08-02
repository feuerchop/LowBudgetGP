clc;clear;close all;

addpath('stickman-exp/MOCAP0p136/');
addpath('stickman-exp/NDLUTIL0p161/');

skel=acclaimReadSkel('stickman-exp/09.asf');
[channels,skel]=acclaimLoadChannels('stickman-exp/09_01.amc',skel);

selExp=[5,10,22,29];
N=size(channels,1);
D=3;
M=length(selExp);
M=min(M, length(skel.tree));

X=[1:N]';
Y=zeros(N,D,M);


for j=1:N
    allexp=skel2xyz(skel,channels(j,:));
    Y(j,:,:)=allexp(selExp,:)';
end

kopt=koptDefault(Y);
kopt.UserKernel.AddTerms='111';
kopt.InstanceKernel.AddTerms='011';
model1 = trainCrowdGPLVM(X,Y,kopt);