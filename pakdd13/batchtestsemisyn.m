clc; close all; clear all;
addpath('~/Documents/lightspeed/');
addpath('~/Documents/lbfgs/');

Kfold=14;
rpath='result-semisyn/';
modelname='CrowdGPLVM';

%% open parallel setting
if matlabpool('size') > 0
    matlabpool close force local
end

matlabpool;

%% load data set
% data set is in csv, each row is an instance
% the last column of each row is the groundtruth response
dpath='dataset/';
dataname='Wine';
ds=importdata(strcat(dpath, dataname,'.csv'));
[~,L]=size(ds.data);
N=1000; % get first N instances
D=1;
M=5;
X=ds.data(1:N,1:(L-1));
Y=zeros(N,D,M);
Z=ds.data(1:N,end);

%% generate expert and their observations
s_g=2*rand(1,M);
nsteps=5;
for m=1:M
    f=getPLfunc([min(Z),max(Z)],[min(Z),max(Z)],nsteps);
    Y(:,1,m)=randnorm(1,f(Z),s_g(m));
end

% write down this data set
fname=sprintf('%s-%s.mat',dataname,datestr(now));
save(strcat(dpath, fname), 'X','Y','Z');


%% start testing for CrowdGPLVM
fname=sprintf('%s-%s-%s.mat',modelname,dataname,datestr(now));
crossvalCGPLVM( X, Y, Z, Kfold, 'mae', strcat(rpath, fname));

% TODO: start testing for SVR
% TODO: start testing for avg.
% ...
