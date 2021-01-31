clc;clear;close all;

% load necessary dependicies
GPNeilPackage(1);

load real-test/datasets/opars_all.mat

Y=reshape(Y,size(Y,1),size(Y,3));

% number of latent dimension
q=10;
iters=5000;
useApprox=0;

model=lvmobCreate(XTrain,Y,q,[],[],[],iters,useApprox);

Zt=lvmobPredict(model,XTest,0);

Zt=(Zt-min(Zt))/(max(Zt)-min(Zt))*5;

Zobs=zeros(size(XTest,1),size(Y,2));
% for each observer
for m=1:size(Y,2)
    [Zobs(:,m),~]=lvmobPredict(model,XTest,m);
    Zobs(:,m)=(Zobs(:,m)-min(Zobs(:,m)))/(max(Zobs(:,m))-min(Zobs(:,m)))*5;
end

save('real-results/newmodelScore-alltest.mat','Zt','Zobs','model')


