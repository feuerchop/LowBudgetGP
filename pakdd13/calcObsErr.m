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