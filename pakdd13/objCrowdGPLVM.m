function f = objCrowdGPLVM(x, auxdata)
% --just for testing--
% clc;clear;close all;
% M=5;
% N=4;
% D=6;
% L=10;
% theta=rand(D,5,M);
% kappa=rand(D,5);
% Z=rand(N,D);
% X=rand(N,L);
% Xdist=sqdist(X',X');
% Y=rand(N,D,M);
% options.InstanceKernel.ReturnKernel=0;
% options.InstanceKernel.AddTerms='111';
% options.UserKernel.ReturnKernel=0;
% options.UserKernel.AddTerms='111';



[N M D X Y Xdist options] = deal(auxdata{:});
[Z, theta, kappa]=reshapePars(x, N, M, D, options.UserKernel.NumPars, options.InstanceKernel.NumPars);
% init
f=0;


for d=1:D
	KL=calcKernel(Xdist,kappa(d,:),X,[],options.InstanceKernel);
	f=f+ 2*sum(log(diag(KL))) + Z(:,d)'*(KL'\(KL\Z(:,d)));
	Zdist=sqdist(Z(:,d)',Z(:,d)');
	for m=1:M
		CL=calcKernel(Zdist,theta(d,:,m),Z(:,d),[],options.UserKernel);
		f=f+ 2*sum(log(diag(CL)))+Y(:,d,m)'*(CL'\(CL\Y(:,d,m)));
		if isinf(f) || isnan(f)
			error('something is wrong when calulating objective function! nan/inf?')
		end
	end
end

if isfield(options.UserKernel,'ARD') && options.UserKernel.ARD
    % add L1-regularization
    penalty=options.UserKernel.RegC*sum(sum(abs(theta(:,(options.UserKernel.ARDIndex):end,:))));
end

f=f+penalty;

end