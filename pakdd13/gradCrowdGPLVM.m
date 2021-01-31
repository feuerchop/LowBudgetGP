function dx = gradCrowdGPLVM(x, auxdata)
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
dZ=zeros(size(Z));
dtheta=zeros(size(theta));
dkappa=zeros(size(kappa));


DELTA = 1e-8;

for d=1:D
	[KL, dkappa(d,:)]=calcKernel(Xdist,kappa(d,:),X, Z(:,d),options.InstanceKernel);
	dZ(:,d)=dZ(:,d)-2*(KL'\(KL\Z(:,d)));
	Zdist=sqdist(Z(:,d)',Z(:,d)');
	for m=1:M
		[~, dtheta(d,:,m), dZ_md]=calcKernel(Zdist,theta(d,:,m),Z(:,d),Y(:,d,m),options.UserKernel);
		dZ(:,d)=dZ(:,d)+dZ_md;
	end
end


if isfield(options.UserKernel,'ARD') && options.UserKernel.ARD
    % add sub-gradient for L1-regularization
    dtheta(:,(options.UserKernel.ARDIndex):end,:)=...
        dtheta(:,(options.UserKernel.ARDIndex):end,:)+...
        options.UserKernel.RegC*sign_tol(dtheta(:,(options.UserKernel.ARDIndex):end,:), DELTA);
end


% since we are minimize f, we need to reverse the gradient
dZ=-dZ;
dtheta=-dtheta;
dkappa=-dkappa;

dx=[dZ(:);dtheta(:);dkappa(:)];

end

function z=sign_tol(x, DELTA)
z = (x > DELTA) - (x < -DELTA); % sign(x) with DELTA tolerance
end