function dx = gradMKLGPLVM(x, auxdata)
% --just for testing--
% clc;clear;close all;
% M=5;
% N=4;
% D=6;
% L=10;
% Z=rand(N,D);
% X=rand(N,L);
% Xdist=sqdist(X',X');
% X_delta=(Xdist==0);
% X_linear=X*X';
% Y=rand(N,D,M);
% options.InstanceKernel.Width=2.^[-10:10];
% options.UserKernel.Width=2.^[-10:10];
% theta=rand(D,length(options.UserKernel.Width)+3,M);
% kappa=rand(D,length(options.InstanceKernel.Width)+3);
% options.InstanceKernel.LogTheta=0;
% options.UserKernel.LogTheta=0;


[N M D X Y X_delta X_linear X_sqrt options] = deal(auxdata{:});
[Z, theta, kappa]=reshapePars(x, N, M, D, options.UserKernel.NumPars, options.InstanceKernel.NumPars);
% init
dZ=zeros(size(Z));
dtheta=zeros(size(theta));
dkappa=zeros(size(kappa));

% threshold value below which we consider an element to be zero
DELTA = 1e-8;


for d=1:D
	[KL, dkappa(d,:)] = calcMKLKernel(kappa(d,:), X_delta, X_linear, X_sqrt, X, Z(:,d),options.InstanceKernel, 0);
	dZ(:,d)=dZ(:,d)-2*(KL'\(KL\Z(:,d)));
	Z_sqrt=sqdist(Z(:,d)',Z(:,d)');
	Z_delta=(Z_sqrt==0);
	Z_linear=Z(:,d)*Z(:,d)';
	for m=1:M
		[~, dtheta(d,:,m), dZ_md]=calcMKLKernel(theta(d,:,m),Z_delta,Z_linear,Z_sqrt,Z(:,d),Y(:,d,m),options.UserKernel,0);
		dZ(:,d)=dZ(:,d)+dZ_md;
	end
end

switch options.UserKernel.Regularization
	case 'L1'
		dtheta=dtheta-options.UserKernel.RegC*sign_tol(theta, DELTA);
	case 'L2'
		dtheta=dtheta-2*options.UserKernel.RegC*theta;
end

switch options.InstanceKernel.Regularization
	case 'L1'
		dkappa=dkappa-options.InstanceKernel.RegC*sign_tol(kappa, DELTA);
	case 'L2'
		dkappa=dkappa-2*options.InstanceKernel.RegC*kappa;
end

% since we are minimizing f, we need to reverse the gradient
dZ=-dZ;
dtheta=-dtheta;
dkappa=-dkappa;

dx=[dZ(:);dtheta(:);dkappa(:)];

end

function z=sign_tol(x, DELTA)
z = (x > DELTA) - (x < -DELTA); % sign(x) with DELTA tolerance
end