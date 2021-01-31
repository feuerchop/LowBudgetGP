function f = objMKLGPLVM(x, auxdata)
% --just for testing--
% clc;clear;close all;
% M=5;
% N=4;
% D=6;
% % L=10;
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
f=0;



for d=1:D
	KL= calcMKLKernel(kappa(d,:),X_delta, X_linear, X_sqrt, X, [], options.InstanceKernel,0);
	f = f+ 2*sum(log(diag(KL))) + Z(:,d)'*(KL'\(KL\Z(:,d)));
	Z_sqrt  = sqdist(Z(:,d)',Z(:,d)');
	Z_delta =(Z_sqrt==0);
	Z_linear= Z(:,d)*Z(:,d)';
	for m = 1:M
		CL = calcMKLKernel(theta(d,:,m),Z_delta,Z_linear,Z_sqrt,Z(:,d),[],options.UserKernel,0);
		f=f+ 2*sum(log(diag(CL)))+Y(:,d,m)'*(CL'\(CL\Y(:,d,m)));
		if isinf(f) || isnan(f)
			error('something is wrong when calulating objective function! nan/inf?')
		end
		
		switch options.UserKernel.Regularization
			case 'L1'
				f=f+options.UserKernel.RegC*sum(abs(theta(d,:,m)));
			case 'L2'
				f=f+options.UserKernel.RegC*(theta(d,:,m)*theta(d,:,m)');
		end
	end
	
	switch options.InstanceKernel.Regularization
		case 'L1'
			f=f+options.InstanceKernel.RegC*sum(abs(kappa(d,:)));
		case 'L2'
			f=f+options.InstanceKernel.RegC*(kappa(d,:)*kappa(d,:)');
	end
end

end