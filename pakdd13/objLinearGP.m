function f = objLinearGP(x, auxdata)
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
% options.InstanceKernel.LogTheta=0;
% options.UserKernel.ReturnKernel=0;
% options.UserKernel.AddTerms='111';
% options.Regularization.UserCov=eye(M);
% options.Regularization.DimCov=eye(D);
% options.Regularization.Mode=1;
% options.Regularization.Lambda=rand(1,M);

% %
[N M D X Y Xdist options] = deal(auxdata{:});
%[theta, kappa]=reshapePars(x, N, M, D, options.UserKernel.NumPars, options.InstanceKernel.NumPars);
[theta, kappa]=reshapeParsLn(x, M, D, options.InstanceKernel.NumPars);
% init
f=0;

W=reshape(theta(:,1,:),[D,M]);
mu=reshape(theta(:,2,:),[D,M]);
sigma=reshape(theta(:,3,:),[D,M]);
T=options.Regularization.DimCov;
S=options.Regularization.UserCov;
lambda=options.Regularization.Lambda;
eta=options.eta;



for d=1:D
	K=calcKernel(Xdist,kappa(d,:),X,[],options.InstanceKernel);
	for m=1:M
		% theta(d,1,m) is w
		% theta(d,2,m) is noise
		w_dm=W(d,m);
		C = w_dm^2*K+sigma(d,m)^2*eye(N);
		v = Y(:,d,m)-w_dm*mu(d,m);
		L = jitChol(C)';
		f = f + 2*sum(log(diag(L))) + v'*(L'\(L\v));
		if isinf(f) || isnan(f)
			error('something wrong!')
		end
	end
end

% compute the penalty of parameters
switch options.Regularization.Mode
	case 'L2'
		% L2 regularization
		
		penalty=trace(((1-eta)*(W-1)'*(W-1)+eta*(W+1)'*(W+1)+sigma'*sigma+mu'*mu)*S)+trace(kappa*kappa'*T);
	case 'L1'
		% L1 regularization
		penalty=sum((1-eta)*abs(W-1)+eta*abs(W+1)+abs(sigma)+abs(mu),1)*lambda'+sum(abs(kappa));
		% penalty=2*sqrt(2/options.Regularization.Lambda)*sum(sum(abs(W)+abs(sigma)+abs(mu)));
	case 'Quad'
		p=penaltyFunc(W,eta);
		penalty=p+trace((sigma'*sigma+mu'*mu)*S)+trace(kappa*kappa'*T);
	case 'Ad'
		% times -2 to meet f
		penalty=-.5*log(eta*mvnpdf(W(1,:),ones(1,M),S)+(1-eta)*mvnpdf(W(1,:),-ones(1,M),S));
	otherwise
		penalty=0;
end

f=f+penalty;

% we want to maximize f, but l-bfgs minimize f, thus
% f=-f;
%checked!
end