function dx = gradLinearGP(x, auxdata)
% --just for testing--
% clc;clear;close all;
% M=5;
% N=4;
% D=6;
% L=10;
% theta=rand(D,3,M);
% kappa=rand(D,5);
% Z=rand(N,D);
% X=rand(N,L);
% Xdist=sqdist(X',X');
% Y=rand(N,D,M);
% options.InstanceKernel.ReturnKernel=1;
% options.InstanceKernel.AddTerms='111';
% options.InstanceKernel.LogTheta=0;
% options.UserKernel.ReturnKernel=0;
% options.UserKernel.AddTerms='111';
% options.Regularization.UserCov=eye(M);
% options.Regularization.DimCov=eye(D);
% options.Regularization.Mode=1;
% options.Regularization.Lambda=rand(1,M);


[N M D X Y Xdist options] = deal(auxdata{:});
%[theta, kappa]=reshapePars(x, N, M, D, options.UserKernel.NumPars, options.InstanceKernel.NumPars);
[theta, kappa]=reshapeParsLn(x, M, D, options.InstanceKernel.NumPars);

W=reshape(theta(:,1,:),[D,M]);
mu=reshape(theta(:,2,:),[D,M]);
sigma=reshape(theta(:,3,:),[D,M]);
S=options.Regularization.UserCov;
T=options.Regularization.DimCov;
lambda=options.Regularization.Lambda;
eta=options.eta;

% init
dW=zeros(size(W));
dmu=zeros(size(mu));
dsigma=zeros(size(sigma));
dtheta=zeros(size(theta));
dkappa=zeros(size(kappa));

% threshold value below which we consider an element to be zero
DELTA = 1e-8;

% --- see paper for details
for d=1:D
	%[K, dK{1}, dK{2}, dK{3}]=computeSEkernel(X,kappa(d,:));
	K=calcKernel(Xdist,kappa(d,:),X,[],options.InstanceKernel);
	for m=1:M
		w_dm=W(d,m);
		C=w_dm^2*K+sigma(d,m)^2*eye(N);
		invC=inv_posdef(C);
		tmpvar=(Y(:,d,m)-w_dm*mu(d,m))'*invC;
		dW(d,m)=-w_dm*trace(invC*K)...
			+mu(d,m)*sum(tmpvar)...
			+w_dm*tmpvar*K*tmpvar';
		dmu(d,m)=dmu(d,m)+w_dm*sum(invC*Y(:,d,m))-mu(d,m)*w_dm^2*sum(sum(invC));
		dsigma(d,m)=-2*sigma(d,m)*(trace(invC)-tmpvar*tmpvar');
		[~, dkappa_tmp]=calcKernel(Xdist,kappa(d,:),X, (Y(:,d,m)-w_dm*mu(d,m)),options.InstanceKernel,C);
		dkappa(d,:)=dkappa(d,:)+w_dm^2*dkappa_tmp;
	end
end

switch options.Regularization.Mode
	case 'L2'
		dW=2*dW-2*((1-eta)*(W-1)+eta*(W+1))*S;
		dmu=2*dmu-2*mu*S;
		dsigma=dsigma-2*sigma*S;
		dkappa=dkappa-2*T*kappa;
		%deta=trace(((W-1)'*(W-1)-(W+1)'*(W+1))*S);
	case 'L1'
		tmpvar=repmat(lambda,D,1);%2*repmat(sqrt(2./lambda),D,1);
		dW=2*dW-tmpvar.*((1-eta)*sign_tol(W-1, DELTA)+eta*sign_tol(W+1, DELTA));
		dmu=2*dmu-tmpvar.*sign_tol(mu, DELTA);
		dsigma=dsigma-tmpvar.*sign_tol(sigma, DELTA);
		dkappa=dkappa-sign_tol(kappa, DELTA);
	case 'Quad'
		[~,g]=penaltyFunc(W,eta);
		dW=2*dW-g;
		dmu=2*dmu-2*mu*S;
		dsigma=dsigma-2*sigma*S;
		dkappa=dkappa-2*T*kappa;
	case 'Ad'
		% times -2 to meet f
		p1=1/(eta*mvnpdf(W(1,:),ones(1,M),S)+(1-eta)*mvnpdf(W(1,:),-ones(1,M),S));
		p2=eta*mvnpdf(W(1,:),ones(1,M),S)*(-(W(1,:)-ones(1,M))/S)+(1-eta)*mvnpdf(W(1,:),-ones(1,M),S)*(-(W(1,:)+ones(1,M))/S);
		dW=2*p1*p2;
	otherwise
		dW=2*dW;
		dmu=2*dmu;
end
% --- end

dtheta(:,1,:)=dW;
dtheta(:,2,:)=dmu;
dtheta(:,3,:)=dsigma;


% since we are minimize f, we need to reverse the gradient
dtheta=-dtheta;
dkappa=-dkappa;

% dkappa(2)=0;

dx=[dtheta(:);dkappa(:)];
end

function z=sign_tol(x, DELTA)
z = (x > DELTA) - (x < -DELTA); % sign(x) with DELTA tolerance
end