%% Train a CrowdGPLVM model
%
% Usage:
%   model = trainCrowdGPLVM(X,Y)
%   model = trainCrowdGPLVM(X,Y,koption)
%
% Input:
%   X           a N x L instances matrix, where each row is an instance
%               with L-dimensional feature.
%   Y           a N x D x M response tensor, where N is the number of
%               instance, D is the dimension of response. M is the number
%               of sources.
%   [koption]   a stucture must contains two structures UserKernel and
%               InstanceKernel, which control the kernel of GP used in two
%               layes, respectively. See help koptDefault for details.
% Output:
%   model       trained model.
%
% Example:
%
% See also PREDICTCROWDGPLVM KOPTDEFAULT
%
% Author:
% Han Xiao, Technical University of Munich
% xiaoh@in.tum.de
function koption = trainCrowdGPLVM(X,Y,koption)
if ~exist('lightspeed')
	error('Require Tom Minka''s LightSpeed toolbox!');
end

if nargin==2
	disp('kernel options are not specified, use default option.')
	koption = koptDefault(Y); %struct('InstanceKernel',struct('ph',1),'UserKernel',struct('ph',1),'Regularization', struct('ph',1),'TrainMethod',1);
end


[Z0, theta0, kappa0, lb, ub, koption] = iniCrowdGPLVM(X, Y, koption);
[N,D,M]=size(Y);
auxdata={N M D X Y sqdist(X',X') koption};


switch koption.TrainMethod
	case 'MAP'
		x0=[Z0(:);theta0(:);kappa0(:)]; % maximum the posterior
		funObj = @(x) wrapper1(x,auxdata);
	case 'MLE'
		theta0(:,1,:)=koption.UserConfidence.*rand(M,1)*1e-2;
		x0=[theta0(:);kappa0(:)]; % maximum likelihood
		funObj = @(x) wrapper2(x,auxdata);
% 		for m=1:M
% 			mal_expert_idxs=combnk(1:M,m);
% 			for p=1:size(mal_expert_idxs,1)
% 				theta0(:,1,:)=ones(1,M);
% 				theta0(:,1,mal_expert_idxs(p,:))=-ones(1, length(mal_expert_idxs(p,:)));
% 				x0=[theta0(:);kappa0(:)]; % maximum likelihood
% 				fval0 = funObj(x0);
% 				if fval0<f_old
% 					f_old=fval0;
% 					best_x0=x0;
% 				end
% 			end
% 		end
% 		x0=best_x0;
	case 'MKL'
		X_sqrt=sqdist(X',X');
		X_delta=(X_sqrt==0);
		X_linear=X*X';
		auxdata={N M D X Y X_delta X_linear X_sqrt koption};
		x0=[Z0(:);theta0(:);kappa0(:)]; % maximum the posterior
		funObj = @(x) wrapper3(x,auxdata);
	otherwise
		error('please choose from : ''[MAP]'',''MLE'',''MKL''.')
end

if ~all(x0<=ub & x0>=lb)
	error('bad initialization, out of boundary!');
end

fval0 = funObj(x0);
fprintf('total # variables: %d\n', length(x0));
fprintf('initial value of objective function: %.3f\n', fval0);


if exist('lbfgsb') %&& 0
	printBlock('lbfgsb started');
	% setup options for LBFGS
	opts    = struct( 'x0', x0 );
	opts.printEvery     = 100;
	opts.m  = 20;
	opts.maxIts = 1e4;
	opts.maxTotalIts =1e6;
	opts.pgtol      = 1e-10;
	opts.factr      = 1e3;
	%     f = @(x) objCrowdGPLVM(x, auxdata);
	%     g = @(x) gradCrowdGPLVM(x, auxdata);
	[x,fval] = lbfgsb( funObj, lb, ub, opts );
elseif exist('minConf_TMP')% && 0
	printBlock('minConf started');
	[x,fval] = minConf_TMP(funObj,x0,lb,ub,[]);
else
	printBlock('slow warning');
	warning('Can''t find any faster L-BFGS code. Matlab ''fmincon'' will be used! Press any key to continue...');pause;
	options = optimset('GradObj','on','Display','iter','FunValCheck','on','DerivativeCheck','on','Diagnostics','on'); % indicate gradient is provided
	%options = optimset('GradObj','on','Display','iter'); % indicate gradient is provided
	[x,fval] = fmincon(funObj,x0,[],[],[],[],lb,ub,[],options);
end

switch koption.TrainMethod
	case {'MAP','MKL'}
		[Z, theta, kappa]=reshapePars(x, N, M, D, koption.UserKernel.NumPars, koption.InstanceKernel.NumPars);
	case 'MLE'
		[theta, kappa]=reshapeParsLn(x, M, D, koption.InstanceKernel.NumPars);
		Z=calcZ(theta, kappa, auxdata, koption.InstanceKernel);
end

if isfield(koption.UserKernel,'ARD') && koption.UserKernel.ARD
    if koption.UserKernel.LogTheta
        f_theta =@(x) log(1+exp(x));
    else
        f_theta =@(x) x.^2;
    end
    global_coef=repmat(f_theta(theta(:,koption.UserKernel.ARDIndex,:)),D,size(X,2));
    koption.UserKernel.ARDWeights=global_coef.*(f_theta(theta(:,(koption.UserKernel.ARDIndex+1):end,:)));
end


koption.UserKernel.Hyperparameters=theta;
koption.InstanceKernel.Hyperparameters=kappa;
koption.X=X;
koption.Y=Y;
koption.Z=Z;
koption.ObjectiveFunVal=fval;

fprintf('finish! objective function converged at %.3f\n\n\n', fval);
end

function [f,g]=wrapper1(x,auxdata)
f = objCrowdGPLVM(x, auxdata);
g = gradCrowdGPLVM(x, auxdata);
end

function [f,g]=wrapper2(x,auxdata)
f = objLinearGP(x, auxdata);
g = gradLinearGP(x, auxdata);
end

function [f,g]=wrapper3(x,auxdata)
f = objMKLGPLVM(x, auxdata);
g = gradMKLGPLVM(x, auxdata);
end

function [z,var_z]=calcZ(theta, kappa, auxdata, koption)
[N M D X Y Xdist] = deal(auxdata{1:6});

W=theta(:,1,:);
mu=theta(:,2,:);
sigma=theta(:,3,:);

z= zeros(N,D);
var_z=zeros(N,D);

for d=1:D
	K=calcKernel(Xdist,kappa(d,:),X,[],koption);
	L = jitChol(K)';
	invK=L'\(L\eye(N));
	A=sum( ( (1+W(d,:))./sigma(d,:) ).^2)*eye(N)+invK;
	tmpvar=0;
	for m=1:M
		tmpvar=tmpvar+(1+W(d,m))/(sigma(d,m)^2)*Y(:,d,m)+sum(mu(d,m)*invK,2);
	end
	z(:,d)=A\tmpvar;
	var_z(:,d)=diag(inv(A));
end
end
