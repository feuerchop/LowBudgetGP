%% Predict the response of test data using a trained CrowdGPLVM model
%
% Usage:
%   [mean_p,var_p,z] = predictCrowdGPLVM(Xt, model, user_id)
%
% Input:
%   Xt          a N x L matrix, where each row is a test instance.
%   model       a trained CrowdGPLVM model.
%               see 'help trainCrowdGPLVM' for details.
%   user_id     0 for groundtruth prediction.
%               1,...,M for users' response prediction.
%
% Output:
%   mean_p      if users=0 then return a N x D groudtruth matrix,
%               where D is the dimension of response variable.
%               else if users=1 then return a N x D x M response matrix.
%   var_p       same as mean_p, each element represents the corresponding
%               variance of the prediction.
%   z           don't need in general, only for plotting
%
% Example:
%
%
% See also TRAINCROWDGPLVM TESTSYNTHETIC
%
% Author:
% Han Xiao, Technical University of Munich
% xiaoh@in.tum.de

function [mean_p,var_p,Zt]=predictCrowdGPLVM(Xt, model, user_id)

X=model.X;
Y=model.Y;
Z=model.Z;

mean_p=zeros(size(Xt,1), size(Z,2));
var_p=zeros(size(Xt,1), size(Z,2));

for d=1:size(Z,2)
	% predict groundtruth
	koption = model.InstanceKernel;
	hypar = koption.Hyperparameters(d,:);
	
	[mean_p(:,d), var_p(:,d)]= predictSub(X, Z(:,d), Xt, hypar, koption, model.TrainMethod);
	
	if user_id>0
		if user_id>size(Y,3)
			error('user_id is out of range!')
		end
		
		koption = model.UserKernel;
		hypar = koption.Hyperparameters(d,:,user_id);
		Zt=mean_p(:,d);
		
		if nargout==3
			Zt=linspace(min(mean_p(:,d)),max(mean_p(:,d)))';
			mean_p=zeros(size(Zt,1), size(Z,2));
			var_p=zeros(size(Zt,1), size(Z,2));
		end
		
		% ARD kernel
		if isfield(koption,'ARD') && koption.ARD
			if koption.LogTheta
				f_theta =@(x) log(1+exp(x));
			else
				f_theta =@(x) x.^2;
			end
			ard_K= @(x,y) hypar(koption.ARDIndex(1)) * ...
				exp (-.5 * sqdist(x', y',f_theta(diag(hypar((koption.ARDIndex+1):end)))));
			koption.ARDKernelMat{1} = ard_K(model.X,model.X);
			koption.ARDKernelMat{2} = ard_K(model.X,Xt);
			koption.ARDKernelMat{3} = ard_K(Xt,Xt);
		end
		
		switch model.TrainMethod
			case {'MAP','MKL'}
				[mean_p(:,d), var_p(:,d)]=predictSub(Z(:,d), Y(:,d,user_id), Zt, hypar, koption, model.TrainMethod);
			case 'MLE'
				var_p(:,d)=repmat(hypar(3),size(Zt));
				mean_p(:,d)=Zt*(hypar(1))+hypar(2);
		end
	end
end

end

function [mean_p, var_p]=predictSub(x_train, y_train, x_test, hypar, koption, tr_method)

if tr_method=='MKL'
	pKf=@predMKLKernel;
else
	pKf=@predKernel;
end
	

K = pKf(x_train, x_train, hypar, koption);
K1= pKf(x_train, x_test, hypar, koption);
K2= pKf(x_test, x_test, hypar, koption);

if isfield(koption,'ARDKernelMat')
	K=K+koption.ARDKernelMat{1};
	K1=K1+koption.ARDKernelMat{2};
	K2=K2+koption.ARDKernelMat{3};
end


KL=jitChol(K)';
alpha = KL'\(KL\y_train);
mean_p= K1'*alpha;
v  = KL\K1;
var_p = diag(K2-v'*v);
end