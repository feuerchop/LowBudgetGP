function [KL, dP, dinV]=calcKernel(in_D, hypar, in_V, out_V, options, preK)
% compute the squared exponential kernel and gradient for HGP
% TYPE 0: k(x,y)=theta1*exp(-(x-y)^2/(2*theta2^2))
% AddTerms: [hasbias, hasnoise, haslinear]
% hyperparameters: 3 ~ 5

N=size(in_D,1);

in_knock = (in_D==0);
% note that if kappa2 is too small, then everything is uncorrelated, the result could be bad and unstable

%declare some functions
if options.LogTheta
	f_theta =@(x) log(1+exp(x));
else
	f_theta =@(x) x.^2;
end
% f_ard = @(x,y) f_theta(x(1)) * ;

if nargin<=5
	% compute noise diagonal matrix
	K = f_theta(hypar(1))*in_knock;
	% add additional terms
	switch bin2dec(options.AddTerms)
		case 0
			error('no kernel function! add more kernel functions!')
		case 1
			% [exp, bias, LINEAR]
			v_linear = in_V*in_V';
			K = K + f_theta(hypar(2))*v_linear;
		case 2
			% [exp, BIAS, linear]
			K= K + f_theta(hypar(2));
		case 3
			% [exp, BIAS, LINEAR]
			v_linear = in_V*in_V';
			K= K + f_theta(hypar(2)) + f_theta(hypar(3))*v_linear;
		case 4
			% [EXP, bias, linear]
			v_exp = exp(-in_D/(2*f_theta(hypar(3))));
			K= K + f_theta(hypar(2))*v_exp;
		case 5
			% [EXP, bias, LINEAR]
			v_exp = exp(-in_D/(2*f_theta(hypar(3))));
			v_linear = in_V*in_V';
			K = K + f_theta(hypar(2))*v_exp + f_theta(hypar(4))*v_linear;
		case 6
			% [EXP, BIAS, linear]
			v_exp = exp(-in_D/(2*f_theta(hypar(3))));
			K= K + f_theta(hypar(2))*v_exp + f_theta(hypar(4));
		case 7
			v_exp = exp(-in_D/(2*f_theta(hypar(3))));
			v_linear = in_V*in_V';
			K = K + f_theta(hypar(2))*v_exp + f_theta(hypar(4))+ f_theta(hypar(5))*v_linear;
	end
	
	if isfield(options,'ARD') && options.ARD
		% ard kernel
		v_ard = exp (-.5* sqdist(options.ARDSource', options.ARDSource',f_theta(diag(hypar((options.ARDIndex+1):end)))));
		K= K + f_theta(hypar(options.ARDIndex))*v_ard;
	end
else
	K=preK;
end

if options.ReturnKernel && nargout==1
	KL=K;
	return
end

% BUT return the lower triangular cholesky decomp.
KL = jitChol(K)';

if nargout>1
	dP=zeros(size(hypar));
	
	% return the gradient w.r.t to hyperparameters
	tmpVar1 = KL'\(KL\(out_V*out_V'))- eye(N);
	invKL=inv_triu(KL')';
	dLdK=tmpVar1*(invKL'*invKL);
	
	% declare some gradients
	if options.LogTheta
		g_theta = @(x) exp(x)/(1+exp(x));
	else
		g_theta = @(x) 2*x;
	end
	g_com = @(x,y) g_theta(x) * trace(dLdK * y);
	% 	g_bias = @(x) g_com(x, ones(N));
	% 	g_linear = @(x) g_com(x, v_linear);
	% 	g_exp1 = @(x,y) g_com(x, v_exp);
	% 	g_exp2 = @(x,y) g_com(y, f_theta(hypar(x))*v_exp.*in_D)/ (2*f_theta(y)^2);
	% 	g_ard1 = @(x,y) g_com(x(1), v_ard);
	% 	g_ard2 = @(x,y,d) g_com(x(d+1), (f_theta(hypar(options.ARDIndex))*v_ard.* (-.5 * sqdist(y(:,d)',y(:,d)'))));
	
	% gradient for hyperpar of noise
	dP(1) = g_theta(hypar(1)) * trace(dLdK * in_knock);
	
	% compute the gradient for additional terms
	switch bin2dec(options.AddTerms)
		case 1
			% [exp, bias, LINEAR]
			if ~exist('v_linear','var')
				v_linear = in_V*in_V';
			end
			dP(2) = g_com(hypar(2), v_linear);
		case 2
			% [exp, BIAS, linear]
			dP(2) = g_com(hypar(2), ones(N));
		case 3
			% [exp, BIAS, LINEAR]
			if ~exist('v_linear','var')
				v_linear = in_V*in_V';
			end
			dP(2) = g_com(hypar(2), ones(N));
			dP(3) = g_com(hypar(3), v_linear);
		case 4
			% [EXP, bias, linear]
			if ~exist('v_exp','var')
				v_exp = exp(-in_D/(2*f_theta(hypar(3))));
			end
			dP(2) = g_com(hypar(2), v_exp);
			dP(3) = g_com(hypar(3), f_theta(hypar(2))*v_exp.*in_D)/ (2*f_theta(hypar(3))^2);
		case 5
			% [EXP, bias, LINEAR]
			if ~exist('v_linear','var')
				v_linear = in_V*in_V';
			end
			if ~exist('v_exp','var')
				v_exp = exp(-in_D/(2*f_theta(hypar(3))));
			end
			dP(2) = g_com(hypar(2), v_exp);
			dP(3) = g_com(hypar(3), f_theta(hypar(2))*v_exp.*in_D)/ (2*f_theta(hypar(3))^2);
			dP(4) = g_com(hypar(4), v_linear);
		case 6
			% [EXP, BIAS, linear]
			if ~exist('v_exp','var')
				v_exp = exp(-in_D/(2*f_theta(hypar(3))));
			end
			dP(2) = g_com(hypar(2), v_exp);
			dP(3) = g_com(hypar(3), f_theta(hypar(2))*v_exp.*in_D)/ (2*f_theta(hypar(3))^2);
			dP(4) = g_com(hypar(4), ones(N));
		case 7
			if ~exist('v_linear','var')
				v_linear = in_V*in_V';
			end
			if ~exist('v_exp','var')
				v_exp = exp(-in_D/(2*f_theta(hypar(3))));
			end
			dP(2) = g_com(hypar(2), v_exp);
			dP(3) = g_com(hypar(3), f_theta(hypar(2))*v_exp.*in_D)/ (2*f_theta(hypar(3))^2);
			dP(4) = g_com(hypar(4), ones(N));
			dP(5) = g_com(hypar(5), v_linear);
	end
	
	if isfield(options,'ARD') && options.ARD
		% compute gradient w.r.t ard hyperparameters
		dP(options.ARDIndex)= g_com(hypar(options.ARDIndex), v_ard);
		for d=1:size(options.ARDSource,2)
			dP(options.ARDIndex+d) = g_com(hypar(options.ARDIndex+d), (f_theta(hypar(options.ARDIndex)) * v_ard.* (-.5 * options.ARDSqdist{d})));
		end
	end
	
	if nargout>2
		% return the gradient w.r.t to latent variable (in_V)
		dinV=zeros(N,1);
		
		if options.AddTerms(1)=='1' % has exp term
			dLdK1 = dLdK.*(f_theta(hypar(2))*v_exp); %always the 2nd, 3rd parameter
			dKdVall= - bsxfun(@minus, in_V', in_V) / (f_theta(hypar(3)));
			for n=1:N
				dinV(n) = dinV(n) + sum(bsxfun(@times, dLdK1(:,n), dKdVall(:,n)), 1);
			end
		end
		
		if options.AddTerms(end)=='1' % has linear term
			lnIdx=length(hypar);
			if isfield(options,'ARD') && options.ARD
				lnIdx=options.ARDIndex-1;
			end
			dLdK2 = dLdK*(f_theta(hypar(lnIdx))); % always the last hyperparameter
			for n=1:N
				dinV(n) = dinV(n) + sum(bsxfun(@times, dLdK2(:,n), in_V), 1);
			end
		end
		
		dinV = 2*dinV;
	end
end

end