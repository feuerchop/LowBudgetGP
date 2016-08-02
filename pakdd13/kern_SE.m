function [KL, dP, dinV]=kern_SE(in_D, hypar, in_V, out_V, options)
% compute the squared exponential kernel and gradient for HGP
% TYPE 0: k(x,y)=theta1*exp(-(x-y)^2/(2*theta2^2))
% AddTerms: [hasbias, hasnoise, haslinear]
% hyperparameters: 3 ~ 5

N=size(in_D,1);

in_knock = (in_D==0);
% note that if kappa2 is too small, then everything is uncorrelated, the result could be bad and unstable
in_exp = exp(-in_D/(2*hypar(2)^2));

%declare some functions
f_bias = @(x) x^2;
f_linear= @(x) x^2*(in_V*in_V');



% compute standard SE kernel matrix
K = hypar(1)^2*in_exp + hypar(3)^2*in_knock;
% add additional terms
switch bin2dec(options.AddTerms)
	case 1
		% [nobias, HASLINEAR]
		K=K+f_linear(hypar(4));
	case 2
		% [HASBIAS, nolinear]
		K=K+f_bias(hypar(4));
	case 3
		% [HASBIAS, HASLINEAR]
		K=K+f_bias(hypar(4))+f_linear(hypar(5));
end

if options.ReturnKernel
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
	
	dP(1) = 2 * hypar(1) * trace(dLdK * in_exp);
	dP(2) = hypar(1)^2 * (hypar(2)^-3) * trace(dLdK * (in_exp.*in_D));
	dP(3) = 2 * hypar(3) * trace(dLdK * in_knock);
	% declare some gradients
	g_bias = @(x) 2*x*trace(dLdK * ones(N));
	g_linear= @(x) 2*x*trace(dLdK * (in_V*in_V'));
	
	% compute the gradient for additional terms
	switch bin2dec(options.AddTerms)
		case 1
			% [nobias, HASLINEAR]
			dP(4)=g_linear(hypar(4));
			dLdK2 = dLdK*(hypar(4)^2);
		case 2
			% [HASBIAS, nolinear]
			dP(4)=g_bias(hypar(4));
		case 3
			% [nobias, HASNOISE, HASLINEAR]
			% 			K=K+hypar(3)^2*in_knock+hypar(4)^2*(in_V*in_V');
			dP(4)=g_bias(hypar(4));
			dP(5)=g_linear(hypar(5));
			dLdK2 = dLdK*(hypar(5)^2);
	end
	
	if nargout>2
		% return the gradient w.r.t to latent variable (in_V)
		dinV=zeros(N,1);
		dLdK1 = dLdK.*(hypar(1)^2*in_exp);
		dKdVall= - bsxfun(@minus, in_V', in_V) / (hypar(2)^2);
		
		for n=1:N
			dinV(n) = sum(bsxfun(@times, dLdK1(:,n), dKdVall(:,n)), 1);
		end
		
		if options.AddTerms(end)=='1' % has linear term
			for n=1:N
				dinV(n) = dinV(n) + sum(bsxfun(@times, dLdK2(:,n), in_V), 1);
			end
		end
		
		dinV = 2*dinV;
	end
end

end