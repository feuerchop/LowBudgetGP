function [KL, dP, dinV]=calcMKLKernel(hypar, mat_delta, mat_linear, mat_sqrt, in_V, out_V, options, returnKernel)
% compute multiple kernel and corresponding gradient.
% mat_delta: the matrix which gives 0 if two instances are not same,
% otherwise 1
% mat_linear: the matrix which gives X*X';
% mat_sqrt: the square distance between X
% coefficient:
% hypar(1): delta noise
% hypar(2): bias
% hypar(3): linear
% hypar(4:end): rbf
%
% Used options
% options.Logtheta
% options.width is a vector for instance [2^-5, 2^5]

N=size(mat_linear,1);

%declare some functions
if options.LogTheta
	f_theta =@(x) log(1+exp(x));
else
	f_theta =@(x) x^2;
end
% f_ard = @(x,y) f_theta(x(1)) * ;

%WARNING ONLY ON 1 DIMENSION
if isfield(options,'InitKernel')
    K=options.InitKernel{1};
else
    K=zeros(N,N);
end
%

allKernels=cell(1,length(hypar)-3);

% add three kernels
K = K+ f_theta(hypar(1))*mat_delta + f_theta(hypar(2))+ f_theta(hypar(3))*mat_linear;
for j = 4:length(hypar)
	allKernels{j-3}=exp(-mat_sqrt/options.Width(j-3));
	K = K + f_theta(hypar(j))*allKernels{j-3};
end

% if isfield(options,'ARD') && options.ARD
% 		% ard kernel
% 		v_ard = exp (-.5* sqdist(options.ARDSource', options.ARDSource',f_theta(diag(hypar((options.ARDIndex+1):end)))));
% 		K= K + f_theta(hypar(options.ARDIndex))*v_ard;
% end

if returnKernel && nargout==1
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
	
	% gradients
	dP(1) = g_com(hypar(1), mat_delta);
	dP(2) = g_com(hypar(2), ones(N));
	dP(3) = g_com(hypar(3), mat_linear);
	
	for j = 4:length(hypar)
		dP(j) = g_com(hypar(j), allKernels{j-3});
	end
	
% 	if isfield(options,'ARD') && options.ARD
% 		% compute gradient w.r.t ard hyperparameters
% 		dP(options.ARDIndex)= g_com(hypar(options.ARDIndex), v_ard);
% 		for d=1:size(options.ARDSource,2)
% 			dP(options.ARDIndex+d) = g_com(hypar(options.ARDIndex+d), (f_theta(hypar(options.ARDIndex)) * v_ard.* (-.5 * options.ARDSqdist{d})));
% 		end
% 	end
	
	if nargout>2
		% return the gradient w.r.t to latent variable (in_V)
		dinV=zeros(N,1);
		
		% gradient w.r.t latent variable in rbfs
		for j = 4:length(hypar)
			dLdK1 = dLdK.*(f_theta(hypar(j))*allKernels{j-3});
			dKdVall= - bsxfun(@minus, in_V', in_V) / options.Width(j-3);
			for n=1:N
				dinV(n) = dinV(n) + sum(bsxfun(@times, dLdK1(:,n), dKdVall(:,n)), 1);
			end
		end
		dinV = 2*dinV;
		
		
		dLdK2 = dLdK*(f_theta(hypar(3)));
		for n=1:N
			dinV(n) = dinV(n) + sum(bsxfun(@times, dLdK2(:,n), in_V), 1);
		end
		
% 		if options.AddTerms(end)=='1' % has linear term
% 			lnIdx=length(hypar);
% 			if isfield(options,'ARD') && options.ARD
% 				lnIdx=options.ARDIndex-1;
% 			end
% 			dLdK2 = dLdK*(f_theta(hypar(lnIdx))); % always the last hyperparameter
% 			for n=1:N
% 				dinV(n) = dinV(n) + sum(bsxfun(@times, dLdK2(:,n), in_V), 1);
% 			end
% 		end
		
		dinV = 2*dinV;
	end
end

end