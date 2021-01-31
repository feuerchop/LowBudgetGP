function K=predMKLKernel(in_V1, in_V2, hypar, options)
% compute the squared exponential kernel and gradient for HGP
% TYPE 0: k(x,y)=theta1*exp(-(x-y)^2/(2*theta2^2))
% AddTerms: [hasbias, hasnoise, haslinear]
% hyperparameters: 3 ~ 5

mat_sqrt=sqdist(in_V1',in_V2');
mat_linear=in_V1*in_V2';
mat_delta = (mat_sqrt==0);

% note that if kappa2 is too small, then everything is uncorrelated, the result could be bad and unstable

%declare some functions
if options.LogTheta
	f_theta =@(x) log(1+exp(x));
else
	f_theta =@(x) x.^2;
end

% add three kernels
K = f_theta(hypar(1))*mat_delta + f_theta(hypar(2))+ f_theta(hypar(3))*mat_linear;
for j = 4:length(hypar)
	K = K + f_theta(hypar(j))*exp(-mat_sqrt/options.Width(j-3));
end

end