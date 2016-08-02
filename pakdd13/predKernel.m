function K=predKernel(in_V1, in_V2, hypar, options)
% compute the squared exponential kernel and gradient for HGP
% TYPE 0: k(x,y)=theta1*exp(-(x-y)^2/(2*theta2^2))
% AddTerms: [hasbias, hasnoise, haslinear]
% hyperparameters: 3 ~ 5

in_D=sqdist(in_V1',in_V2');
in_knock = (in_D==0);
% note that if kappa2 is too small, then everything is uncorrelated, the result could be bad and unstable

%declare some functions
if options.LogTheta
	f_theta =@(x) log(1+exp(x));
else
	f_theta =@(x) x.^2;
end
f_bias = @(x) f_theta(x);
f_linear = @(x) f_theta(x) * (in_V1*in_V2');
f_exp = @(x,y) f_theta(x) * exp(-in_D/(2*f_theta(y)));

% compute noise diagonal matrix
K = f_theta(hypar(1))*in_knock;
% add additional terms
switch bin2dec(options.AddTerms)
	case 0
		error('Dont use noise kernel only! Add more kernel functions!');
	case 1
		% [exp, bias, LINEAR]
		K=K+f_linear(hypar(2));
	case 2
		% [exp, BIAS, linear]
		K=K+f_bias(hypar(2));
	case 3
		% [exp, BIAS, LINEAR]
		K=K+f_bias(hypar(2))+f_linear(hypar(3));
	case 4
		% [EXP, bias, linear]
		K=K+f_exp(hypar(2),hypar(3));
	case 5
		% [EXP, bias, LINEAR]
		K=K+f_exp(hypar(2),hypar(3))+f_linear(hypar(4));
	case 6
		% [EXP, BIAS, linear]
		K=K+f_exp(hypar(2),hypar(3))+f_bias(hypar(4));
	case 7
		K=K+f_exp(hypar(2),hypar(3))+f_bias(hypar(4))+f_linear(hypar(5));
end
end