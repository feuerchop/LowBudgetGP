function l = infoPriorLogProb(prior, x)

% GAUSSIANPRIORLOGPROB Log probability of Gaussian prior.
%
% l = gaussianPriorLogProb(prior, x)
%

% Copyright (c) 2005 Neil D. Lawrence
% gaussianPriorLogProb.m version 1.3



% Compute log prior
L=jitChol(prior.K)';
l = -.5*(x'*(L'\(L\x)) + size(X,2)*log(2*pi) + 2*sum(log(diag(L))));
