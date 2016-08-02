function kval = k_rbf(u,v,rbf_sigma,varargin)
%RBF_KERNEL Radial basis function kernel for SVM functions

% Copyright 2004-2010 The MathWorks, Inc.
% $Revision: 1.1.12.4.14.2 $  $Date: 2011/03/17 22:25:59 $

 
if nargin < 3 || isempty(rbf_sigma)
    rbf_sigma = 1;
else
     if ~isscalar(rbf_sigma) || ~isnumeric(rbf_sigma)
        error('Bioinfo:rbf_kernel:RBFSigmaNotScalar','The rbf_sigma be a numeric scalar.');
    end
    if rbf_sigma == 0
        error('Bioinfo:rbf_kernel:SigmaZero','The rbf_sigma must be non-zero.');
    end
    
end



kval = exp(-(1/(2*rbf_sigma^2))*(repmat(sqrt(sum(u.^2,2).^2),1,size(v,1))...
    +repmat(sqrt(sum(v.^2,2)'.^2),size(u,1),1)-2*(u*v')));