%% Initialize a structure for kernel options
% Usage:
%   koption = koptDefault(Y)
%
% Input:
%   Y           a N x D x M response tensor, where N is the number of
%               instance, D is the dimension of response. M is the number
%               of sources.
% Output:
%   koption               a structure of kernel option
%   .TrainMethod         'MAP', 'MLE'
%   .InstanceKernel      setup for instance kernel  
%   ..AddTerms           ['001'] to '111', turn on/off the 
%                         + squared exponential function
%                         + bias (constant) function
%                         + linear function
%   ..ReturnKernel       [0] or 1, return kernel matrix in each iteration
%   ..RandInit           [0] or 1, initialize the hyperparameters as random 
%   ..LogTheta           [0] or 1, reparamtermize with log(1+exp(theta)) or
%                         theta^2, default is theta^2
%
%   .UserKernel          setup for user kernel, see above
%
%   .Regularization      setup for regularization
%   ..Mode                'L1', 'L2'
%   ..UserCov             a M x M matrix represents the similarity of
%                         M sources
%   ..DimCov              a D x D matrix represents the similarity of D
%                         dimensions
%   ..Lambda              a 1 x M vector represents the uncertainty of M
%                         sources, default is ones(1,M). Smaller value
%                         means higher confidence about the source
%
%   .InitZ                initial value of latent groundtruth, if empty
%                         then follow the standard initialization procedure
%   
%   .InitKappa            initial value of kappa, if empty then follow the 
%                         standard initialization procedure

function koption = koptDefault(Y)
[~,D,M]=size(Y);

koption.InstanceKernel.AddTerms='111';
koption.InstanceKernel.ReturnKernel=0;
koption.InstanceKernel.RandInit=0;
koption.InstanceKernel.LogTheta=0;
koption.InstanceKernel.Width=10;
koption.InstanceKernel.Regularization='L1';
koption.InstanceKernel.RegC=10;


koption.UserKernel.AddTerms='111';
koption.UserKernel.ReturnKernel=0;
koption.UserKernel.RandInit=0;
koption.UserKernel.LogTheta=0;
koption.UserKernel.Width=10;
koption.UserKernel.Regularization='L1';
koption.UserKernel.RegC=10;
koption.UserKernel.ARD=0;

% for linearGPLVM
koption.Regularization.Mode='L2';
koption.Regularization.UserCov=eye(M);
koption.Regularization.DimCov=eye(D);
koption.Regularization.Lambda=ones(1,M);

koption.InitZ=[];
koption.InitKappa=[];

koption.UserConfidence=ones(1,M); 
koption.TrainMethod='MAP';

end