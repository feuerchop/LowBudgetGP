%% Train a CrowdGPLVM model
%
% Usage:
%   [Z,theta,kappa,fval,koption] = trainCrowdGPLVM(X,Y)
%   [Z,theta,kappa,fval,koption] = trainCrowdGPLVM(X,Y,koption)
%
% Input:
%   X           a N x L instances matrix, where each row is an instance
%               with L-dimensional feature.
%   Y           a N x D x M response tensor, where N is the number of
%               instance, D is the dimension of response. M is the number
%               of sources.
%   [koption]   a stucture must contains two structures UserKernel and
%               InstanceKernel, which control the kernel of GP used in two
%               layes, respectively. One need setup the following properties 
%               koption.InstanceKernel.AddTerms          '001' to ['111']
%               koption.InstanceKernel.ReturnKernel      [0] or 1
%               koption.InstanceKernel.RandInit          [0] or 1
%               koption.InstanceKernel.LogTheta          [0] or 1
%   .AddTerms   a binary indicator string from '001' to '111', default '111'
%               1. turn on/off the squared exponential function
%               2. turn on/off the bias
%               3. turn on/off the linear function
% Output:
%   Z           a N x D groundtruth matrix, where D is the dimension of
%               response variable.
%   theta       a D x K x M tensor, the kernel hyparparmeters for UserKernel.
%               K is the number of hyparparmeters, which depends on the
%               type of kernel. M is the number of sources.
%   kappa       a D x K matrix, the kernel hyparparmeters for InstanceKernel.
%   fval        a scalar, the log-posterior value of the objective function 
%               (up to a constant factor) for the current model.
%   koption     kernel options
%
% Example:
%         M=2; % nb. of experts
%         N=40; % nb. of instances
%         D=1; % dim. of response (currently we only have 1 dim response, i.e. time interval)
%         L=100; % dim. of instance (i.e. nb. of features of an instance)
%         X=rand(N,L); % instance matrix N x L
%         Y=rand(N,D,M); % response matrix from all experts  N x D x M
% 
%         % set kernel options
%         koption.InstanceKernel.AddTerms='111';
%         koption.InstanceKernel.ReturnKernel=0;
%         koption.InstanceKernel.RandInit=0;
%         koption.InstanceKernel.LogTheta=0;
%         koption.UserKernel.AddTerms='111';
%         koption.UserKernel.ReturnKernel=0;
%         koption.UserKernel.RandInit=0;
%         koption.UserKernel.LogTheta=0;
%         % train the model
%         [Z, theta, kappa,fval]=trainCrowdGPLVM(X,Y,koption);
%
% See also PREDICTCROWDGPLVM
%
% Author:
% Han Xiao, Technical University of Munich
% xiaoh@in.tum.de 
function [Z,theta,kappa,fval,koption] = trainLinearGP(X,Y,koption)
if ~exist('lightspeed')
    error('Require Tom Minka''s LightSpeed toolbox!');
end

if nargin==2
	koption = struct('InstanceKernel',struct('ph',1),'UserKernel',struct('ph',1));
end


[Z0, theta0, kappa0, lb, ub, koption] = iniCrowdGPLVM(Y, koption);
[N,D,M]=size(Y);
auxdata={N M D X Y sqdist(X',X') koption};
x0=[Z0(:);theta0(:);kappa0(:)];

if ~all(x0<=ub & x0>=lb)
    error('bad initialization, out of boundary!');
end

if exist('lbfgsb')
    printBlock('lbfgsb started');
    % setup options for LBFGS
    opts    = struct( 'x0', x0 );
    opts.printEvery     = 100;
    opts.m  = 20;
    opts.maxIts = 1e4;
    opts.maxTotalIts =1e6;
    opts.pgtol      = 1e-10;
    opts.factr      = 1e3;
    f = @(x) objCrowdGPLVM(x, auxdata);
    g = @(x) gradCrowdGPLVM(x, auxdata);
    [x,fval] = lbfgsb( {f,g} , lb, ub, opts );
elseif exist('minConf_TMP')
    printBlock('minConf started');
    funObj = @(x) wrapperCrowdGPLVM(x,auxdata);
    [x,fval] = minConf_TMP(funObj,x0,lb,ub,[]);
else
    printBlock('slow warning');
    warning('Can''t find a faster L-BFGS implementation. Matlab ''fmincon'' will be used!\n Press any key to continue...');pause;
    %options = optimset('GradObj','on','Display','iter','FunValCheck','on','DerivativeCheck','on','Diagnostics','on'); % indicate gradient is provided
    options = optimset('GradObj','on','Display','iter'); % indicate gradient is provided
    [x,fval] = fmincon(@(x) wrapperCrowdGPLVM(x,auxdata),x0,[],[],[],[],lb,ub,[],options);
end

[Z, theta, kappa]=reshapePars(x, N, M, D, koption.UserKernel.NumPars, koption.InstanceKernel.NumPars);
end

function [f,g]=wrapperCrowdGPLVM(x,auxdata)

f = objCrowdGPLVM(x, auxdata);
g = gradCrowdGPLVM(x, auxdata);

end

