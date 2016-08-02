function [Z,theta,kappa] = CrowdGPLVMold(N,M,D,X,Y,koption)

[Z0, theta0, kappa0, lb, ub, koption] = iniCrowdGPLVM(N, M, D, [0,1], koption);
auxdata={N M D X Y sqdist(X',X') koption}; 

% setup options for LBFGS
maxiter   = 1e4;
m_lbfgs   = 20;
tolerance = 1e-9;

printBlock('lbfgs setup');
fprintf('max # iteration: %d\n', maxiter);
fprintf('min progress before stop: %f\n', tolerance);


x0=[Z0(:);theta0(:);kappa0(:)];

if ~all(x0<=ub & x0>=lb)
	error('bad initialization, out of boundary!');
end

printBlock('optimization');
disp('   iteration           log-likelihood');
% Run the limited-memory BFGS algorithm.
x = lbfgsb(x0, lb, ub,...
    'objCrowdGPLVM','gradCrowdGPLVM',...
    auxdata,'cbCrowdGPLVM',...
    'm',m_lbfgs,'factr',1e-3,'pgtol',...
    tolerance,'maxiter',maxiter);

[Z, theta, kappa]=reshapePars(x, N, M, D, koption.UserKernel.NumPars, koption.InstanceKernel.NumPars);

end
%{
maxiter   = 1e4;
m_lbfgs   = 20;
tolerance = 1e-6;
lb        = [.1 0 .1 .5];
ub        = [3 5 5 5];

[W0 mu0 sigma0 kappa0] = deal(inipoint{1:4});
[N M D X Y S] = deal(auxdata{1:6});

% Run the limited-memory BFGS algorithm.
[W, mu, sigma, kappa] = lbfgsb({W0, mu0, sigma0, kappa0},...
    {repmat(lb(1),M,D) repmat(lb(2),M,D) repmat(lb(3),M,1)  repmat(lb(4),3,1)},...
    {repmat(ub(1),M,D) repmat(ub(2),M,D) repmat(ub(3),M,1)  repmat(ub(4),3,1)},...
    'computeObjectiveCrowdsGPMAP','computeGradientsCrowdsGPMAP',...
    auxdata,'callbackCrowdsGPMAP',...
    'm',m_lbfgs,'factr',1e-3,'pgtol',...
    tolerance,'maxiter',maxiter);
%}
