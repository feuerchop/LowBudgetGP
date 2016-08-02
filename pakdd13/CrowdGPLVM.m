function [Z,theta,kappa,fval] = CrowdGPLVM(N,M,D,X,Y,koption)

[Z0, theta0, kappa0, lb, ub, koption] = iniCrowdGPLVM(N, M, D, [0,1], koption);
auxdata={N M D X Y sqdist(X',X') koption};
x0=[Z0(:);theta0(:);kappa0(:)];

if ~all(x0<=ub & x0>=lb)
    error('bad initialization, out of boundary!');
end

if exist('lbfgsb')
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
    [x,fval,~] = lbfgsb( {f,g} , lb, ub, opts );
else
    warning('Can''t find L-BFGS, internal fmincon is used. VERY SLOW!');
    options = optimset('GradObj','on','Display','iter','FunValCheck','on','DerivativeCheck','on','Diagnostics','on'); % indicate gradient is provided
    %options = optimset('GradObj','on','Display','iter'); % indicate gradient is provided
    [x,fval] = fmincon(@(x) wrapperCrowdGPLVM(x,auxdata),x0,[],[],[],[],lb,ub,[],options);
end

[Z, theta, kappa]=reshapePars(x, N, M, D, koption.UserKernel.NumPars, koption.InstanceKernel.NumPars);
end

function [f,g]=wrapperCrowdGPLVM(x,auxdata)

f = objCrowdGPLVM(x, auxdata);
g = gradCrowdGPLVM(x, auxdata);

end

