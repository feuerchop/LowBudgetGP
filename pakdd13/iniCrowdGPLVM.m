function [Z, theta, kappa, lb, ub, koption] = iniCrowdGPLVM(X, Y, koption)

% 	function out = setOpts( f1, f2, default)
% 		if ~isfield( koption.(f1), f2 )
% 			koption.(f1).(f2)= default;
% 			fprintf('%s.%s is set to %s by default\n', f1,f2,num2str(default));
% 		end
% 		out = koption;
% 	end

[N,D,M]= size(Y);

% printBlock('Kernel options');
% koption = setOpts( 'InstanceKernel','AddTerms', '111');
% koption = setOpts( 'InstanceKernel','ReturnKernel', 0);
% koption = setOpts( 'InstanceKernel','RandInit', 0);
% koption = setOpts( 'InstanceKernel','LogTheta', 0);
% koption = setOpts( 'UserKernel','AddTerms', '111');
% koption = setOpts( 'UserKernel','ReturnKernel', 0);
% koption = setOpts( 'UserKernel','RandInit', 0);
% koption = setOpts( 'UserKernel','LogTheta', 0);
% koption = setOpts( 'Regularization','Mode', 0);
% koption = setOpts( 'Regularization','UserCov', eye(M));
% koption = setOpts( 'Regularization','DimCov', eye(D));
% koption = setOpts( 'Regularization','Lambda', ones(1,M));

% initialize the parameters of CrowdGPLVM

boundZ = [min(Y(:)),max(Y(:))];
% init user kernel
boundHypar=[-inf, inf];

switch koption.TrainMethod
    case 'MAP'
        koption.UserKernel= setNumberParameter(X, koption.UserKernel);
        koption.InstanceKernel= setNumberParameter(X, koption.InstanceKernel);
    case 'MLE'
        disp('restrict to linear function');
        koption.UserKernel.NumPars=3;
        koption.InstanceKernel.ReturnKernel=1;
        koption.InstanceKernel= setNumberParameter(X, koption.InstanceKernel);
        if ~isfield(koption,'eta')
            koption.eta=0.5;
        end
    case 'MKL'
        if koption.InstanceKernel.Width>0
            koption.InstanceKernel.Width=2.^[-koption.InstanceKernel.Width:koption.InstanceKernel.Width];
        else
            koption.InstanceKernel.Width=[];
        end
        koption.InstanceKernel.NumPars=length(koption.InstanceKernel.Width)+3;
        
        if koption.UserKernel.Width>1
            koption.UserKernel.Width=2.^[-koption.UserKernel.Width:koption.UserKernel.Width];
        else
            koption.UserKernel.Width=[];
        end
        koption.UserKernel.NumPars=length(koption.UserKernel.Width)+3;
        
    otherwise
        error('please choose from : ''[MAP]'',''MLE'',''MKL''.')
end

%%
printBlock('user kernel');

% set UL bound
lb_theta=boundHypar(1)*ones(1, koption.UserKernel.NumPars);
ub_theta=boundHypar(2)*ones(1, koption.UserKernel.NumPars);
lb_theta=repmat(lb_theta, [D, 1, M]);
ub_theta=repmat(ub_theta, [D, 1, M]);

if strcmp(koption.TrainMethod,'MKL') && ~isempty(koption.UserKernel.Width)
    fprintf('width from %.3f to %.3f (%d in total)\n',koption.UserKernel.Width(1),koption.UserKernel.Width(end), length(koption.UserKernel.Width));
    fprintf('regularization: %s\npenalty: %d\n', koption.UserKernel.Regularization, koption.UserKernel.RegC);
end
% parameters
if koption.UserKernel.RandInit
    theta=rand(D,koption.UserKernel.NumPars,M).*(ub_theta-lb_theta)+lb_theta;
    disp('initialized as random')
else
    switch koption.TrainMethod
        case {'MAP','MKL'}
            if koption.UserKernel.LogTheta
                theta=zeros(D,koption.UserKernel.NumPars,M);
                disp('reparameterize by log(1+exp(x))');
                disp('initialized as zero');
            else
                theta=ones(D,koption.UserKernel.NumPars,M);
                disp('reparameterize by x^2');
                disp('initialized as one');
            end
        case 'MLE'
            theta=zeros(D,koption.UserKernel.NumPars,M);
            disp('initialized as zero');
    end
end

%%
% init instance kernel
printBlock('instance kernel');

% set UL bound
lb_kappa=boundHypar(1)*ones(1, koption.InstanceKernel.NumPars);
ub_kappa=boundHypar(2)*ones(1, koption.InstanceKernel.NumPars);
lb_kappa=repmat(lb_kappa, [D, 1]);
ub_kappa=repmat(ub_kappa, [D, 1]);

% bounds of parameters
lb_Z=repmat(boundZ(1),[N,D]);
ub_Z=repmat(boundZ(2),[N,D]);

if strcmp(koption.TrainMethod,'MKL') && ~isempty(koption.InstanceKernel.Width)
    fprintf('width from %.3f to %.3f (%d in total)\n',koption.InstanceKernel.Width(1),koption.InstanceKernel.Width(end), length(koption.InstanceKernel.Width));
    fprintf('regularization: %s\npenalty: %d\n', koption.InstanceKernel.Regularization, koption.InstanceKernel.RegC);
end


if isempty(koption.InitZ)
    % latent ground truth
    % Z=rand(N,D).*(boundZ(2)-boundZ(1))+boundZ(1);
    Z=mean(Y,3);
else
    Z=koption.InitZ;
end

if isempty(koption.InitKappa)
    if koption.InstanceKernel.RandInit
        kappa=rand(D,koption.InstanceKernel.NumPars).*(ub_kappa-lb_kappa)+lb_kappa;
        disp('initialized as random')
    else
        if koption.InstanceKernel.LogTheta
            kappa=zeros(D, koption.InstanceKernel.NumPars);
            disp('reparameterize by log(1+exp(x))');
            disp('initialized as zero')
        else
            kappa=ones(D, koption.InstanceKernel.NumPars);
            disp('reparameterize by x^2');
            disp('initialized as one')
        end
    end
else
    kappa=koption.InitKappa;
end

Z=addJitter(Z,1e-1);
% make sure Z is in the boundary
Z=0.9*(Z-min(Z))/(max(Z)-min(Z))*(boundZ(2)-boundZ(1))+boundZ(1);
theta=addJitter(theta,1e-1);
kappa=addJitter(kappa,1e-1);


printBlock('input summary');
fprintf('# instances: %d\n# sources: %d\n# dim. of groundtruth: %d\n', N,M,D);
fprintf('bound of hyperparameters: [%.2f,%.2f]\n',boundHypar(1),boundHypar(2));
fprintf('bound of latent variables: [%.2f,%.2f]\n',boundZ(1),boundZ(2));

koption.Regularization.Lambda=koption.UserConfidence.^2;%(max(koption.UserConfidence)./koption.UserConfidence).^3;

switch koption.TrainMethod
    case 'MAP'
        lb        = [lb_Z(:); lb_theta(:); lb_kappa(:)];
        ub        = [ub_Z(:); ub_theta(:); ub_kappa(:)];
        disp('maximum a posterior');
    case 'MLE'
        lb        = [lb_theta(:); lb_kappa(:)];
        ub        = [ub_theta(:); ub_kappa(:)];
        fprintf('maximum likelihood with %s regularization\n', koption.Regularization.Mode);
    case 'MKL'
        lb        = [lb_Z(:); lb_theta(:); lb_kappa(:)];
        ub        = [ub_Z(:); ub_theta(:); ub_kappa(:)];
        disp('multiple kernel learning');
end

end

function y=addJitter(x, prec)
y=x+(rand(size(x))-0.5)*prec;
end