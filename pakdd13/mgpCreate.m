function result=mgpCreate(X,Y,k_instance,k_observer,iters,is_approx)
if nargin<3 || isempty(k_instance)
    k_instance={'rbf', 'lin','bias','white'};
end
if nargin<4 || isempty(k_observer)
    k_observer={'rbf', 'lin','bias','white'};
end
if nargin<5 || isempty(iters)
    iters=2000;
end
if nargin<6 || isempty(is_approx)
    is_approx=0;
end

display=0;

L=size(X,2);
M=size(Y,2);

%% 
disp('step 1: training latent variable multi-observer model')
if is_approx
    options = fgplvmOptions('fitc');
else
    options = fgplvmOptions('ftc');
end
options.kern=k_instance;

if hasInfNaN(Y)
    options.isMissingData=1;
    options.isSpherical=0;
end

model = fgplvmCreate(1,M,Y,options);



% % Add dynamics model.
% options = gpOptions('ftc');
% options.kern = kernCreate(X, {'rbf', 'white'});
% options.kern.comp{1}.inverseWidth = 0.2;
% % This gives signal to noise of 0.1:1e-3 or 100:1.
% options.kern.comp{1}.variance = 0.1^2;
% options.kern.comp{2}.variance = 1e-3^2;
% model = fgplvmAddInfoPrior(model, 'gpFixed', X, options);

model = fgplvmOptimise(model, display, iters);

result.lvmob=model;
Zc=model.X;

%%
disp('step 2: training model for ground truth prediction')
if is_approx
    options=gpOptions('fitc');
else
    options = gpOptions('ftc');
end
options.kern=k_instance;
model = gpCreate(L,1,X,Zc,options);
model = gpOptimise(model, display, iters);
result.gt=model;


%%
disp('step 3: training observer models')
if is_approx
    options=gpOptions('fitc');
else
    options = gpOptions('ftc');
end
options.kern=k_observer;
for m=1:M
    if hasInfNaN(Y(:,m))
        options.isMissingData=1;
        options.isSpherical=0;
    end
    model = gpCreate(1,1,Zc,Y(:,m),options);
    model = gpOptimise(model, display, iters);
    result.obs{m}=model;
end


end