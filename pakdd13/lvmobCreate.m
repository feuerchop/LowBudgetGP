function result = lvmobCreate(X,Y,q,k_instance,k_observer,k_latent,iters,is_approx)

if nargin<4 || isempty(k_instance)
    k_instance={'rbf', 'lin','bias','white'};
end
if nargin<5 || isempty(k_observer)
    k_observer={'rbf', 'lin','bias','white'};
end
if nargin<6 || isempty(k_latent)
    k_latent={'rbf', 'whitefixed'};
end
if nargin<7 || isempty(iters)
    iters=2000;
end
if nargin<8 || isempty(is_approx)
    is_approx=0;
end

display=1;

L=size(X,2);
M=size(Y,2);

if q<1||q>L
    error(['number of latent dimension must in the range of [1,',num2str(L),']']);
end

% combining data
inputData=[X,Y];

% declare feature group
featureIndGroups = {[1:L], [(L+1):(L+M)]};

% declare node and latent dimension structure
nodeStructure = {[1,2]};
latentDim=[q,1,q];

% init option
options = hgplvmOptions(latentDim, featureIndGroups, nodeStructure,is_approx);

if hasInfNaN(Y)
    disp('has missing value in response');
    options.tree(2).hasMissingData=1;
end

model = hgplvmCreate(latentDim, size(inputData,2), inputData, options);

% set kernel for nodes
for i = 1:length(model.node)
    origNparams = model.node(i).kern.nParams;
    switch i
        case 1
            % set up kernel for instance
            model.node(i).kern = kernCreate(model.node(i).X, k_instance);
        case 2
            % set up kernel for response
            model.node(i).kern = kernCreate(model.node(i).X, k_observer);
        case 3
            model.node(i).kern = kernCreate(model.node(i).X, k_latent);
            model.node(i).kern.comp{2}.variance = 1e-6;
            model.node(i).kern.whiteVariance = 1e-6;
    end
    model.node(i).numParams = model.node(i).numParams - origNparams ...
        + model.node(i).kern.nParams;
end
model.numParams = 0;
for i = 1:length(model.node)
    model.numParams = model.numParams + model.node(i).numParams;
end

%%
disp('step 1: training latent variable multi-observer model')
model = hgplvmOptimise(model, display, iters);
Zc=model.node(2).X; % groundtruth score is in the 2nd node
result.lvmob=model;
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