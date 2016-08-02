
function result = lvmob3Create(X,Y,k_instance,k_observer,k_latent,iters,is_approx)

if nargin<4 || isempty(k_instance)
    k_instance={'rbf', 'bias','white'};
end
if nargin<5 || isempty(k_observer)
    k_observer={'lin','bias','white'};
end
if nargin<6 || isempty(iters)
    iters=2000;
end
if nargin<7 || isempty(is_approx)
    is_approx=0;
end
% clc;clear;close all;
%
% X=rand(10,6);
% Y=rand(10,3);
% k_instance={'rbf', 'lin','bias','white'};
% k_observer={'rbf', 'lin','bias','white'};
% k_latent={'rbf', 'whitefixed'};
% iters=2000;
% is_approx=0;
% q=1;
display=1;

L=size(X,2);
M=size(Y,2);
N=size(X,1);

% declare feature group
featureIndGroups = {[(L+1):(L+M)]};

% declare node and latent dimension structure
nodeStructure = {1};
latentDim=[1,L];

% init option
options = hgplvmOptions(latentDim, featureIndGroups, nodeStructure,is_approx);
% options.tree(2).featureInd=[1:L];

if hasInfNaN(Y)
    disp('has missing value in response');
    options.tree(1).hasMissingData=1;
end
%%
model.type = 'hgplvm';
model.optimiser = options.optimiser;

% set the leaf node (observers)
if isfield(options.tree(1),'dataInd') && ~isempty(options.tree(1).dataInd)
    dataInd = options.tree(1).dataInd;
else
    dataInd = 1:N;
end

YNode = Y(dataInd,:);
optionsNode = options.tree(1).fgplvmOptions;
optionsNode.fixedX=0;


if isfield(options.tree(1),'hasMissingData') && ~isempty(options.tree(1).hasMissingData)
    optionsNode.isMissingData=1;
    optionsNode.isSpherical=0;
end
model.node(1) = fgplvmCreate(1, M, YNode, optionsNode);
model.node(1).prior = [];
model.node(1).dynamics = [];
model.tree(1) = rmfield(options.tree(1), 'fgplvmOptions');


% set root node
if isfield(options.tree(2),'dataInd') && ~isempty(options.tree(2).dataInd)
    dataInd = options.tree(2).dataInd;
else
    dataInd = 1:N;
end

YNode = model.node(1).X(dataInd, :);

optionsNode = options.tree(2).fgplvmOptions;
if isfield(options.tree(2),'hasMissingData') && ~isempty(options.tree(2).hasMissingData)
    optionsNode.isMissingData=1;
    optionsNode.isSpherical=0;
end

optionsNode.initX=X;
optionsNode.fixedX=1;

model.node(2) = fgplvmCreate(L, 1, YNode, optionsNode);
model.node(2).prior = [];
model.node(2).dynamics = [];
model.tree(2) = rmfield(options.tree(2), 'fgplvmOptions');




model.numParams = 0;
for i = 1:length(model.node)
    model.numParams = model.numParams + model.node(i).numParams;
end

%model = hgplvmCreate(latentDim, size(inputData,2), inputData, options);
%%
% set kernel for nodes
for i = 1:length(model.node)
    origNparams = model.node(i).kern.nParams;
    switch i
        case 1
            % set up kernel for instance
            model.node(i).kern = kernCreate(model.node(i).X, k_observer);
        case 2
            % set up kernel for response
            model.node(i).kern = kernCreate(model.node(i).X, k_instance);
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
Zc=model.node(1).X; % groundtruth score is in the 2nd node
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