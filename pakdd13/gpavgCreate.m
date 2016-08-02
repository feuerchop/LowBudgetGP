function result=gpavgCreate(X,Y,k_instance,k_observer,iters,is_approx)
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
disp('step 1: training model for ground truth prediction')
if is_approx
    options=gpOptions('fitc');
else
    options = gpOptions('ftc');
end
options.kern=k_instance;
Zc=nanmean(Y,2);
if hasInfNaN(Zc)
    options.isMissingData=1;
    options.isSpherical=0;
end

model = gpCreate(L,1,X,Zc,options);
model = gpOptimise(model, 0, iters);
result.gt=model;

%%
disp('step 2: training observer models')
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