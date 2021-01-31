%% GP baseline model
% [sel_id,sel_var,all_score]=gpactivelearning(X,Xl_id,Yl,k)
% input:
% X        all training instances
% Xl_id    labeled id
% Yl       labeled score
% k        return top k
% output
% sel_id   selected k indicies
% sel_var  variance of selected k instances
% all_scores scores of all instances
% all_var  variance of all instances
% f_gp     learnt GP, useage f_gp(X)

function [sel_id,sel_var,all_score,all_var,f_gp] = gpactivelearning(X,Xl_id,Yl,k)

if length(Xl_id)~=length(Yl)
    error('Yl must have the same length as Xl_id');
end

mydir='~/Documents/gpml-matlab-v3.1-2010-09-27/';
addpath(mydir(1:end-1))
addpath([mydir,'cov'])
addpath([mydir,'doc'])
addpath([mydir,'inf'])
addpath([mydir,'lik'])
addpath([mydir,'mean'])
addpath([mydir,'util'])

meanfunc = {@meanSum, {@meanLinear, @meanConst}};
likfunc = @likGauss;
covfunc = {'covSum', {'covLINone','covConst','covNoise','covSEiso'}};
hyp.cov = rand(eval(feval(covfunc{:})),1); hyp.mean = rand(size(X,2)+1,1); hyp.lik = log(0.1);
hyp = minimize(hyp, @gp, -100, @infExact, meanfunc, covfunc, likfunc, X(Xl_id,:), Yl);
[all_score, all_var] = gp(hyp, @infExact, meanfunc, covfunc, likfunc, X(Xl_id,:), Yl, X);

f_gp=@(x) gp(hyp, @infExact, meanfunc, covfunc, likfunc, X(Xl_id,:), Yl, x);
[sortedValues,sortIndex] = sort(all_var,'descend');

sel_id=[];sel_var=[];
for pk=1:length(sortIndex)
    if ~ismember(sortIndex(pk),Xl_id)
        sel_id=[sel_id;sortIndex(pk)];
        sel_var=[sel_var;sortedValues(pk)];
    end
    if length(sel_id)==k
        break;
    end        
end

rmpath(mydir(1:end-1))
rmpath([mydir,'cov'])
rmpath([mydir,'doc'])
rmpath([mydir,'inf'])
rmpath([mydir,'lik'])
rmpath([mydir,'mean'])
rmpath([mydir,'util'])
end