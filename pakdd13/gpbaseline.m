%% GP baseline model
% [m,s2]=gpbaseline(x,y,z)
% input:
% x        training instances
% y        training response
% z        test instances
% output
% m        predicted response on test instances
% s2       variance on each prediction

function [m,s2]=gpbaseline(x,y,z)
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
hyp.cov = zeros(eval(feval(covfunc{:})),1); hyp.mean = zeros(size(x,2)+1,1); hyp.lik = log(0.1);
hyp = minimize(hyp, @gp, -100, @infVB, meanfunc, covfunc, likfunc, x, y);
[m s2] = gp(hyp, @infExact, meanfunc, covfunc, likfunc, x, y, z);

rmpath(mydir(1:end-1))
rmpath([mydir,'cov'])
rmpath([mydir,'doc'])
rmpath([mydir,'inf'])
rmpath([mydir,'lik'])
rmpath([mydir,'mean'])
rmpath([mydir,'util'])
end