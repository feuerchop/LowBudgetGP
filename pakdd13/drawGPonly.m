clc;

N=30;
D=1;
M=4;

xrange=[0,2*pi];
X=sort(rand(N,1))*(xrange(2)-xrange(1))+xrange(1);
Xt=(0:0.05:2*pi)';
Xt=setdiff(Xt,X);

% the groundtruth function (instance -> groundtruth)
% g=@(x) 10*sinc(x);
g=@(x) 10*sin(6*x).*sin(x/2);
% g=@(x) sin(x);

Z=g(X);
Y=zeros(N,D,M);
s_g=2*rand(1,M);
% s_g(4)=0;

mycolor=distinguishable_colors(M);
lcolor=gray(10);
mymarker={'o','s','+','x','*','d','v','^','<','>','p','h'};

% generate piecewise nonlinear function
nsteps=4;
reg_prob=0.8; % probabilty of a regular user;
exp_info=[];
for m=1:M
	[f,b] = genNLfunc2([min(Z),max(Z)],nsteps,reg_prob);
	exp_info=[exp_info;b];
	expert_func{m}=@(a) f(a); %w(1)*a; %abs(a);
end

for m=1:M
	Y(:,1,m)=randnorm(1,expert_func{m}(Z),s_g(m));
end

% GP baseline
subplot(1,3,2)
[mean_y,var_y]=gpbaseline(X,mean(Y,3),Xt);
shadedErrorBar(Xt',mean_y,var_y,{'k-'});
% 	hold on;plot(X,normaltoUnit(mean(Y,3)),'k*');hold off;
% plot(Xt,normaltoUnit(mean_y),'k--','LineWidth',2);
axis tight;
set(gca,'YTick',[min(mean_y), (max(mean_y)-min(mean_y))/2+min(mean_y), max(mean_y)]);
set(gca,'YTickLabel',{'0','0.5','1'});
axis([min(Xt),max(Xt),min(mean_y),max(mean_y)]);
box on;
mname{3}=sprintf('GP: %.2f',calcNMAE(mean_y,g(Xt)));
perfarray(2)=calcNMAE(mean_y,g(Xt));
title(mname{3});