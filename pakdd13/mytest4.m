clc;clear;close all;

% load necessary dependicies
GPNeilPackage(1);

isApprox=0;
missingProb=0;
N=50;
L=1;
M=4;
Q=1;
numModel=5;

xrange=[0,2*pi];
X=sort(rand(N,1))*(xrange(2)-xrange(1))+xrange(1);
Xt=(0:0.05:2*pi)';
Xt=setdiff(Xt,X);

% the groundtruth function (instance -> groundtruth)
% g=@(x) 10*sinc(x);
g=@(x) 10*sin(6*x).*sin(x/2);
% g=@(x) sin(x);

Z=g(X);
% save('./new-exp-results/synsinc100.mat','X','Z');return;
Y=zeros(N,M);
s_g=2*rand(1,M);
% s_g(4)=0;

mycolor=distinguishable_colors(M);
mymarker={'o','s','+','x','*','d','v','^','<','>','p','h'};

% generate piecewise nonlinear function
nseeds=2;
reg_prob=1; % probabilty of a regular user;
exp_info=[];
for m=1:M
    [f,b] = genNLfunc2([min(Z),max(Z)],nseeds,reg_prob);
    exp_info=[exp_info;b];
    expert_func{m}=@(a) f(a); %w(1)*a; %abs(a);
    Y(:,m)=randnorm(1,expert_func{m}(Z),s_g(m));
end


% create missing value
Y(rand(N,M)<missingProb)=nan;
iters = 2000;

%% synthetic data
figure(1)
subplot(numModel,1,1);
% draw syn data
plot(Xt,g(Xt),'k-','linewidth',2)
hold all;
for m=1:M
    plot(X,Y(:,m),mymarker{m},'color',mycolor(m,:));
end
hold off;
axis tight;
box on;
title(['groundtruth and responses from ', num2str(M), ' observers'])

%% proposed model

subplot(numModel,1,2)
% Optimise the model.

model1=lvmobCreate(X,Y,1,[],[],[],iters,isApprox);
[pred1,var1]=lvmobPredict(model1,Xt,0);
shadedErrorBar(Xt,pred1,var1,{'k--'});
axis tight;
box on;
title('LVMOB1')

%% naive average

% subplot(numModel,1,3);
% model2 = gpavgCreate(X,Y,[],[],iters,isApprox);
% [pred2, var2]=lvmobPredict(model2,Xt,0);
% shadedErrorBar(Xt,pred2,var2,{'k--'});
% axis tight;
% box on;
% title('AVG')


subplot(numModel,1,3);
model2 = gpGtCreate(X,Y,Z,[],[],iters,isApprox);
[pred2, var2]=lvmobPredict(model2,Xt,0);
shadedErrorBar(Xt,pred2,var2,{'k--'});
axis tight;
box on;
title('AVG')

%% gplvm

subplot(numModel,1,4);
model3 = mgpCreate(X,Y,[],[],iters,isApprox);
[pred3, var3]=lvmobPredict(model3,Xt,0);
shadedErrorBar(Xt,pred3,var3,{'k--'});
axis tight; 
box on;
title('GPLVM')


%% lvmob3

subplot(numModel,1,5);
model4 = lvmob3Create(X,Y,[],[],iters,isApprox);
[pred4, var4]=lvmobPredict(model4,Xt,0);
shadedErrorBar(Xt,pred4,var4,{'k--'});
axis tight; 
box on;
title('LVMOB3')
%% draw expertise of each observer
figure(2)

% draw original expertise
for m=1:M
    subplot(3,M,m);
    Zt=[min(Z):0.1:max(Z)]';
    mean_y=expert_func{m}(Zt);
    var_y=repmat(s_g(m),[length(Zt),1]);
    shadedErrorBar(Zt,mean_y,var_y,{'color',mycolor(m,:)'});
    axis ([min(Zt), max(Zt), min(mean_y), max(mean_y)]);
    axis square;
    % 		set(gca,'XTick',[min(Zt), (max(Zt)-min(Zt))/2+min(Zt), max(Zt)],'YTick',[min(mean_y), (max(mean_y)-min(mean_y))/2+min(mean_y), max(mean_y)]);
    % 		set(gca,'XTickLabel',{'0','0.5','1'},'YTickLabel',{'0','0.5','1'});
    xlabel('Ground truth')
    ylabel(strcat('Ob.',num2str(m), ' resp.'));
end

% draw estimate expertise
for m=1:M
    subplot(3,M,M+m);
    Zt=linspace(min(pred1),max(pred1),100)';
    [mean_y,var_y]=gpOut(model1.obs{m},Zt);
    shadedErrorBar(Zt,mean_y,var_y,{'color',mycolor(m,:)'});
    axis ([min(Zt), max(Zt), min(mean_y), max(mean_y)]);
    axis square;
    % 		set(gca,'XTick',[min(Zt), (max(Zt)-min(Zt))/2+min(Zt), max(Zt)],'YTick',[min(mean_y), (max(mean_y)-min(mean_y))/2+min(mean_y), max(mean_y)]);
    % 		set(gca,'XTickLabel',{'0','0.5','1'},'YTickLabel',{'0','0.5','1'});
    xlabel('Ground truth')
    ylabel(strcat('Ob.',num2str(m), ' resp.'));
end

% draw estimate expertise
for m=1:M
    subplot(3,M,2*M+m);
    Zt=linspace(min(pred2),max(pred2),100)';
    [mean_y,var_y]=gpOut(model2.obs{m},Zt);
    shadedErrorBar(Zt,mean_y,var_y,{'color',mycolor(m,:)'});
    axis ([min(Zt), max(Zt), min(mean_y), max(mean_y)]);
    axis square;
    % 		set(gca,'XTick',[min(Zt), (max(Zt)-min(Zt))/2+min(Zt), max(Zt)],'YTick',[min(mean_y), (max(mean_y)-min(mean_y))/2+min(mean_y), max(mean_y)]);
    % 		set(gca,'XTickLabel',{'0','0.5','1'},'YTickLabel',{'0','0.5','1'});
    xlabel('Ground truth')
    ylabel(strcat('Ob.',num2str(m), ' resp.'));
end
GPNeilPackage(0);