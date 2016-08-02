while 1
	clc;clear;close all;
	% also need libsvm for SVR
	rpath='results-syn/';
	addpath('~/Documents/lightspeed/');
	addpath('~/Documents/lbfgs/');
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
	
	figure(1)
	mname{1}='Ground truth';
	perfarray=zeros(4,1);
	subplot(2,4,[1,2,5,6]);
	% draw syn data
	plot(Xt,normaltoUnit(g(Xt)),'k-')
	hold all;
	for m=1:M
		Y(:,1,m)=randnorm(1,expert_func{m}(Z),s_g(m));
		%shadedErrorBar(x',(1+w(m))*y',sqrt(s_g(m))*ones(1,N),{'-','color',mycolor(m,:)},1);
		plot(X,normaltoUnit(Y(:,1,m)),mymarker{m},'color',mycolor(m,:));
	end
	legend('Ground truth','Ob.1','Ob.2','Ob.3','Ob.4')
	set(gca,'YTick',[0, 0.5, 1]);
	hold off;
	axis tight;
	
	
	% draw response curve of each expert
	for m=1:M
		subplot(2,M,2*m+mod(m,2))
		Zt=[min(Z):0.1:max(Z)]';
		mean_y=expert_func{m}(Zt);
		var_y=repmat(s_g(m),[length(Zt),1]);
		shadedErrorBar(Zt,mean_y,var_y,{'color',mycolor(m,:)'});
		axis ([min(Zt), max(Zt), min(mean_y), max(mean_y)]);
		axis square;
		set(gca,'XTick',[min(Zt), (max(Zt)-min(Zt))/2+min(Zt), max(Zt)],'YTick',[min(mean_y), (max(mean_y)-min(mean_y))/2+min(mean_y), max(mean_y)]);
		set(gca,'XTickLabel',{'0','0.5','1'},'YTickLabel',{'0','0.5','1'});
		xlabel('Ground truth')
		ylabel(strcat('Ob.',num2str(m), ' resp.'));
	end
	%set(gca,'XTick',[],'YTick',[]);
	%title('groundtruth and observations from four experts')
    return
	%% SVR baseline
	figure(2)
	
% 	subplot(1,3,1)
% 	% train SVR using avg as baseline
% 	model0=svmtrain(mean(Y,3),X,'-s 3');
% 	mean_y=svmpredict(g(Xt),Xt,model0);
% 	%plot(X,normaltoUnit(mean(Y,3)),'m.');hold on;
% 	plot(Xt,normaltoUnit(mean_y),'k-');
% 	% 	hold on;plot(X,normaltoUnit(mean(Y,3)),'k*');hold off;
% 	axis tight;
% 	set(gca,'YTick',[0, 0.5, 1]);
% 	% set(gca,'XTick',[],'YTick',[]);
% 	mname{2}=sprintf('MANE:%.2f, PCC:%.2f',calcNMAE(mean_y,g(Xt)), corr(mean_y,g(Xt)));
% 	perfarray(1)=calcNMAE(mean_y,g(Xt));
% 	title(mname{2});
	
	% GP baseline
	subplot(1,3,2)
	[mean_y,var_y]=gpbaseline(X,Z,Xt);
	shadedErrorBar(Xt',mean_y,var_y,{'k-'});
	% 	hold on;plot(X,normaltoUnit(mean(Y,3)),'k*');hold off;
	% plot(Xt,normaltoUnit(mean_y),'k--','LineWidth',2);
	axis tight;
	set(gca,'YTick',[min(mean_y), (max(mean_y)-min(mean_y))/2+min(mean_y), max(mean_y)]);
	set(gca,'YTickLabel',{'0','0.5','1'});
	axis([min(Xt),max(Xt),min(mean_y),max(mean_y)]);
	box on;
	mname{3}=sprintf('MANE:%.2f, PCC:%.2f',calcNMAE(mean_y,g(Xt)), corr(mean_y,g(Xt)));
	perfarray(2)=calcNMAE(mean_y,g(Xt));
	title(mname{3});
	
	%% Linear Observer Model

	% set kernel options
	kopt=koptDefault(Y); %MUST use koptDefault to initialize the option!
	kopt.TrainMethod='MLE'; % can be 'MLE' or 'MAP'
	kopt.Regularization.Mode='Quad'; % can be 'L1' or 'L2', only for 'MLE'
	kopt.UserConfidence=exp_info;%ones(1,M); % higher means more reliable
	kopt.eta=1;
	% train model
	model1=trainCrowdGPLVM(X,Y,kopt);
	
	% predict on Xt, 0 for prediting the groundtruth, 1...M for predicting corresponding users
	[mean_y,var_y]=predictCrowdGPLVM(Xt,model1,0);
	
	% draw plot
	subplot(1,3,3)
	%plot(Xt,normaltoUnit(mean_y),'k-.','LineWidth',2);
	shadedErrorBar(Xt',mean_y,var_y,{'k-'});
	axis tight;
	set(gca,'YTick',[min(mean_y), (max(mean_y)-min(mean_y))/2+min(mean_y), max(mean_y)]);
	set(gca,'YTickLabel',{'0','0.5','1'});
	axis([min(Xt),max(Xt),min(mean_y),max(mean_y)]);
	box on;
	mname{4}=sprintf('MANE:%.2f, PCC:%.2f',calcNMAE(mean_y,g(Xt)), corr(mean_y,g(Xt)));
    perfarray(3)=calcNMAE(mean_y,g(Xt));
	title(mname{4});
	% plotExpertPrediction(g, expert_func, s_g, Xt, model1, mycolor,model_name);
% 	perfarray(3)=calcNMAE(mean_y,g(Xt));
% 	for m=1:M
% 		subplot(2,M,2*m+mod(m,2))
% 		[mean_y,var_y, Zt]=predictCrowdGPLVM(Xt, model1, m);
% 		shadedErrorBar(normaltoUnit(Zt),normaltoUnit(mean_y),var_y,{'color',mycolor(m,:)'});
% 		axis ([0,1,0,1]);
% 		%axis ([min(Zt), max(Zt), min(mean_y), max(mean_y)]);
% 		axis square;
% % 		set(gca,'XTick',[min(Zt), (max(Zt)-min(Zt))/2+min(Zt), max(Zt)],'YTick',[min(mean_y), (max(mean_y)-min(mean_y))/2+min(mean_y), max(mean_y)]);
% % 		set(gca,'XTickLabel',{'0','0.5','1'},'YTickLabel',{'0','0.5','1'});
% 		xlabel('Ground truth')
% 		ylabel(strcat('Ob.',num2str(m), ' resp.'));
% 	end
	
	
	%% Nonlinear Observer Model
	figure(3);
	% set kernel options
	kopt=koptDefault(Y);
	kopt.TrainMethod='MAP';
	
	% initialize the new model with previous model
	kopt=inheritModel(model1,kopt);
	
	% train model
	model2=trainCrowdGPLVM(X,Y,kopt);
	
	% predict on Xt, 0 for prediting the groundtruth, 1...M for predicting corresponding users
	[mean_y,var_y]=predictCrowdGPLVM(Xt,model2,0);
	
	% draw plot
	subplot(2,4,[1,2,5,6]);
	%plot(Xt,normaltoUnit(mean_y),'k-.','LineWidth',2);
	shadedErrorBar(Xt',mean_y,var_y,{'k-'});
	axis tight;
	set(gca,'YTick',[min(mean_y), (max(mean_y)-min(mean_y))/2+min(mean_y), max(mean_y)]);
	set(gca,'YTickLabel',{'0','0.5','1'});
	box on;
	mname{5}=sprintf('MANE:%.2f, PCC:%.2f',calcNMAE(mean_y,g(Xt)), corr(mean_y,g(Xt)));
	axis([min(Xt),max(Xt),min(mean_y),max(mean_y)]);
	% title(sprintf('%s: %.3f',model_name, calcNMAE(mean_y,g(Xt))));
	%plotExpertPrediction(g, expert_func, s_g, Xt, model2, mycolor,model_name);
	perfarray(4)=calcNMAE(mean_y,g(Xt));
	title(mname{5});
	for m=1:M
		subplot(2,M,2*m+mod(m,2))
		[mean_y,var_y, Zt]=predictCrowdGPLVM(Xt, model2, m);
		shadedErrorBar(Zt,mean_y,var_y,{'color',mycolor(m,:)'});
		axis ([min(Zt), max(Zt), min(mean_y), max(mean_y)]);
		axis square;
		set(gca,'XTick',[min(Zt), (max(Zt)-min(Zt))/2+min(Zt), max(Zt)],'YTick',[min(mean_y), (max(mean_y)-min(mean_y))/2+min(mean_y), max(mean_y)]);
		set(gca,'XTickLabel',{'0','0.5','1'},'YTickLabel',{'0','0.5','1'});
		xlabel('Ground truth')
		ylabel(strcat('Ob.',num2str(m), ' resp.'));
	end
	
	
	if perfarray==sort(perfarray,'descend')
		for j=1:3
			figure(j);
			saveas(gcf, sprintf('%s%s-%s.fig',rpath,mname{j},datestr(now)), 'fig');
		end
		break;
	end
end


return;
%% Multiple Kernel Learning
% set kernel options
kopt=koptDefault(Y);
kopt.TrainMethod='MKL';

% must >=0, controls the number of kernels
kopt.InstanceKernel.Width=10;
kopt.UserKernel.Width=10;

% higher value indicates a more sparse solution.
kopt.InstanceKernel.RegC=10;
kopt.UserKernel.RegC=10;

% initialize the new model with previous model
kopt=inheritModel(model1,kopt);

% train model
model3=trainCrowdGPLVM(X,Y,kopt);

% predict on Xt, 0 for prediting the groundtruth, 1...M for predicting corresponding users
[mean_y,var_y]=predictCrowdGPLVM(Xt,model3,0);

% draw plot
figure(1);subplot(5,1,5);
shadedErrorBar(Xt',normaltoUnit(mean_y),normaltoUnit(var_y),{'k--'});
axis tight
set(gca,'XTick',[],'YTick',[]);
model_name='MKL-GP (LinearGP ini.)';
title(sprintf('%s: %.3f',model_name, calcNMAE(mean_y,g(Xt))));
plotExpertPrediction(g, expert_func, s_g, Xt, model3, mycolor,model_name);


%%
figure;

for m=1:M
	subplot(4,M,3*M+m)
	[mean_y,var_y, Zt]=predictCrowdGPLVM(Xt, model3, m);
	shadedErrorBar(Zt,mean_y,var_y,{'color',mycolor(m,:)'});
	axis ([min(Zt), max(Zt), min(mean_y), max(mean_y)]);
	axis square;
	set(gca,'XTick',[],'YTick',[]);
	xlabel('groundtruth')
	ylabel(strcat(num2str(m), '-response'));
end


