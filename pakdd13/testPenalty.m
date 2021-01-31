clc;clear;close all;

y1=[];y2=[];y3=[];y4=[];y5=[];
xd=-3:.1:3;

fl=@(x,y) .5*log(y)+sqrt(2/y)*abs(x-1);
fg=@(x) (x-1)^2;
colors=distinguishable_colors(5);

hold all;
for x=1:length(xd)
	f1=penaltyFunc(xd(x),0.5);
	f2=penaltyFunc(xd(x),2);
	f3=fl(xd(x),1);
	f4=fl(xd(x),2);
	f5=fg(xd(x));
	y1=[y1;f1];
	y2=[y2;f2];
	y3=[y3;f3];
	y4=[y4;f4];
	y5=[y5;f5];
end

plot(xd,y1,'-','color',colors(1,:),'LineWidth',1)
plot(xd,y2,'-','color',colors(2,:),'LineWidth',1)
plot(xd,y3,'--','color',colors(3,:),'LineWidth',1)
plot(xd,y4,'--','color',colors(4,:),'LineWidth',1)
plot(xd,y5,'-.','color',colors(5,:),'LineWidth',1)
legend('General \eta=0.5', 'General \eta=2', 'Laplace \lambda=1', 'Laplace \lambda=2', 'Gaussian');
box on;
axis([-3,3,0,8])