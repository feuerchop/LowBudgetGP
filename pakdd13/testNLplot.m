clc;clear;close all;

xrange=[0,10];
x=linspace(xrange(1),xrange(2),1000)';
N=10;
mycolor=hsv(N);
lvp=linspace(0.3,1,N); %linearity <0.3 is really hard to generated!!!

lvp_text=cell(10,1);

hold all;
for n=1:N
    req_lv=lvp(n);
    [f,b,lv]=genNLfunc2(xrange,2,0.5,req_lv);
    plot(x,f(x),'color',mycolor(n,:));
    lvp_text{n}=sprintf('linearity:%.2f',lv);
end
legend(lvp_text);
hold off;
axis tight;
return

% 
% for d=1:N;
% 	f=genNLfunc2(xrange,2,1);
% 	funcSet{d}=f;
% 	funcCorr(d)=corr(x,f(x));
% end
% 
% 
% [~,b]=sort(funcCorr,'ascend');
% 
% hold all;
% for d=1:N
% 	plot(x,funcSet{b(d)}(x),'color',mycolor(d,:));
% end
% hold off;
% title(['linearity (corr):' num2str(min(funcCorr)) '(blue) to ' num2str(max(funcCorr)) '(green)'])
% axis tight;
% 
