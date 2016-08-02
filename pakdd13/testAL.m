clc;clear;close all;
N=100;
xrange=[0,2*pi];
X=sort(rand(N,1))*(xrange(2)-xrange(1))+xrange(1);
% the groundtruth function (instance -> groundtruth)
% g=@(x) 10*sinc(x);
g=@(x) 10*sin(6*x).*sin(x/2);
% g=@(x) sin(x);
Y=g(X);

selK=1;
X_sel=[];
Y_sel=[];
% random select the first instance
ind=randsample(1:N,1);

X_sel=[X_sel;ind];
Y_sel=[Y_sel,Y(ind)];

for t=1:floor(N/selK)
    [ind,sel_var,all_score,all_var,f_gp] = gpactivelearning(X,X_sel,Y_sel,selK);
    shadedErrorBar(X,all_score,all_var,{'k-'});    hold all;
    plot(X,Y,'k.');
    for p=1:length(X_sel)
        plot(X(X_sel(p)),Y_sel(p),'*r','MarkerSize',10);
    end
    plot(X(ind),Y(ind),'ob','MarkerSize',10);
    X_sel=[X_sel;ind];
    Y_sel=[Y_sel;Y(ind)];
    hold off;
    axis tight;
    axis([min(X),max(X),min(Y),max(Y)]);
    title(['active learning : selected ', num2str(t*selK+1)]);
    pause;
end



