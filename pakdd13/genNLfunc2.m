%% Generate a nonlinear function
% [f, b, lv] = genNLfunc2(xrange, nseeds, prob, ndegree)
% 
% Input:
% xrange      the range of response, e.g. [0,10]
% nseeds      the number of random seeds, more points -> more linear!!!
%             suggested 2~5
% prob        the probabilty that generated function has postive slope
%             e.g.
%             y=2x+1  has postive slope corresp. to regular observer
%             y=-2x+1 has negative slope corresp. to adversarial observer
%             default, prob:=0.5
%             set prob:=1 if you only want to generate regular observer
% ndegree     linearity (absoulte), 0~1
%             !!!NOTE!!!
%             ndegree<0.3 may takes a long time (may even be infeasible) 
%             to generate.
% Output:
% f           generated function
% b           +1 (regular) or -1 (adversarial)
% lv          generated linearity, small value means nonlinear

function [f, b, nl] = genNLfunc2(xrange, nseeds, prob, ndegree)



myeps=1e-2;

if nargin<3 || isempty(prob)
    prob=0.5;
end

[x_train, y_train,b]=genPoints(xrange, nseeds, prob);
f=@(x) pchip(x_train, y_train, x);

if nargin==4
    x_test=linspace(xrange(1),xrange(2),1000)';
    ntries=0;
    while 1
        nl=corr(x_test,f(x_test));
        if abs(abs(nl)-abs(ndegree))<myeps
            break;
        else
            [x_train, y_train,b]=genPoints(xrange, nseeds, prob);
            f=@(x) pchip(x_train, y_train, x);
            ntries=ntries+1;
        end
        if ntries==2000
            disp(['After ', num2str(ntries), ' tries,'...
                ' we can''t generate a function with linearity of ', ...
                num2str(ndegree), ' under precision ',num2str(myeps)]);
            myeps=1.5*myeps;
            disp(['Relax the precision to ',num2str(myeps)]);
            ntries=0;
        end
    end
    
end
end

function [x_train, y_train,b]=genPoints(xrange, nseeds, prob)
rpts=rand(nseeds,2)*(xrange(2)-xrange(1))+xrange(1);
x_train=rpts(:,1);
y_train=rpts(:,2);
x_train=[x_train;xrange(1)];
y_train=[y_train;xrange(1)];
x_train=[x_train;xrange(2)];
y_train=[y_train;xrange(2)];
x_train=sort(x_train);

if (rand<prob)
    % regular
    y_train=sort(y_train,'ascend');
    b=1;
else
    % malicious
    y_train=sort(y_train,'descend');
    b=-1;
end
end