function f = getPLfunc(xrange, yrange, nsteps)

x0=rand(1,nsteps)*(xrange(2)-xrange(1))+xrange(1);
x0=sort(x0);
x0(1)=xrange(1);
x0(end)=xrange(2);

y0=rand(1,nsteps)*(yrange(2)-yrange(1))+yrange(1);
% monotone?
% y0=sort(y0);

% reverse response?
a=(rand>0.5);
y0(1)=yrange(1)*a+yrange(2)*(1-a);
y0(end)=yrange(2)*a+yrange(1)*(1-a);


f= @(x) twoPointsLinear(x,x0,y0);
end

function y = twoPointsLinear(x,x0,y0)
y=zeros(size(x));
for n=1:length(x)
	for ind=1:(length(x0)-1)
		if x(n)>=x0(ind) && x(n)<=x0(ind+1)
			x1=x0(ind);
			x2=x0(ind+1);
			y1=y0(ind);
			y2=y0(ind+1);
			y(n)=(y2-y1)/(x2-x1)*(x(n)-x1)+y1;
			break
		end
	end
end
end