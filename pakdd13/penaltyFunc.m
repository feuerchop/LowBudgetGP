function [f,g]= penaltyFunc(x, factor)
g=zeros(size(x));f=0;

for p=1:length(x)
	[fx,gx]=penaltyEach(x(p),factor);
	f=f+fx;
	g(p)=gx;
end

end

function [f,g]=penaltyEach(x, factor)
f=0;g=0;
if x>1
	f=factor*(x-1)^2;
	g=2*factor*(x-1);
elseif x<-1
	f=factor*(x+1)^2;
	g=2*factor*(x+1);
end
end