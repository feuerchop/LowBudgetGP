function z=calcNMAE(x,y)
x=(x-min(x))/(max(x)-min(x));
y=(y-min(y))/(max(y)-min(y));
z=mean(abs(x-y));