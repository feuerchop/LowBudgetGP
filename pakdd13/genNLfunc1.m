function f = genNLfunc1(xrange, nseeds)
rpts=rand(nseeds,2)*(xrange(2)-xrange(1))+xrange(1);
x_train=rpts(:,1);
y_train=rpts(:,2);
x_train=[x_train;xrange(1)];
y_train=[y_train;xrange(1)];
x_train=[x_train;xrange(2)];
y_train=[y_train;xrange(2)];
x_train=sort(x_train);
y_train=sort(y_train);
f=@(x) pchip(x_train, y_train, x);

end

function y = GPnonlinear(x_test,x_train,y_train)

K = calcNN(x_train, x_train)+x_train*x_train';
K1= calcNN(x_train, x_test)+x_train*x_test';

KL=jitChol(K)';
alpha = KL'\(KL\y_train);
y= K1'*alpha;

end

function K=calcNN(x,x2)
  innerProd = x*x2';  
  numer = innerProd;
  vec1 = sum(x.*x, 2) + 1;
  vec2 = sum(x2.*x2, 2) + 1;
  denom = sqrt(vec1*vec2');
  arg = numer./denom;
  K = 2/pi*asin(arg);
end