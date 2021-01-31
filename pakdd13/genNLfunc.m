function f = genNLfunc(xrange, linearty)
N=100;
x_train=linspace(xrange(1),xrange(2),N)';
K=calcNN(x_train,x_train)+x_train*x_train';

while 1
	y_train=mvnrnd(zeros(1,N),K,1)';
	y_train(1) = x_train(1);
	y_train(end) = x_train(end);
	if abs(linearty-corr(x_train, y_train))<0.05
		break
	end
end

f=@(x) GPnonlinear(x, x_train, y_train);

end

function y = GPnonlinear(x_test,x_train,y_train)

K = calcNN(x_train, x_train);
K1= calcNN(x_train, x_test);

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