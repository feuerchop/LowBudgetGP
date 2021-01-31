function y = sigmoid(x)

y = 1./(1+exp(-x));

y(y>0.5)=1;
y(y<=0.5)=0;
    

end