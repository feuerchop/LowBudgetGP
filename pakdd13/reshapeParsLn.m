function [theta kappa] = reshapeParsLn(v, M, D, K)
sI=1;
eI=D*3*M;
theta=reshape(v(sI:eI),D,3,M);
sI=eI+1;
eI=eI+D*K;
kappa=reshape(v(sI:eI),D,K);
% eta=v(end);
end