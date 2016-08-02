function [Z theta kappa] = reshapePars(v, N, M, D, T, K)
sI=1;
eI=N*D;
Z=reshape(v(sI:eI),N,D);
sI=eI+1;
eI=eI+D*T*M;
theta=reshape(v(sI:eI),D,T,M);
sI=eI+1;
eI=eI+D*K;
kappa=reshape(v(sI:eI),D,K);
end