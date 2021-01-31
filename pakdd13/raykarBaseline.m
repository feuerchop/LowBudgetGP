function [w,prec]=raykarBaseline(X,Y,maxIter)
% X=rand(10,3);
% Y=rand(10,2);
% maxIter=1000;
D=size(X,2);
N=size(X,1);

if length(size(Y))==3
    M=size(Y,3);
    Y=reshape(Y,N,M);
end

prec=rand(1,M);

w=100*(rand(1,D)-1);

% for m=1:M
%     prec(m)=0;
%     for n=1:N
%         prec(m)=prec(m)+(Y(n,m)-w*X(n,:)')^2;
%     end
%     prec(m)=prec(m)/N;
% end

stopCr=1e-6;

oldw=0;
oldp=0;

for iter=1:maxIter
    
    for m=1:M
        prec(m)=mean((Y(:,m)'-w*X').^2);
    end
    
    lp=jitChol(X'*X)';
    p2=X'*sum(repmat(1./prec,N,1).*Y,2)./sum(1./prec);
%     for n=1:N
%         p2=p2+X(n,:)'*(sum(Y(n,:)./prec)/sum(1./prec));
%     end
    w=(lp'\(lp\p2))';
    
    delta_w=sum(abs(w-oldw));
    delta_p=sum(abs(prec-oldp));
    
    oldp=prec;
    oldw=w;
    
%     nl=0;
%     for n=1:N
%         for m=1:M
%             params=[w*X(n,:)',1/prec(m)];
%             nl=nl+normlike(params,Y(n,m));
%         end
%     end
%     fprintf('nl:%.3f\n',nl);
    
    
    if delta_w<stopCr && delta_p<stopCr
        fprintf('converged after %d iters\n',iter);
        break;
    end
    
    if mode(iter,100)==0
        fprintf('iter:%d\t w:%.3f\t prec:%.3f\n',iter,delta_w,delta_p);
    end
    
    
end
end

