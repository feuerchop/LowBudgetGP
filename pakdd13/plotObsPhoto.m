clc;clear;close all;

load('real-results/RankedObrFotos-5TopBottom.mat')


topn=20;
actn=topn;
img_h=1/(2*topn);
mar1=0;


for j=1:10
    img_w1=1/(2*topn);
    for k=1:topn
        subplot('position',[(k-1)*img_w1+mar1,(1-mar1-img_h)-(j-1)*img_h,img_w1,img_h]);
        imshow(imresize(rankedFotoArray{k,j},[64,NaN]));
    end
    img_w2=1/(4*topn);
    for k=(2*actn-topn+1):(2*actn)
        subplot('position',[rem(k-1,topn)*img_w2+1.5*mar1+topn*img_w1,(1-mar1-img_h)-(j-1)*img_h,img_w2,img_h]);
        imshow(imresize(rankedFotoArray{k,j},[64,NaN]));
    end
end
text(0,0,'Top-5')
text(0,0,'Bottom-5')

