clc;clear;close all;

%load('all_photos.mat');
load('real-results/RankedObrFotos-20TopBottom.mat');

topn=20;
actn=topn;
img_h=1/(0.8*topn);
%mname={'GPR-AVG','Raykar','LOB','NLOB'};
mar1=0.03;


for j=1:10
    img_w1=1/(2*topn);
    for k=1:topn
        subplot('position',[(k-1)*img_w1+mar1,(1-mar1-img_h)-(j-1)*img_h,img_w1,img_h]);
        idx=(j-1)*actn*2+k;
        imshow(imresize(rankedFotoArray{idx},[64,NaN]));
    end
    img_w2=1/(4*topn);
    for k=(2*actn-topn+1):(2*actn)
        subplot('position',[rem(k-1,topn)*img_w2+1.5*mar1+topn*img_w1,(1-mar1-img_h)-(j-1)*img_h,img_w2,img_h]);
        idx=(j-1)*actn*2+k;
        imshow(imresize(rankedFotoArray{idx},[64,NaN]));
    end
end
for j=1:4
text(0,0,mname{j},'rotation',90)
end
text(0,0,'Top-5')
text(0,0,'Bottom-5')

