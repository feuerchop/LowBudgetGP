function plotExpertPrediction(g, expert_func, s_g, Xt, model1, mycolor, suptitle)
figure;
M=length(s_g);
for m=1:M
    subplot(M,1,m)
    hold all;
    tY=expert_func{m}(g(Xt));
    shadedErrorBar(Xt',normaltoUnit(tY),sqrt(s_g(m))*ones(size(Xt')),{'k--'},1);
    [pY,vY]=predictCrowdGPLVM(Xt,model1,m);
    shadedErrorBar(Xt',normaltoUnit(pY),normaltoUnit(vY),{'-','color',mycolor(m,:)},1);
    %plot(Xt,pY,'-','color',mycolor(m,:),'linewidth',1);
    hold off
    axis tight
    title_txt=sprintf('predict (color) vs actural (gray) response of [%d] expert: %.3f', m, calcNMAE(tY,pY));
    title(title_txt)
    set(gca,'XTick',[],'YTick',[]);
    box on;
end
suplabel(suptitle,'t');
end

function z=calcRMSE(x,y)
z=sqrt(mean((x-y).^2));
end

function z=calcMAE(x,y)
z=mean(abs(x-y));
end