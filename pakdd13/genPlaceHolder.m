clc;clear;close all;
hold all;
for j=1:7
    plot(10:10:100, sort(rand(10,1)));
end
hold off;
axis on;
axis tight;
axis square;
title('PLACE HOLDER: MANE on XXX data sets')
legend('SVR','GPR','SVR-AVG','GPR-AVG','GG','LOB','NLOB')