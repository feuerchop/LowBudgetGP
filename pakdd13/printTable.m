clc;clear;close all;

load('exp2-results/BaselineOnly-Concrete_Data.mat-N500-M20-F20-R0.7-26-Sep-2012 09:11:17.mat');

a=results.GroundtruthPCC';
a_mean=mean(a);
a_std=std(a);

o_str='';


for j=1:4
    o_str=sprintf('%s $%.2f \\pm %.2f$ &',o_str,a_mean(j),a_std(j));
end
o_str
%%
if a_mean(2)>a_mean(1)
    o_str=sprintf('$%.2f \\pm %.2f^{g}$ &', a_mean(2), a_std(2));
else
    o_str=sprintf('$%.2f \\pm %.2f^{s}$ &', a_mean(1), a_std(1));
end
if a_mean(4)>a_mean(3)
    o_str=sprintf('%s $%.2f \\pm %.2f^{g}$ &', o_str,a_mean(4), a_std(4));
else
    o_str=sprintf('%s $%.2f \\pm %.2f^{s}$ &',o_str, a_mean(3), a_std(3));
end

o_str

