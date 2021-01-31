clc;clear;close all;

figure;
load('real-results/fotorating-s201-obs10withIDs-10cv-201trainset.mat');
load('real-test/datasets/fotoratings-unrated.mat');
load('real-results/fotorating-s201-obs10withIDs-RatingsOnRest2733-C1e4.mat');
[~, bestNLOBId] = max(foto_results.Ratios(4, :));
bestNLOBModel = foto_results.Models{4}{bestNLOBId};
nObr = length(names);
nIns = size(XRest,1);
gndRatings = 5*dataScale(ratings,1);
obrRatings = zeros(nIns, nObr);
A=XRest;
for m = 1:nObr
	obrRatings(:,m) = predictCrowdGPLVM(A(:,3:end), bestNLOBModel, m);
end
obrRatings = 5*dataScale(obrRatings, 2);
%%
for m = 1:nObr
	x=gndRatings(4,:)';
	y=obrRatings(:,m);
	ax = subplot(2,nObr/2,m);
	scatplot(x,y);
	hold on;
	plot([0;5],[0;5],'k-');
	hold off;
	axis([0,5,0,5]);axis square;
	box on;
	set(gca,'YTick',[0, 5]);
	set(gca,'XTick',[0, 5]);
	ttext=sprintf('PCC: %.2f, \\rho: %.2f',corr(x,y),corr(x,y,'type','Spearman'));
	title(ttext)
	ttext=sprintf('Obs. %d', m);
	ylabel(ttext);
end


