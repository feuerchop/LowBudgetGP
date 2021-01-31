clc;clear;close all;

load('real-results/fotorating-s201-obs10withIDs-10cv-201trainset.mat');
nObr = length(names);
numFoto = 20;
rankedFotoArray = {};
baseurl = 'http://gallery.photo.net/photo/';
endurl = '-md.jpg';
pid=1;
for m = 1:nObr
	[top, bottom] = getObrFotoRank(m, numFoto);
	for f = 1:size(top,1)
		[img,fst] = urlwrite([baseurl, int2str(top(f, 1)), endurl], 'temp.jpg');
		if fst
			rankedFotoArray{pid} = imread('temp.jpg');
		else
			fprintf('broke image')
			rankedFotoArray{pid} =zeros(10);
		end
		fprintf('.%d',pid);pid=pid+1;
	end
	for f = 1:size(bottom,1)
		[img,fst] = urlwrite([baseurl, int2str(bottom(f, 1)), endurl], 'temp.jpg');
		if fst
			rankedFotoArray{pid} = imread('temp.jpg');
		else
			fprintf('broke image')
			rankedFotoArray{pid} =zeros(10);
		end
		fprintf('.%d',pid);pid=pid+1;
	end
end
save(['real-results/RankedObrFotos-', int2str(numFoto), 'TopBottom.mat'], 'rankedFotoArray');
delete('temp.jpg');