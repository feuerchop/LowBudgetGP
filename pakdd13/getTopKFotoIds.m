function ids = getTopKFotoIds( k )
%GETTOPKFOTOIDS Summary of this function goes here
%   Detailed explanation goes here
   load('real-results/foto-ranking-4demo.mat');
   ids = Fids(idx(1:k));
end

