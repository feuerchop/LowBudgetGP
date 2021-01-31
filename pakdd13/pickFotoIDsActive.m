function [sel_foto_id, top_foto_id, all_score] = pickFotoIDsActive( user, topk )
%PICKFOTOIDSACTIVE Summary of this function goes here
% load Foto features mat
features = load('real-test/datasets/photonet_features.mat');
X_all = features.data;
X_all(:,[1,3,4]) = [];
Fids = X_all(:,1);
X = X_all(:,2:end);
% connect database
dbname = 'photonetstayer';
uname = 'root';
pass = 'damnshit';
driver = 'com.mysql.jdbc.Driver';
server = 'jdbc:mysql://localhost:3306/photonetstayer';
conn = database(dbname, uname, pass, driver, server);
% query data from db
query = ['select pic_id, uname, rating from picstay where uname=', '''', user, '''', ';'];
curs = exec(conn, query);
d = fetch(curs);
rated_cols = d.data;
if size(rated_cols,2) ~= 1
   rated_id = rated_cols(:,1);
   rated_id = cellfun(@str2num, rated_id);
   rated_score = cell2mat(rated_cols(:,3));   
   [~, ia, ib] = intersect(Fids, rated_id);
   % ia is the rated foto idx
   rated_score_sorted = rated_score(ib);
   [sel_id,sel_var,all_score,all_var,f_gp] = gpactivelearning(X,ia,rated_score_sorted,topk);
   sel_foto_id = Fids(sel_id);
   [all_score_sorted, idx] = sort(all_score, 'descend');
   top_foto_id = Fids(idx(1:topk));
else
   % if user has not rated any foto, then randomly select top k
   % give all zero on all the fotos
   sel_foto_id = randsample(length(Fids), topk);
   top_foto_id = randsample(length(Fids), topk);
   all_score = zeros(length(Fids,1));
end
end

