function  fetchAvgFotoScores()
%FETCHAVGFOTOSCORES Summary of this function goes here
%   Detailed explanation goes here
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
query_users = 'SELECT uname from picstay group by uname having count(pic_id) > 150;';
curs = exec(conn, query_users);
users = fetch(curs);
users = users.data;
user_score = zeros(length(Fids),1);
for u = 1:length(users)
   [sel_foto_id, top_foto_id, score] = pickFotoIDsActive( users{u}, 1 );
   score = 5*datascale(score, 1);
   user_score = user_score + score;
end
avg_score = user_score/u;
[all_score, idx] = sort(avg_score, 'descend');
save('real-results/foto-ranking-4demo.mat', 'idx', 'Fids');
end

