function [X,Z,Xt,Zt]=loadExpData(dmat,trs,tss)
d = load(dmat);
data = d.data;
fullsize = size(data);
if nargout ~= 2*(nargin-1)
  fprintf('Input or output arguments not matched...!\n');
  return;
end

% ......

if nargin == 2
  % no test set
  if trs < fullsize(1)
    % data is big enough
    train_idx = randsample(fullsize(1), trs);
    Z = data(train_idx, end);     % real ground-truth
    X = data(train_idx, 1:end-1); % features
  else
    % data is not big enough, take all
    Z = data(:, end);
    X = data(:, 1:end-1);
  end
else
  if nargin == 3 && ((trs+tss) < fullsize(1))
    train_idx = randsample(fullsize(1), trs);
    test_idx = randsample( setdiff(1:fullsize(1), train_idx), tss);
    Z = data(train_idx, fullsize(2));     % real ground-truth
    X = data(train_idx, 1:fullsize(2)-1); % features
    Zt = data(test_idx, fullsize(2));
    Xt = data(test_idx, 1:fullsize(2)-1);
  else
    fprintf('data is not large enough for your choice...!\n');
    return;
  end
end


end