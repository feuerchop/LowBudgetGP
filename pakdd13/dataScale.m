function X = dataScale( X, dim )
% scale data X along dim to [0,1]
len = size(X, dim);
for n = 1:len
  if dim == 1
    % rowwise
    vec = X(n, :);
    X(n, :) = (vec - min(vec))/(max(vec) - min(vec));
  else
    if dim == 2
    % colwise
      vec = X(:, n);
      X(:, n) = (vec - min(vec))/(max(vec) - min(vec));
    end
  end
end
