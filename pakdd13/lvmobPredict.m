function [pred, var] = lvmobPredict(model,Xt,m)

% do prediction for the ground truth
[pred, var]=gpOut(model.gt,Xt);

% do prediction for observer
if m>0
    [pred, var]=gpOut(model.obs{m},pred);
end
end