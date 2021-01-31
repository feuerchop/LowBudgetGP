function [f,g]=wrapperCrowdGPLVM(x,auxdata)

f = objCrowdGPLVM(x, auxdata);
g = gradCrowdGPLVM(x, auxdata);

end
