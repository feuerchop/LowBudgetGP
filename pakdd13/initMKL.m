function kopt=initMKL(model,kopt)
kopt.InitZ=model.Z;

model.InstanceKernel.ReturnKernel=1;
for d=1:size(model.Z,2)
kopt.InstanceKernel.InitKernel{d}=calcKernel(sqdist(model.X',model.X'), model.InstanceKernel.Hyperparameters, model.X, [], model.InstanceKernel, 0);
end

end