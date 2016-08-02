function kopt=inheritModel(model,kopt)
kopt.InitZ=model.Z;

switch kopt.TrainMethod
	case 'MKL'
		model.InstanceKernel.ReturnKernel=1;
		for d=1:size(model.Z,2)
			kopt.InstanceKernel.InitKernel{d}=calcKernel(sqdist(model.X',model.X'), model.InstanceKernel.Hyperparameters, model.X, [], model.InstanceKernel, 0);
		end
	case 'MAP'
		kopt.InitKappa=model.InstanceKernel.Hyperparameters;
end

end