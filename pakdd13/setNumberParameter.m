function UserKernel=setNumberParameter(X, UserKernel)

if length(UserKernel.AddTerms)<3 || strcmp(UserKernel.AddTerms,'000')
	error('incorrect kernel function!')
end

UserKernel.NumPars=1;

if UserKernel.AddTerms(1)=='1'
	UserKernel.NumPars=UserKernel.NumPars+2; %SE par
	disp('+ squared exponential function')
end
if UserKernel.AddTerms(end)=='1'
	UserKernel.NumPars=UserKernel.NumPars+1; %linear par
	disp('+ linear function')
end
if UserKernel.AddTerms(2)=='1'
	UserKernel.NumPars=UserKernel.NumPars+1; %bias par
	disp('+ constant (bias)')
end

if isfield(UserKernel,'ARD') && UserKernel.ARD
	% turn on auto-relevance determiant kernel
	UserKernel.ARDIndex = UserKernel.NumPars +1;
	UserKernel.NumPars = UserKernel.NumPars+size(X,2)+1;
	UserKernel.ARDSource = X;
	for d=1:size(X,2)
		UserKernel.ARDSqdist{d}=sqdist(X(:,d)',X(:,d)');
	end
	disp('+ auto relvance determinant')
end

fprintf('# hyperparameters: %d\n', UserKernel.NumPars);


end