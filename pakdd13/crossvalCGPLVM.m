function allresult=crossvalCGPLVM( X, Y, Z, nbFold, errFunc, fname)

[N,~,M] = size(Y);
if nbFold>N
    error('data set is too small for cross-validation!')
end
cvp = cvpartition(N, 'kfold', nbFold);
allresult=[];
save(fname,'allresult');

% only testing this kernels, otherwise put all 7 combinations in it
AllowedKernels={'111','110','011','101'}; % ARD will be always tested
% now instead of testing 7x7 kernels, we only test 3x3 kernels.

%% for MAP
kopt=koptDefault(Y(cvp.training(1), :, :));
kopt.TrainMethod='MAP';
for p1=1:length(AllowedKernels) % for all user kernels
    kopt.UserKernel.AddTerms=AllowedKernels{p1};
    for p2=1:length(AllowedKernels) % for all instance kernels
        kopt.InstanceKernel.AddTerms=AllowedKernels{p2};
        for p3=0:1 % turn on ARD for user kernels
            kopt.UserKernel.ARD=p3;
            results_y = zeros(nbFold,M);
            results_z = zeros(nbFold,1);
            parfor r = 1:nbFold
                % for each folder
                XTrain = X(cvp.training(r), :);
                YTrain = Y(cvp.training(r), :, :);
                XTest = X(cvp.test(r), :);
                YTest = Y(cvp.test(r), :, :);
                ZTest = Z(cvp.test(r), :);
                [results_y(r, :),results_z(r)] = cvwrapper(XTrain, YTrain, XTest, YTest, ZTest, errFunc, kopt);
            end
            kopt.UserErr=mean(results_y);
            kopt.LatentErr=mean(results_z);
            allresult=[allresult; kopt];
            save(fname,'allresult','-append');
        end
    end
end

%% for MLE
kopt=koptDefault(Y(cvp.training(1), :, :));
kopt.TrainMethod='MLE';
for p1=1:2 %try different regularization
    kopt.Regularization.Mode=strcat('L',num2str(p1));
    for p2=1:length(AllowedKernels) % for all instance kernels
        kopt.InstanceKernel.AddTerms=AllowedKernels{p2};
        results_y = zeros(nbFold,M);
        results_z = zeros(nbFold,1);
        parfor r = 1:nbFold
            % for each folder
            XTrain = X(cvp.training(r), :);
            YTrain = Y(cvp.training(r), :, :);
            XTest = X(cvp.test(r), :);
            YTest = Y(cvp.test(r), :, :);
            ZTest = Z(cvp.test(r), :);
            [results_y(r, :),results_z(r)] = cvwrapper(XTrain, YTrain, XTest, YTest, ZTest, errFunc, kopt);
        end
        kopt.UserErr=mean(results_y);
        kopt.LatentErr=mean(results_z);
        allresult=[allresult; kopt];
        save(fname,'allresult','-append');
    end
end


%% for MAP
kopt=koptDefault(Y(cvp.training(1), :, :));
kopt.TrainMethod='MAP';
for p1=1:length(AllowedKernels) % for all user kernels
    kopt.UserKernel.AddTerms=AllowedKernels{p1};
    for p2=1:length(AllowedKernels) % for all instance kernels
        kopt.InstanceKernel.AddTerms=AllowedKernels{p2};
        for p3=0:1 % turn on ARD for user kernels
            kopt.UserKernel.ARD=p3;
            
            % for MLE-MAP
            kopt1=koptDefault(Y(cvp.training(1), :, :));
            kopt1.TrainMethod='MLE';
            for p4=1:2 %try different regularization
                kopt1.Regularization.Mode=strcat('L',num2str(p4));
                kopt1.InstanceKernel.AddTerms=AllowedKernels{p2};
                results_y = zeros(nbFold,M);
                results_z = zeros(nbFold,1);
                for r = 1:nbFold
                    XTrain = X(cvp.training(r), :);
                    YTrain = Y(cvp.training(r), :, :);
                    XTest = X(cvp.test(r), :);
                    YTest = Y(cvp.test(r), :, :);
                    ZTest = Z(cvp.test(r), :);
                    [~,model_ln] = cvwrapper(XTrain, YTrain, XTest, YTest, ZTest, errFunc, kopt1);
                    
                    
                    % set initial par
                    kopt2=kopt;
                    kopt2.InitZ=model_ln.Z;
                    kopt2.InitKappa=model_ln.InstanceKernel.Hyperparameters;
                    kopt2.TrainMethod='MAP';
                    
                    [results_y(r, :), results_z(r)] = cvwrapper(XTrain, YTrain, XTest, YTest, Ztest, errFunc, kopt2);
                    
                end
                kopt.UserErr=mean(results_y);
                kopt.LatentErr=mean(results_z);
                allresult=[allresult; kopt];
                save(fname,'allresult','-append');
            end
        end
    end
end
end


function [result_y,result_z, model] = cvwrapper(XTrain, YTrain, XTest, YTest, Ztest, errFunc, kopt)
model = trainCrowdGPLVM(XTrain, YTrain, kopt);

M= size(YTest,3);
result_y=zeros(1,M);
[mean_z, ~] = predictCrowdGPLVM(XTest, model, 0);
result_z = calError(mean_z,Ztest,errFunc);
% mean_y and var_y are NtxM matrix where Nt is the size of test set
for m = 1:M
    [mean_y, ~] = predictCrowdGPLVM(XTest, model, m);
    result_y(m)=calError(mean_y,YTest(:,:,m),errFunc);
end
end

function result=calError(x,y,errFunc)
if strcmp(errFunc,'rmse')
    result = sqrt(mean((normalizeZ(x) - normalizeZ(y)).^2));
elseif strcmp(errFunc, 'mae')
    result = mean(abs((normalizeZ(x) - normalizeZ(y))));
end
end


function z=normalizeZ(z)
z=(z-min(z))/(max(z)-min(z));
end
