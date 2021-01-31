function allresult=crossValid( X, Y, nbFold, errFunc)

[N,D,M] = size(Y);
if nbFold>N
    error('data set is too small for cross-validation!')
end
cvp = cvpartition(N, 'kfold', nbFold);
allresult=[];
save(fname,'allresult');

%% for MAP
kopt=koptDefault(Y(cvp.training(1), :, :));
kopt.TrainMethod='MAP';
for p1=1:7 % for all user kernels
    kopt.UserKernel.AddTerms=dec2bin(p1,3);
    for p2=1:7 % for all instance kernels
        kopt.InstanceKernel.AddTerms=dec2bin(p2,3);
        for p3=0:1 % turn on ARD for user kernels
            kopt.UserKernel.ARD=p3;
            results = zeros(nbFold,M);
            parfor r = 1:nbFold
                % for each folder
                XTrain = X(cvp.training(r), :);
                YTrain = Y(cvp.training(r), :, :);
                XTest = X(cvp.test(r), :);
                YTest = Y(cvp.test(r), :, :);
                results(r, :) = cvwrapper(XTrain, YTrain, XTest, YTest, errFunc, kopt);
            end
            kopt.rmse=mean(results);
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
    for p2=1:7 % for all instance kernels
        kopt.InstanceKernel.AddTerms=dec2bin(p2,3);
        results = zeros(nbFold,M);
        parfor r = 1:nbFold
            % for each folder
            XTrain = X(cvp.training(r), :);
            YTrain = Y(cvp.training(r), :, :);
            XTest = X(cvp.test(r), :);
            YTest = Y(cvp.test(r), :, :);
            results(r, :) = cvwrapper(XTrain, YTrain, XTest, YTest, errFunc, kopt);
        end
        kopt.rmse=mean(results);
        allresult=[allresult; kopt];
        save(fname,'allresult','-append');
    end
end


%% for MAP
kopt=koptDefault(Y(cvp.training(1), :, :));
kopt.TrainMethod='MAP';
for p1=1:7 % for all user kernels
    kopt.UserKernel.AddTerms=dec2bin(p1,3);
    for p2=1:7 % for all instance kernels
        kopt.InstanceKernel.AddTerms=dec2bin(p2,3);
        for p3=0:1 % turn on ARD for user kernels
            kopt.UserKernel.ARD=p3;
            
            % for MLE-MAP
            kopt1=koptDefault(Y(cvp.training(1), :, :));
            kopt1.TrainMethod='MLE';
            for p4=1:2 %try different regularization
                kopt1.Regularization.Mode=strcat('L',num2str(p4));
                kopt1.InstanceKernel.AddTerms=dec2bin(p2,3);
                
                results = zeros(nbFold,M);
                for r = 1:nbFold
                    XTrain = X(cvp.training(r), :);
                    YTrain = Y(cvp.training(r), :, :);
                    XTest = X(cvp.test(r), :);
                    YTest = Y(cvp.test(r), :, :);
                    [~,model_ln] = cvwrapper(XTrain, YTrain, XTest, YTest, errFunc, kopt1);
                    
                    
                    % set initial par
                    kopt2=kopt;
                    kopt2.InitZ=model_ln.Z;
                    kopt2.InitKappa=model_ln.InstanceKernel.Hyperparameters;
                    kopt2.TrainMethod='MAP';
                    
                    results(r, :) = cvwrapper(XTrain, YTrain, XTest, YTest, errFunc, kopt2);
                    
                end
                kopt2.rmse=mean(results);
                allresult=[allresult; kopt];
                save(fname,'allresult','-append');
            end
        end
    end
end
end


function [result,model] = cvwrapper(XTrain, YTrain, XTest, YTest, errFunc, kopt)
model = trainCrowdGPLVM(XTrain, YTrain, kopt);

M= size(YTest,3);
result=zeros(1,M);
% mean_y and var_y are NtxM matrix where Nt is the size of test set
for m = 1:M
    [mean_y, ~] = predictCrowdGPLVM(XTest, model, m);
    if strcmp(errFunc,'rmse')
        result(m) = sqrt(mean((mean_y - YTest(:, :, m)).^2));
    elseif strcmp(errFunc, 'mae')
        result(m)= mean(abs(mean_y-YTest(:,:,m)));
    end
end
end
