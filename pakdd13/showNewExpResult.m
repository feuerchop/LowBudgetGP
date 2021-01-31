function showNewExpResult(expsetup)

modelName={'LVMOB','LVMOB2','LVMOB3','LVM','AVG'};
measureName={'PCC','SCC','NMAE'};

for m=1:length(modelName)
    result=zeros(3,expsetup.NumberOfRepeats);
    for p=1:expsetup.NumberOfRepeats
        for t=1:length(measureName)
            cmdStr=sprintf('result(%d,p)=abs(expsetup.%s{p}.gt.%s);',t,modelName{m},measureName{t});
            eval(cmdStr)
        end
    end
    
    fprintf('-%s-\n',modelName{m});
    showTable(result,measureName);
end

end

function showTable(result, tablecol)
fprintf('metric\tmin\tmax\tavg\tstd\n');
for j=1:3
    fprintf('%s\t%.2f\t%.2f\t%.2f\t%.2f\n',tablecol{j}, min(result(j,:)),max(result(j,:)),mean(result(j,:)),std(result(j,:)));
end
end