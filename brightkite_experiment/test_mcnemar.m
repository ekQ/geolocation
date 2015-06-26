function test_mcnemar(all_errors, all_errors_bl, max_error, alpha, iter)

critical_value = chi2inv(1-alpha,1);

if iscell(all_errors)
    for j = 1:size(all_errors,1)
        a = []; b = [];
        for i = 1:size(all_errors,2)
            a = [a; all_errors{j,i}(:,iter)<=max_error];
            b = [b; all_errors_bl{j,i}(:,iter)<=max_error];
        end
        x = zeros(2,2);
        x(1,1) = sum(a==0 & b==0);
        x(1,2) = sum(a==0 & b==1);
        x(2,1) = sum(a==1 & b==0);
        x(2,2) = sum(a==1 & b==1);
        %mcnemar(x)
        chi2 = (abs(x(1,2)-x(2,1))-1)^2 / (x(1,2)+x(2,1));
        if chi2 > critical_value
            fprintf('s ');
        else
            fprintf('n ');
        end
    end
    fprintf('\n');
else
    a = all_errors(:,iter);
    b = all_errors_bl(:,iter);
    x = zeros(2,2);
    x(1,1) = sum(a==0 & b==0);
    x(1,2) = sum(a==0 & b==1);
    x(2,1) = sum(a==1 & b==0);
    x(2,2) = sum(a==1 & b==1);
    mcnemar(x,alpha)
    chi2 = (abs(x(1,2)-x(2,1))-1)^2 / (x(1,2)+x(2,1));
    if chi2 > critical_value
        fprintf('significant\n');
    else
        fprintf('not significant\n');
    end
end