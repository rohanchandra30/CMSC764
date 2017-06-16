%% Question 1
clc
clear all
close all


% Set values
tau = 1;
C = 10;
eps = 1e-3;
index = 1;
N = 100;
Nfeat = 20;

% Create Problem
[D, c] = create_classification_problem(N, Nfeat, 10);

A = c.*D;

% initial values
x_new = zeros(Nfeat,1);
y_new = zeros(N,1);
p(index) = N;
d(index) = N;
lambda_new = zeros(N,1);

tic
while(p(index) > eps || d(index) > eps)
    
    index = index + 1;
    lambda_current = lambda_new;
    x_current = x_new;
    y_current = y_new;
    
    %   Update Y
    for i = 1:N
        z = A(i,:)*x_current - lambda_current(i);
        y_new(i) = z + max(min(1 - z, C/tau), 0);
    end
    
    %   Update X
    LHS = eye(Nfeat) + tau*(A'*A);
    RHS = tau*A'*(y_new + lambda_current);
    x_new = LHS\RHS;
    
    %   Update Lambda
    for j = 1:N
        lambda_new(j) = lambda_current(j) + y_new(j) - A(j,:)*x_new;
        
    end
    
    %   Compute residuals
    p(index) = norm(A*x_new - y_new);
    d(index) = norm(tau*A'*(y_new - y_current));
    
    
end

output_labels = sign(D * x_new);
accuracy = length(find(~(output_labels - c)))
figure
semilogy(p, 'g');
hold on
semilogy(d, 'r');
grid on
legend('primal residal', 'dual residual')
xlabel('Number of Iterations')
ylabel('Semilog Residual Error')
title('Convergence curve for tau = 1')
path = sprintf('tau1.jpg');
saveas(gcf, path)
toc

%% Question 2

clc
clear all
close all
tic
%  Set Values
N = 100;
per = .6;
Nfeat = 20;

% Create Problem
[D, c] = create_classification_problem(N, Nfeat, 1);

training_labels = c(1:per*N,:);
testing_labels = c(per*N + 1:N,:);
svm = svmtrain(D(1:per*N,:), c(1:per*N,:));
test = svmclassify(svm, D(per*N + 1:N,:));

%Accuracy
temp = test - testing_labels;
accuracy = (numel(find(temp==0))/numel(test))*100
toc
