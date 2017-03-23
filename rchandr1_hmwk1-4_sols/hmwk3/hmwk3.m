
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                               Question 1 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% test gradient checker
fprintf('Problem 1: Testing gradient checker\n');
A = randn(3,3);
b = randn(3,5);
x = randn(3,5);
f = @(x) 0.5*norm(A*x-b,'fro')^2;
grad = @(x) A'*(A*x-b);
check_gradient( f, grad, x );


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                               Question 2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\nProblem 2: Testing TV gradient\n');
%% Test tv objective
b = randn(10,5);
x = randn(10,5);
mu = 1;
eps = .01;
f = @(x) tv_objective( x, b, mu, eps );
grad = @(x) tv_grad( x, b, mu, eps );
check_gradient( f, grad, x );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                               Question 3
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\nProblem 2: Testing logreg gradient\n');
%% test logreg objective function
[D,c] = create_classification_problem(100,10,5);
x = randn(10,1);
f = @(x) logreg_objective( x, D, c );
grad = @(x) logreg_grad( x, D, c );
check_gradient( f, grad, x );

