
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                               Question 1 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Problem 1: making plots\n');
subplot(1,2,1);
[ D,c ] = create_classification_problem(100, 2, 1);
scatter(D(c==1,1),D(c==1,2),'r');
hold on;
scatter(D(c==-1,1),D(c==-1,2),'b');
hold off;
title('kappa=1')
shg

subplot(1,2,2);
[ D,c ] = create_classification_problem(100, 2, 10);
scatter(D(c==1,1),D(c==1,2),'r');
hold on;
scatter(D(c==-1,1),D(c==-1,2),'b');
hold off;
title('kappa=10')
shg

%% test logreg objective function
x = randn(2,1);
fx = logreg_objective(x,D,c);
fprintf('Test value for logreg objective = %d\n',fx);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                               Question 2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\n\nProblem 2: Testing net objective\n');
load_mnist;
W = create_random_net([784,100,10]);
obj = net_objective(W,x_train,y_train);
fprintf('    objective=%1.3d\n',obj);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                               Question 3
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\n\nProblem 3: Testing adjoints\n');
% random inputs
x = randn(100,100);
y = randn(100,100,2);
%  Compute Ax, where A is the gradient operator
Ax = grad2d(x);
%  Compute Ax, where A is the gradient operator
Aty = div2d(y);
% Compute <Ax,y>
left = vec(Ax)'*vec(y);
right = vec(x)'*vec(Aty);

fprintf('first inner product = %d\n',left);
fprintf('second inner product = %d\n',right);