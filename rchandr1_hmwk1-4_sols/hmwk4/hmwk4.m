%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                               Question 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\n\nProblem 2: Testing net gradient\n');
% Extract dataset and net for MNIST
load_mnist;
W = create_random_net([784,50,50,10]);

% Define the dataset we'll use for the check
w2 = randn(50,50);  % the initial value for the weights at layer 2.
D = x_train;
l = y_train;

net_objective(W,D,l);
net_objective(inject_weights(W,w2),D,l);
% Check the gradient of the net using layer 2
extract_grad = @(W) W{2};
f = @(w2) net_objective(inject_weights(W,w2),D,l);
grad = @(w2) extract_grad(net_gradient(inject_weights(W,w2),D,l));
check_gradient( f, grad, w2 );