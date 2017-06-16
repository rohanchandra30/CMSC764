function [f, grad, x0] = create_problem(num_feat, num_featpervec, CondNumber)

[D, c] = create_classification_problem(200, 20, CondNumber);
f = @(x) logreg_objective(x, D, c);
grad = @(x) logreg_grad(x, D, c);
x0 = rand(size(D,2),1);





end