function [x_sol, res, iter] = grad_descent_BB_Q6(f, grad, x0, eps)

% parameters
alpha = 0.1;
delta = ones(size(x0))*0.01;
L = norm(grad(x0 + delta) - grad(x0))./norm(delta);
tau = 2/L;
x = x0;

iter = 1;
norm_grad_value(iter) = norm(grad(x));

for iter = 1:200
    
    iter_number = sprintf('Iteration Number: %d', iter);
    disp(iter_number)
    d = -(grad(x));
%     Backtracking Condition
    while(f(x + tau*d) > f(x) + alpha*(tau*(d))'*grad(x))
        tau = tau/2;
    end
    
%     Update Rules
    x_new = x + tau*d;
    tau = ((x_new-x)'*(x_new-x))/((x_new-x)'*(grad(x_new)-grad(x)));
    norm_grad_value(iter) = norm(grad(x_new))
    x = x_new;
    
end


x_sol = x_new;
res = norm_grad_value;













end