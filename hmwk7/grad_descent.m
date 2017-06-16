function [x_sol, res, iter] = grad_descent(f, grad, x0, eps)

% Parameters
alpha = 0.1;
delta = ones(size(x0))*0.01;
L = norm(grad(x0 + delta) - grad(x0))/norm(delta);
tau = 2/L;
x = x0;

iter = 1;
norm_grad_value(iter) = norm(grad(x));
while norm_grad_value(iter) > norm(grad(x0))*eps
   
    iter = iter + 1 
    d = -(grad(x));
%     Backtracking Condition
    while(f(x + tau*d) > f(x) + alpha*(tau*(d))'*grad(x))
        tau = tau/2;
    end
%     Update Rules
    x = x + tau*d;
    norm_grad_value(iter) = norm(grad(x));
end

x_sol = x;
res = norm_grad_value
end