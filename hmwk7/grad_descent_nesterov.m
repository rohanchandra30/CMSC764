function [x_sol, res, iter] = grad_descent_nesterov(f, grad, x0, eps)
% Paarameters
eps = 1e-6;
alpha = 0.5;
delta = ones(size(x0))*0.01;
sigma = 1;
L = norm(grad(x0 + delta) - grad(x0))/norm(delta);
tau = 2/L;
x = x0;
y = x0;

iter = 1;
norm_grad_value(iter) = norm(grad(x));

while norm_grad_value(iter) > norm(grad(x0))*eps
    
    iter = iter + 1
%     backtracking Condition
    while(f(y - tau*grad(y)) > f(y) - alpha*(tau*grad(y))'*grad(y))
        tau = tau/2;
    end
%     Update Rules
    x_new = y - tau*grad(y);
    sigma_new = (1 + sqrt(1 + 4*sigma^2))/2;
    y_new = x_new + ((sigma - 1)/sigma_new)*(x_new - x);
    
    norm_grad_value(iter) = norm(grad(x_new))
    x = x_new;
    y = y_new;
    sigma = sigma_new;
end

x_sol = x_new;
res = norm_grad_value;



end


