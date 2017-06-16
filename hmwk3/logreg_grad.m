function g = logreg_grad(x,D,c)

Nd = size(D,1);
f = zeros (Nd,1) ; % allocate space to store objective values for each data point
z = c.*(D*x); % Compute CDx
f_prime = -exp(-z)./(1+exp(-z)); % grad(f) evaluated at CDx

% Chain rule: multiply by (CD)'=D'C'

g = D'*(c.*f_prime);

end

