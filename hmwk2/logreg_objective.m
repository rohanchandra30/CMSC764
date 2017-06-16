%  This function implements the loss function of a logistic regression with
%  training data stores in 'D' and +1/-1 class labels stored in 'c'.

function y = logreg_objective(x,D,c)

Nd = size(D,1);
f = zeros (Nd,1) ; % allocate space to store objective values for each data point
z = c.*(D*x); % Compute CDx
f(z>=0) = log(1+exp(-z(z>=0))); % compute terms in f 
f(z<0) = -z(z<0)+log(1+exp(z(z<0)));
y = sum(f); % sum everything

end

