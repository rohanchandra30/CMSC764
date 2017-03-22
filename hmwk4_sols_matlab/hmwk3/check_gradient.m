% hmwk 2:  Check that 'grad' computed the gradient of 'f'

function didpass = check_gradient( f, grad, x0 )
% a random permutation
delta = randn(size(x0));
delta = delta/norm(delta,'fro')*norm(x0,'fro');
% the gradient
g = grad(x0);
errors = zeros(10,1);
% loop over different perturbation sizes, and test accuracy
for j = 1:10
    delta = delta/10;
    d1 = f(x0+delta) - f(x0);
    d2 = delta(:)'*g(:);
    errors(j) = (d2-d1)/d1;
    fprintf('diff1 = %1.6d, diff2  = %1.6d, error = %1.6d\n',d1,d2,(d2-d1)/d1);
end
minerr = min(abs(errors));
didpass = minerr<1e-6;

fprintf('smallest error = %d\n',minerr);
if didpass
    disp('Test passed')
else
    disp('Test failed')
end

