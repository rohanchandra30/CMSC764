% Build a random matrix mxn matrix with condition number 'cond'.
function M = buildmat(m,n,cond)

if m<=n
    % Random orthogonal matrix
    U = orth(randn(m,m));
    Vt = orth(randn(n,n));
    % Build geometric progression from 1 to cond
    S = logspace(0,log10(cond),m);
    % Invert the progression so eigs go from 1 down to 1/cond
    S = 1./S;
    % Build the matrix from its SVD components
    M = U.*(ones(m,1)*S)*Vt(1:m,:);
else
    M = buildmat(n,m,cond)';
end

