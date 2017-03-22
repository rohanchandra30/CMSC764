function M=buildmat(m,n,condNumber)
    
    M=randn(m,n);
    [U,S,V]=svd(M);
    S(S~=0)=linspace(condNumber,1,min(m,n));
    fprintf('%dx%d Matrix\n',m,n)
    M=U*S*V'
    
    fprintf('Condition number:')
    x=cond(M);
    x
end
    
