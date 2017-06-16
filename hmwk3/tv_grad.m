function grad = tv_grad( x, b, mu, eps )
grad = mu*div2d(l1_grad(grad2d(x),eps)) + x-b;
return

function rval = l1_grad(x, eps)
rval = x./sqrt(x.*x+eps*eps);
return

