function obj = tv_objective( x, b, mu, eps )
obj = mu*l1_eps(grad2d(x),eps) + 0.5*norm(x-b,'fro')^2;
return

function rval = l1_eps(x, eps)
rval = sum(sqrt(x(:).^2+eps.^2));
return

