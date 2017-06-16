function [ D,c ] = create_classification_problem(Nd, Nf, kappa)
D = buildmat(Nd,Nf,kappa);
w = randn(Nf,1);
c = sign(D*w);
%  The the problem not quite separable by flipping some signs.
flipper = ones(Nd,1) - 2*(rand(Nd,1)>.9);
c = c.*flipper;
end

