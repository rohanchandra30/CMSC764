% hmwk 4:  the gradient of the neural net objective

function dw = net_gradient( weights, data, labels )
num_lays = numel(weights);
y = {data};
z = {data*weights{1}};
% Forward pass: each layer performs data*weights.  This way we have 1 feature vector per row of data
for j = 2:num_lays
    y{j} = smrelu(z{j-1});  % y stores the activations
    z{j} = y{j}*weights{j}; % z stores the inputs to the next layer
end

%  The backward pass:  do the chain rule
dzt = log_entropy_grad(z{num_lays}, labels);  % store the gradient with respect to z
dw{num_lays} = {y{num_lays}'*dzt};            % store gradient with respect to weight matrices
for j = num_lays-1:-1:1
    dzt = dzt*weights{j+1}'.*smrelu_grad(z{j});  % compute the next z gradient
    dw{j} = y{j}'*dzt;                           % compute the next w gradient
end
return


function rval = smrelu(x)
%   the smoothed relu function
rval = log(1+exp(x));
% make sure we don't exponentiate positive numbers
ind = x>=0;
rval(ind) = x(ind) + log(1 + exp(-x(ind)));
return


function rval = smrelu_grad(x)
% the smoothed relu gradient
rval = zeros(size(x));
ind = x<0;
rval(ind) = exp(x(ind))./(1+exp(x(ind)));
ind = x>=0;
rval(ind) = 1./(1+exp(-x(ind)));
return 


function grad = log_entropy_grad(z, ones_hot)
%   hmwk 4:  the gradient of the log entropy of the softmax
Nc = size(z,2);  % The number of classes
Nd = size(z,1);  % The number of data observations
% shift everything so we don't have to exponentiate positive numbers
z = z-max(z,[],2)*ones(1,Nc);
% compute the negative log likelihood
s = sum(exp(z),2);
grad = -ones_hot + exp(z)./(s*ones(1,Nc));
grad = grad/Nd;
return
