%  This function implements the objective function of a neural net with
%  weights given by the cell array 'weights', the feature vectors stored in
%  the rows of 'D', and the ones-hot labels stores in the rows of 'labels'.

function [ obj ] = net_objective( weights, data, labels )
%  hmwk 2:  the objective function of a neural net
num_lays = numel(weights);
z = data*weights{1};
% Each layer performs data*weights.  This way we have 1 feature vector per row of data
for j = 2:num_lays
    z = smrelu(z)*weights{j};
end
obj = log_entropy_softmax(z, labels);
return


function rval = smrelu(x)
%   the smoothed relu function
rval = log(1+exp(x));
% make sure we don't exponentiate positive numbers
ind = x>=0;
rval(ind) = x(ind) + log(1 + exp(-x(ind)));
return


function rval = log_entropy_softmax(z, ones_hot)
%   hmwk 2:  the log entropy of the softmax layer"""
Nc = size(z,2);  % The number of classes
Nd = size(z,1);  % The number of data observations
% shift everything so we don't have to exponentiate positive numbers
m = max(z,[],2);
z = z-m*ones(1,Nc);
% compute the negative log likelihood
s = sum(exp(z),2);
nll = -z+log(s)*ones(1,Nc);
% sum over the entries corresponding to the correct class
rval = sum(sum(nll.*ones_hot))/Nd;
return