%%  Create a cell array of random Gaussian neural net weights

function W = create_random_net( dims )

W = {};
for l = 1:numel(dims)-1
    W{l} = randn(dims(l),dims(l+1))./sqrt(dims(l));
end

