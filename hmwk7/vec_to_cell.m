%  Convert a vector of unknowns into a cell array with the same shape as
%  the argument 'cells'

function cells = vec_to_cell( vec, cells )
loc = 1;
for i=1:numel(cells)
    c = cells{i};
    cells{i} = reshape(vec(loc:loc+numel(c)-1), size(c));
    loc = loc+numel(c);
end

