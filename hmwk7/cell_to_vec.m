% Convert a cell array of matrices into a vector

function vec = cell_to_vec( cells )
vec = [];
for i = 1:numel(cells)
    cell = cells{i};
    vec = [vec;cell(:)];
end

end

