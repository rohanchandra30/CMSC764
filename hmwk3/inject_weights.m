%  Replace the second layer of weights in w_cells with the weight matrix 
%  w_array, and return the result.  This is needed to hand the net gradient
%  to the gradient checker.

function w_cells = inject_weights( w_cells, w_array )
w_cells{2} = w_array;
end

