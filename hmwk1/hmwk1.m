%% Solutions to hmwk1
%% CMSC764 - Tom Goldstein


%% Build 3x5 matrix with condition number 10
M = buildmat(3,5,2);
disp('3x5 matrix');
disp(M);
disp('Condition for 3x5 matrix');
disp(cond(M))


%% Build 5x3 matrix with condition number 10
M = buildmat(5,3,2);
disp('5x3 matrix');
disp(M);
disp('Condition for 5x3 matrix');
disp(cond(M))

%% Build 5x5 matrix with condition number 2
M = buildmat(5,5,2);
disp('square matrix');
disp(M);
disp('Condition for square matrix');
disp(cond(M))
