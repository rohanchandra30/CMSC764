%% Create 'two moons' dataset with n feature vectors and 2 features per 
%% vector.

function [data, labels] = make2moons(n)
%% Create points on two half circles
theta = (0:(2*pi/(n-1)):pi)';
x1 = cos(theta);
y1 = sin(theta);
x2 = cos(theta);
y2 = -sin(theta);

%% Shift the two half circles
x1 = x1+1;
y2 = y2+.5;

%%  Dump all x's and y's togeather
data = [x1 y1 ; x2 y2];

%% Add noise
data = data+randn(size(data))*.05;

%% Make labels
labels = [ones(n/2,1);-ones(n/2,1)];

