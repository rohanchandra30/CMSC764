%  Grab hdf5 data
file = '/mnist.h5';
x_test = h5read(file,'/x_test')';
y_test0 = h5read(file,'/y_test');
x_train = h5read(file,'/x_train')';
y_train0 = h5read(file,'/y_train');

% Convert to ones-hot representation
nd = size(y_train0,1);
y_train = zeros(nd,10);
y_train(sub2ind(size(y_train),1:nd,y_train0'+1)) = 1;

nd = size(y_test0,1);
y_test = zeros(nd,10);
y_test(sub2ind(size(y_test),1:nd,y_test0'+1)) = 1;

clear y_test0, y_train0;

% Convert to floating point 
x_test = double(x_test);
x_train = double(x_train);
y_test = double(y_test);
y_train = double(y_train);
