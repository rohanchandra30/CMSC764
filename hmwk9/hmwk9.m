%% Question 2

clc
clear all
close all
[D, c] = create_classification_problem(200, 5, 1);
random = normrnd(0, 1, [200 5]);
D_hat = [D, random];

mu = 3;    %  regularization parameter

%  The initial value
x0 = zeros(10,1);
opts = [];
opts.accelerate = false;
opts.adaptive = false;

% Perform FASTA
[sol, outs_adapt] = fasta_sparseLogistic_Rohan(D_hat,[],c,mu,x0, opts);
semilogy(outs_adapt.residuals,'g');
grid on;
title('Convergence Curve: Logistic Regression')
xlabel('Iteration Number')
ylabel('residual Error')
saveas(gcf, 'need_this_for_report/LogisticRegression.jpg');
figure
plot(sol, 'r.', 'MarkerSize', 20)
% saveas(gcf, 'need_this_for_report/LogisticRegression_plot.jpg');

%% Question 3
clc
clear all
close all
% Read in Image
I = im2double(phantom());
I = imnoise(I, 'gaussian', 0,0.001);

figure
subplot(1,2,1)
imshow(I);

% Set values
mu = 1;
opts= [];
opts.accelerate = false;
opts.adaptive = false;

% Perform FASTA
[ denoised, outs ] = fasta_totalVariation_Rohan( I, mu, opts);

% Plot
subplot(1,2,2)
imshow(denoised)
saveas(gcf, 'need_this_for_report/TVplot.jpg');

figure
semilogy(outs.residuals,'g');
grid on;
title('Convergence Curve: Total Variation')
xlabel('Iteration Number')
ylabel('residual Error')
saveas(gcf, 'need_this_for_report/TV_residual.jpg');
%% Question 4
clc
clear all
close all

% Set Values
mu = 0;
A = rand(50, 3);
B = rand(50, 3);
D = A*B';
X0 = rand(50, 3);
Y0 = zeros(50, 3);
opts = [];
opts.accelerate = false;
opts.adaptive = false;
opts.recordObjective = true; %  Record the objective function so we can plot it
opts.verbose=true;
opts.stringHeader='    ';      % Append a tab to all text output from FISTA.  This option makes formatting look a bit nicer.


% Perform FASTA
[ Xsol,Ysol, outs_adapt ] = fasta_nonNegativeFactorization( D, X0, Y0, mu, opts );

% Plot
figure
subplot(2,2,1)
imagesc(A); title('Xtrue');
subplot(2,2,2)
imagesc(B); title('Ytrue');
subplot(2,2,3)
imagesc(Xsol); title('Xrecovered');
subplot(2,2,4)
imagesc(Ysol); title('Yrecovered');
saveas(gcf, 'need_this_for_report/MatrixFactorization_plots.jpg');
figure
semilogy(outs_adapt.residuals,'g');
grid on;
title('Convergence Curve: Matrix factorization')
xlabel('Iteration Number')
ylabel('residual Error')
saveas(gcf, 'need_this_for_report/MatrixFactorization_residual.jpg');