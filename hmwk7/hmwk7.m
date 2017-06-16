% Decide which question to answer
x = input('Enter 1 for Questions 1, 2, 3, 4 and 5. Enter 2 for Question 6')


if x == 1
    clc
    clear all
    close all
    
    %     Set Parameters
    eps = 1e-7;
    num_feat = 200;
    num_featpervec = 20;
    
    
    for i = 1:2
        close all
        if i == 1
            CondNumber = 1;
        else
            CondNumber = 100;
        end
        
        %     Gradient Descent Method
        %     Setting the probem
        [f, grad, x0] = create_problem(num_feat, num_featpervec, CondNumber);
        
        [x_sol, res, iter] = grad_descent(f, grad, x0, eps);
        
        %     Plotting Gradient method
        semilogy(res)
        xlabel('iteration number');
        ylabel('Residual error');
        hold on
        disp('BB')
        pause(3);
        
        
        %     BB Method
        %     Setting the probem
        [f, grad, x0] = create_problem(num_feat, num_featpervec, CondNumber);

        [x_sol, res, iter] = grad_descent_BB(f, grad, x0, eps);
        %     Plotting BB
        semilogy(res)
        xlabel('iteration number');
        ylabel('Residual error');
        disp('nesterov')
        pause(3);
        hold on
        
        
        %     Neserov Method
         %    Setting the probem
        [f, grad, x0] = create_problem(num_feat, num_featpervec, CondNumber);

        [x_sol, res, iter] = grad_descent_nesterov(f, grad, x0, eps);
        
        %     Plotting Nesterov
        semilogy(res)
        xlabel('iteration number');
        ylabel('Residual error');
        legend('Grad', 'BB', 'Nestrov');
        title(['Residual Error plots for \kappa = ', num2str(CondNumber)]);
        path = sprintf('descent_plots with kappa = %d.jpg', CondNumber);
        saveas(gcf, path)
    end
    
else
    %     -----------------------------------------------------------------------------------
    %    Question 6
    clear all
    clc
    close all
    load_mnist;
%     Create Weights
    W = create_random_net([784,50,40, 30, 20,10]);
%     Convert them to vector
    W_vector = cell_to_vec(W);
%     Prepare data and labels
    D = x_train;
    D = D(:, [1:784]) - mean(D);
    l = y_train;
    
%     Prepare f and grad
    f = @(W_vector)net_objective(W_vector, D, l, W);
    grad = @(W_vector)net_gradient(W_vector, D, l, W);
    
    if ~exist('hmwk7.mat', 'file')
%     Perform gradient Descent
        [x_sol, res, iter] = grad_descent_BB_Q6(f, grad, W_vector, 1e-8);
        save 'hmwk7.mat'
    else
        load hmwk7.mat
    end
%     Plot convergence curve
    semilogy(res)
    xlabel('iteration number');
    ylabel('Residual error');
    title('convergence plot for MNIST data');
    legend('BB Method');
    saveas(gcf, 'Q6_plot.jpg')
    
    weights_optimum = vec_to_cell(x_sol, W);
    
%     Test for Training Error
    output_train = run_network(D, weights_optimum);
    [~, output_train] = max(output_train');
    [~, labels_train] = max(l');
    accuracy_train = (numsamples(labels_train, output_train)/length(labels_train))*100;
    disp(['Accuracy for Training Set = ', num2str(accuracy_train)])
    
    D = x_test;
    D = D(:, [1:784]) - mean(D);
    l = y_test;

%     Test for Testing Error.     
    output_test = run_network(D, weights_optimum);
    [~, output_test] = max(output_test');
    [~, labels_test] = max(l');
    accuracy_test = (numsamples(labels_test, output_test)/length(labels_test))*100;
    disp(['Accuracy for Test Set = ', num2str(accuracy_test)])
    
end
