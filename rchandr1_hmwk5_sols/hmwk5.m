% Homework 5
%% Question 2
% clc
% close all
% clear all
% 
% % Create 2 moons data
% [Data, L] = make2moons(100);
% 
% % view it
% % scatter(Data(1:50,1),Data(1:50,2));
% 
% sigma = 0.08;
% k = 1;
% 
% %Creating S matrix 
% S = exp(-pdist2(Data,Data)/sigma);
% 
% 
% % creating D matrix
% for i = 1:size(Data,1)
%     
%     D(i,i) = sum(S(i,:));
%    
% end
% 
% % Normalizing S
% norm_S = ((inv(D.^(0.5)))*S)*(inv(D.^(0.5)));
% 
% % eigenvalue decomposition
% [Vectors, Values] = eig(norm_S);
% 
% % Extracting 2nd and 3rd eigenvectors
% U2 = Vectors(:,2:3);
% 
% subplot(1,2,1)
% plot(U2(:,1),U2(:,2),'r.','MarkerSize',10);
% title('Linearly Seperable')
% 
% idx = kmeans(U2,2);
% while (numel(find(idx(1:50)==1)<49))
% idx = kmeans(U2,2);
% end
% subplot(1,2,2)
% plot(Data(find(idx==1),1),Data(find(idx==1),2),'r.','MarkerSize',10);
% hold on
% plot(Data(find(idx==2),1),Data(find(idx==2),2),'b.','MarkerSize',10)
% title('clustering')
% saveas(gcf,'1e2_points_clustering.jpg')
% 
% 
% 
% %% Question 1
% 
clc
close all
clear all

% Noisy Image
b = im2double(rgb2gray(imread('lena.jpg')));
b = b + rand(size(b));
MU = 10;

% Cretaing function handle
x = zeros(size(b));
compute_A = @(x)(MU.*div2d(grad2d(x)) + eye(size(b,1))*x);

%  Doing richardson
[X, R] = richardson(compute_A, b, x, 0.01);

% Doing ConjGrad
% [X,R] = conjgrad(compute_A, b, x);

% Doing L2 Denoise
% x = l2denoise(b,MU);
% N = norm(b-compute_A(x),'fro')
% subplot(1,2,1)
% imagesc(b)
% title(['The residual norm calculated is:' num2str(N)])
% subplot(1,2,2)
% imagesc(x)
% saveas(gcf,'l2denoise.jpg')

% %% Question 3
% 
% clc
% close all
% clear all
% sigma =0.08;
% % Create 2 moons data
% [Data, L] = make2moons(1e5);
% numpts = 1e5;
% n = 200;
% m = numpts - n;
% Un = randperm(numpts);
% R = randperm(numpts,n);
% T = setdiff(Un,R);
% 
% 
% sampled_data = Data(R,:);
% unsampled_data = Data(T,:);
% 
% 
% A = exp(-pdist2(sampled_data,sampled_data)/sigma);
% B = exp(-pdist2(sampled_data,unsampled_data)/sigma);
% 
% d1 = sum([A;B'],1);
% d2 = sum(B,1) + sum(B',1)*inv(A)*B;
% dhat = sqrt(1./[d1, d2])';
% 
% A = A.*(dhat(1:n)*dhat(1:n)');
% B = B.*(dhat(1:n)*dhat(n+1:n+m)');
% 
% S = A + A^-.5*B*B'*A^-.5;
% [U, L, V] = svd(S);
% 
% Q = [A; B']*A^-.5*U*L^-.5;
% QNorm = bsxfun(@rdivide, Q, Q(:,1));
% 
% Q2 = QNorm(:,2:3);
% subplot(1,2,1)
% plot(Q2(:,1), Q2(:,2),'r.','MarkerSize',10)
% title('Linearly Seperable')
% 
% idx = kmeans(Q2,2);
% 
% subplot(1,2,2)
% plot(Data(find(idx==1),1),Data(find(idx==1),2),'r.','MarkerSize',10);
% hold on
% plot(Data(find(idx==2),1),Data(find(idx==2),2),'b.','MarkerSize',10);
% title('clustering')
% saveas(gcf,'1e5_points_clustering.jpg')
