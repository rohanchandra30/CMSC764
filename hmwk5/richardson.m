function [x, resids] = richardson(f, b, x, t)

k = 1;
x = x + t*(b - f(x));
r = b - f(x);
resids(k) = norm(r,'fro');

while resids >1e-6
    k = k+1;
    x = x + t*(b - f(x));
    r = b - f(x);
    resids(k) = norm(r,'fro');
    
end
plot(1:length(resids),resids);
xlabel('Number of iterations');
ylabel('Convergence error')
title(['Number of iterations for mu = 10 are:' num2str(k)])
saveas(gcf,'richardson_10.jpg')
figure
subplot(1,2,1)
imagesc(b)
subplot(1,2,2)
imagesc(x)
saveas(gcf,'richardson_denoised_10.jpg')
disp(k)





end