function [x, resids] = conjgrad(f, b, x)
close all
k = 1;
p = b;
r = b - f(x);
resids(1) = norm(r,'fro');
while resids(end) > 1e-6
    k = k+1;
    alpha = norm((p'*r),'fro') / norm((p'*f(p)),'fro');
    x = x + alpha * p;
    r = r - alpha * f(p);
    p = r - p*alpha;
    resids(k) = norm(r,'fro');
    
end
plot(1:length(resids),resids);
xlabel('Number of iterations');
ylabel('Convergence error')
title(['Number of iterations for mu = 10 are:' num2str(k)])
saveas(gcf,'conjgrad_10.jpg')
figure
subplot(1,2,1)
imagesc(b)
subplot(1,2,2)
imagesc(x)
saveas(gcf,'conjgrad_denoised_10.jpg')
disp(k)

end
