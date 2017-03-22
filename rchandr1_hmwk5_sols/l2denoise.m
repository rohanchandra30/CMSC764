function x =l2denoise(b ,mu)
laplacian = zeros(size(b));
laplacian(1, 1) = 4;
laplacian(1,2) = -1;
laplacian(2,1) = -1;
laplacian(1,end) = -1;
laplacian(end,1) = -1;
x = ifft2(fft2(b)./(mu*fft2(laplacian) + ones(size(b,1))));




end