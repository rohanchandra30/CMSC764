 function  G  = grad2d( X )
% I want to filter my image with the backward-difference stencil [-1 1 0].  
% Remember, you must convolve with the FLIPPED stencil to get the linear
% filtering you want.  So, I will convolve with this:
kernel = zeros(size(X));
kernel(1,1)=1;
kernel(1,2)=-1;
%  create diagonal matrix in decomposition K=F'DF
Dx = fft2(kernel);
%  Use the eigen-decomposition to convolve the stencil with X, and get the
%  differences in the horizontal direction.
Gx = ifft2(Dx.*fft2(X));  %  This array stores the x-differences

% I want to filter my image with the backward-difference stencil [-1 1]'.  
% Remember, you must convolve with the FLIPPED stencil to get the linear
% filtering you want.  So, I will convolve with this:
kernel = zeros(size(X));
kernel(1,1)=1;
kernel(2,1)=-1;
%  create diagonal matrix in decomposition K=F'DF
Dy = fft2(kernel);
%  Use the eigen-decomposition to convolve the stencil with X, and get the
%  differences in the horizontal direction.
Gy = ifft2(Dy.*fft2(X));

G(:,:,1) = Gx;
G(:,:,2) = Gy;
G = real(G);

end

