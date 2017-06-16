function Div = div2d( G )

%  extract the x- and y- derivatives
Gx = squeeze(G(:,:,1));
%  extract the x- and y- derivatives
Gy = squeeze(G(:,:,2));

% I want to filter my image with the forward-difference stencil [0 1 -1].  
% Remember, you must convolve with the FLIPPED stencil to get the linear
% filtering you want.  So, I will convolve with this:
kernel = zeros(size(Gx));
kernel(1,1)=1;
kernel(1,end)=-1;
%  create diagonal matrix in decomposition K=F'DF
Dx = fft2(kernel);
%  Use the eigen-decomposition to convolve the stencil with X, and get the
%  differences in the horizontal direction.
Divx = ifft2(Dx.*fft2(Gx));

% I want to filter my image with the forward-difference stencil [0 1 -1]'.  
% Remember, you must convolve with the FLIPPED stencil to get the linear
% filtering you want.  So, I will convolve with this:
kernel = zeros(size(Gy));
kernel(1,1)=1;
kernel(end,1)=-1;
%  create diagonal matrix in decomposition K=F'DF
Dy = fft2(kernel);
%  Use the eigen-decomposition to convolve the stencil with X, and get the
%  differences in the horizontal direction.
Divy = ifft2(Dy.*fft2(Gy));

Div = Divx+Divy;


end

