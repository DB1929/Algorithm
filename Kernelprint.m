function[] = Kernelprint(A,tildeA,gamma)


%% ========================================================================
% Building kernel data matrix, full or reduced.                           
%                                                                         
% Inputs                                                                  
% A: full data set                                                        
% tilde A: can be full or reduced set                                     
% gamma: width parameter; kernel value: exp(-gamma(Ai-Aj)^2)              
%                                                                         
% Outputs                                                                 
% K: kernel data using Gaussian kernel                                    
% by Kuang-Yao Lee                                                        
%==========================================================================
%%
row=size(A,1);
% square full kernel
    AA = repmat(sum(A.^2,2),1,row);
    K1 = exp((-AA-AA'+2*A*A')*gamma);
%reduced kernel, or kernel matrix for test data
    AA = repmat(sum(A.^2,2),1,size(tildeA,1));
    tildeAA = repmat(sum(tildeA.^2,2),1,row);
    K2 = exp((-AA-tildeAA'+2*A*tildeA')*gamma);

figure(1)
imagesc(K1)
colorbar
colormap(gray)
figure(2)
imagesc(K2)
colormap(gray)
colorbar
end