function [Result] = Prediction_v2(Model,VInst,VLabel)
%% ========================================================================
% JIAN-PING SYU
%
% Prediction Phase
% ! add the proximal model test loss
%==========================================================================
% Read Me =================================================================
% Inputs :
%        (1) Model        : Initial of the Model
%            Model.W      : classifiers (w,b)   (rs1+1 x RoundNum) 
%            Model.RS     : Reduce set of Kernel
%            Model.gamma  : Parameter of RBF Kernel
%
%        (2) VInst        : Testing dataset
%        (3) VLabel       : Labels of VInst
%
% Outputs :
%        (1) Result.test   : Result in testing phrase
%             - Result.test.PVLabel : Prediction of each model in testing data
%             - Result.test.Testing_error
%             - Result.test.[min\max\avg\std]_testing_err
%             - Result.test.ModelRelate : Model relative for each models
%==========================================================================
%%

PV = [];
PV2 = [];
InstNum = size(VLabel,1);
[~,ModelNum] = size(Model.W);

% data partition
PartNum = floor(InstNum/100);
res = InstNum-PartNum*100;
if res ~=0
    PartNum = PartNum + 1;
end
ind_end = 0;

for part = 1:PartNum
    if part == PartNum && res ~=0
        ind_start = ind_end + 1;
        ind_end = InstNum;
    else
        ind_start = ind_end + 1;
        ind_end = part*100;
    end

    KVInst = KGaussian(Model.gamma,VInst(ind_start:ind_end,:),Model.RS);
    PV = [PV;[KVInst,ones((ind_end-ind_start+1),1)]*Model.W];
    PV2 = [PV2;[KVInst,ones((ind_end-ind_start+1),1)]*Model.wp];
end   
    % PV
    PVLabel = zeros(size(VLabel,1),ModelNum);
    PVLabel(PV>0) = 1;
    PVLabel(PV<0) = -1;
    Result.PVLabel = PVLabel;
    Result.Testing_error = sum(PVLabel-repmat(VLabel,1,ModelNum)~=0,1).*100./size(VLabel,1);
    Result.min_testing_err = min(Result.Testing_error);
    Result.max_testing_err = max(Result.Testing_error);
    Result.avg_testing_err = mean(Result.Testing_error);
    Result.std_testing_err = std(Result.Testing_error);
    % PV2 - proximal model
    PVLabelm = zeros(size(VLabel,1),1);
    PVLabelm(PV2>0) = 1;
    PVLabelm(PV2<0) = -1;
    Result.Test_err_prox = sum(PVLabelm-VLabel~=0)*100/size(VLabel,1);
    % Model relative
    if ModelNum >1
       ModelRelate =[];
       for i = 1:ModelNum-1
           ModelRelate  = [ModelRelate,Model.W(:,i)'*Model.W(:,i+1)/(norm(Model.W(:,i))*norm(Model.W(:,i+1)))];
       end
       Result.ModelRelate  = ModelRelate;
    else
       Result.ModelRelate = []; 
    end

end





function K = KGaussian(gamma, A, tildeA)
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
%=========================================================================%
%%
row=size(A,1);
if nargin<3, % square full kernel
    AA = repmat(sum(A.^2,2),1,row);
    K = exp((-AA-AA'+2*A*A')*gamma);
else   %reduced kernel, or kernel matrix for test data
    AA = repmat(sum(A.^2,2),1,size(tildeA,1));
    tildeAA = repmat(sum(tildeA.^2,2),1,row);
    K = exp((-AA-tildeAA'+2*A*tildeA')*gamma);
end
end
