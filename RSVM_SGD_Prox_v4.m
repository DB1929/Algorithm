function [Model,Report] = RSVM_SGD_Prox_v4(Model,TF,Opt,Set,Inst,Label)
%% ========================================================================
% JIAN-PING SYU
% SGD with (1)Proximal Model
%           Reduce Kernel Support Vector Machine
%  ! Renew the Proximal Model
% Read Me ================================================================%
% Inputs :
%        (1) Model        : Initial of the Model
%            Model.W      : classifiers (w,b)   (rs1+1 x RoundNum) 
%            Model.RS     : Reduce set of Kernel
%            Model.gamma  : Parameter of RBF Kernel.
%
%        (2) TF           : Trade-off in objective function
%            TF.C         : Trade-off with Regularization
%            TF.C1        : Trade-off with Training  data loss
%            TF.C3        : Trade-off with Proximal Model
%
%        (3) Opt          : Parameter of optimization algorithm
%            Opt.eta      : Learning rate
%            Opt.beta     : Parameter of Hypergradient 
%            Opt.N        : Type of Learning rate choose
%                           0-> Hypergradient 
%                           1-> Fixed eta
%
%
%        (4) Set          : Setting the learning.
%            Set.Minibatch: Training data size in every iteration
%            Set.Epoch    : Number of epoch
%            Set.Overlap  : Overlapping times
%
%        (5) Data
%            Inst         : Training data
%            Label        : Training label
% Outputs :
%        (1) Model        : Initial of the Model
%            Model.W      : classifiers (w,b)   (rs1+1 x RoundNum) 
%            Model.RS     : Reduce set of Kernel
%            Model.gamma  : Parameter of RBF Kernel.
%
%        (2) Report
%            Report.time  : time comsuption for each epoch.
%            Report.loss  : Training loss for every iteration
%            Report.eta   : Learning rate in every iteration
%=========================================================================%

%% Setting ===============================================================%
    %% Data Partition=====================================================%
%Get dimension and size of data
[InstNum,~] = size(Inst);

%Number of iteration in single epoch
PartNum = floor(InstNum/Set.Minibatch);
res = InstNum-PartNum*Set.Minibatch;
if res ~=0
    PartNum = PartNum + 1;
end

ind_end = 0; % The end's index of traing data in every iteration 

    %% Initial Setting for Dataset =======================================%
%Dimension of the Kernel Space
nDim = size(Model.RS,1);          
rs1 = nDim + 1;                   

%Statistical Information with Proximal Model
Prox = ProximalModel(TF.C3);

%Model
w       = Model.W(:,1);                      %Initial the weight
%Model.W = zeros(rs1,Set.Epoch);
eta     = Opt.eta;

%Report
Report.time = zeros(Set.Epoch,1);
Report.loss = zeros(Set.Epoch,PartNum);
Report.eta  = zeros(Set.Epoch,PartNum);
%% Training Phase ========================================================%
for round = 1:Set.Epoch
    time = tic;
    
    % Random Permutation the data
    ind    = randperm(InstNum);
    TInst  = Inst(ind,:);
    TLabel = Label(ind,:);
    
    % Initial Setting before every epoch
    if round > 1
       w = Model.W(:,round-1); 
    elseif round ==1
       Hyper_grad = zeros(rs1,1);
    end
    ind_end = 0;    
    %Overlapping setting
    zKTInst_pre = [];
    miniTLabel_pre = [];
    counter_vector_pre = [];

%% Single Epoch ==========================================================%    
    for part = 1:PartNum
        
        %% Index of the iteration 
        if part == PartNum && res ~=0
            ind_start = ind_end + 1;
            ind_end = InstNum;
            miniInstNum = ind_end - ind_start + 1;
        else
            ind_start = ind_end + 1;
            ind_end = part*Set.Minibatch;
            miniInstNum = Set.Minibatch;
        end
             
        %% Training data & Label  
        zKTInst = KGaussian(Model.gamma,TInst(ind_start:ind_end,:),Model.RS);
        miniTLabel = TLabel(ind_start:ind_end,1); 
        
        %% Statistical Information
        if round==1
           Prox = StatInfo(Prox,zKTInst,miniTLabel);
        end
        
        %% Overlapping Strategty
        if part>1 && Set.Overlap > 1
            zKTInst = [zKTInst ; zKTInst_pre];
            miniTLabel = [miniTLabel ; miniTLabel_pre];
            counter_vector = [zeros(miniInstNum,1) ; counter_vector_pre];
        end
        
        %% Passive Part
        loss = 1 - miniTLabel.*(zKTInst*w(1:nDim)+w(end));   % loss vector          
        Ih = find(loss > 0);                                 % update model by using misclassified instance only.
        Report.loss(round,part) = sum(loss(Ih));
        nIh = size(Ih,1);
        
        %% Update Part
        
        if nIh>0
            if round==1
        %% Proximal model
                if Prox.n_p>1 && Prox.n_n>1                   
                    Prox = getProx(Prox);
                    Prox = Grad_prox(Prox);
                end
            end
                
         %% Gradient                        
               % Final gradient
                % gradw_part = loss(Ih).*miniTLabel(Ih)/nIh;
                gradw_part = loss(Ih).*miniTLabel(Ih);
                % oversampling
                if TF.C2 ==1
                    gradw_part(miniTLabel(Ih)>0) =  gradw_part(miniTLabel(Ih)>0)*TF.C2_1;
                    gradw_part(miniTLabel(Ih)<0) =  gradw_part(miniTLabel(Ih)<0)*TF.C2_2;
                end
                grad_w = (TF.C*w(1:end-1) - Prox.grad_wp - 2*TF.C1*zKTInst(Ih,:)'*(gradw_part) ) ;
                grad_b = (TF.C*w(end) - Prox.grad_bp - 2*TF.C1*sum(gradw_part)); 
 
                grad_final = [grad_w;grad_b];               
                direct = grad_final;
         %% Step size
               if Opt.N == 0
                   % Hypergradient
                   H          = Hyper_grad' * grad_final/(Hyper_grad' * Hyper_grad * grad_final' * grad_final+eps)^(1/2);
                   eta        = eta + Opt.beta * H;
                   Hyper_grad = direct;
                   Report.eta(round,part) = eta;                   
                   w = w - eta.* direct; 
               elseif Opt.N == 1
                   w = w - eta * direct;
                   Report.eta(round,part) = eta;   
               end 
        
        %% Overlapping Strategty
        if  part>1 && Set.Overlap > 1
            % overlap misclassified instances
            loss = -miniTLabel.*(zKTInst*w(1:nDim)+w(end))+1;
            zKTInst_pre = zKTInst(loss>0,:);
            miniTLabel_pre = miniTLabel(loss>0,:);
            
            
            % count misclassified times
            counter_vector(loss>0) = counter_vector(loss>0) + 1;
            counter_vector_pre = counter_vector(loss>0);
            
            % misclassified for too many times then discard
            if isempty(counter_vector_pre > Overlapping_times)==0
                ind_keep = find(counter_vector_pre < Set.Overlap);
                counter_vector_pre = counter_vector_pre(ind_keep,1);
                zKTInst_pre = zKTInst_pre(ind_keep,:);
                miniTLabel_pre = miniTLabel(ind_keep,:);
            end
        end
        end  
    end %end of a minibatch 
    elapsedtime = toc(time);
    Report.time(round,1) = elapsedtime;
    Model.W(:,round) = w;
end % end of a sigle epoch
  Model.wp = [Prox.wp ; Prox.bp];      
        
end % end of function




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
%==========================================================================
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

