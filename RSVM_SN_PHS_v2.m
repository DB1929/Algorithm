function [Model,Report] = RSVM_SN_PHS_v2(Model,TF,Opt,Set,Inst,Label)
%% ========================================================================
% JIAN-PING SYU
% Stochastic Newton with (1)Proximal Model (2)Hypergradient (3)Synthetic Data
%           Reduce Kernel Support Vector Machine
%           ! v2 Renew the Synthetic Data 
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
%            TF.C2        : Trade-off with Synthetic data loss
%            TF.C2_1      : Trade-off  in  Synthetic data 
%            TF.C3        : Trade-off with Proximal Model
%
%        (3) Opt          : Parameter of optimization algorithm
%            Opt.eta      : Learning rate
%            Opt.beta     : Parameter of Hypergradient 
%            Opt.N        : Type of Learning rate choose
%                           0-> Default step 1
%                           1-> Armijo 
%                           2-> Hypergradient 
%                           3-> Fixed eta
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
Stat_Info = zeros(4,nDim);        % proximal model with original data 
n_p = 0;
n_n = 0;

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
    if round == 1
       Hyper_grad = zeros(rs1,1);
    else
       w = Model.W(:,round-1); 
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
            [Stat_Info,n_p,n_n] = Stat(zKTInst,miniTLabel,Stat_Info,n_p,n_n);
           %{
            ind_p = find(miniTLabel>0);
            ind_n = find(miniTLabel<0);
            KT_p = zKTInst(ind_p,:);
            KT_n = zKTInst(ind_n,:);
            Stat_Info(1,:) = Stat_Info(1,:) + sum(KT_p,1);
            Stat_Info(2,:) = Stat_Info(2,:) + sum(KT_n,1);
            Stat_Info(3,:) = Stat_Info(3,:) + sum(KT_p.^2,1);
            Stat_Info(4,:) = Stat_Info(4,:) + sum(KT_n.^2,1);
            
            n_p = n_p + size(ind_p,1);
            n_n = n_n + size(ind_n,1);
            %}
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
                if n_p>1 && n_n>1
                    % Means
                    m_p = Stat_Info(1,:)/n_p;
                    m_n = Stat_Info(2,:)/n_n;
                    % Variance
                    s_p = (Stat_Info(3,:)-(((m_p.^2)*n_p)))/(n_p-1);
                    s_n = (Stat_Info(4,:)-(((m_n.^2)*n_n)))/(n_n-1);
                    % Weight & Bias
                    wp   = (((m_p-m_n)./(s_p + s_n + eps*ones(1,size(s_p',1)))))';
                    bp   = -0.5*(m_p+m_n)*wp;
                    % Gradient of Proximal Model
                    grad_prox_w = TF.C3*wp;
                    grad_prox_b = TF.C3*bp;

                else
                    % Gradient of Proximal Model
                    m_p = zeros(1,nDim);
                    m_n = zeros(1,nDim);
                    s_p = zeros(1,nDim);
                    s_n = zeros(1,nDim);
                    wp  = zeros(1,nDim)';
                    bp  = 0;
                    grad_prox_w = zeros(nDim,1);
                    grad_prox_b = 0;
                end
            end
            
        %% Synthestic data
        if TF.C2>0
                % Divide the Positive & Negative data
                Syn_ind_p = find(miniTLabel>0);
                Syn_ind_n = find(miniTLabel<0);
                Syn_l_p   = length(Syn_ind_p);
                Syn_l_n   = length(Syn_ind_n);
                % Generate the Synthestic data
                C4 = TF.C2_1;
                if Syn_l_p>0 && Syn_l_n>0
                    %Syndata_P = (1-C4)*zKTInst(Syn_ind_p,:)+ones(Syn_l_p,1)*(C4*m_p);
                    %Syndata_N = (1-C4)*zKTInst(Syn_ind_n,:)+ones(Syn_l_n,1)*(C4*m_n);
                    SynData  = [(1-C4)*zKTInst(Syn_ind_p,:)+ones(Syn_l_p,1)*(C4*m_p);(1-C4)*zKTInst(Syn_ind_n,:)+ones(Syn_l_n,1)*(C4*m_n)];
                    SynLabel = [ones(Syn_l_p,1);-ones(Syn_l_n,1)];
                    loss_syn = 1 - SynLabel.*(SynData*w(1:nDim)+w(end));
                elseif Syn_l_p>0 && Syn_l_n == 0
                    SynData  = (1-C4)*zKTInst(Syn_ind_p,:)+ones(Syn_l_p,1)*(C4*m_p);
                    SynLabel = ones(Syn_l_p,1);
                    loss_syn = 1 - SynLabel.*(SynData*w(1:nDim)+w(end));
                elseif Syn_l_p == 0 && Syn_l_n>0
                    SynData  = (1-C4)*zKTInst(Syn_ind_n,:)+ones(Syn_l_n,1)*(C4*m_n);
                    SynLabel = -ones(Syn_l_n,1);
                    loss_syn = 1 - SynLabel.*(SynData*w(1:nDim)+w(end));
                else
                    SynData  = [];
                    SynLabel = [];
                    loss_syn = 0;
                end
                % Passive
                Syn_Ih = find(loss_syn > 0);
                Syn_nIh  = length(Syn_Ih);
                % Gradient of Synthestic data
                if Syn_nIh == 0
                 grad_syn_w = 0;
                 grad_syn_b = 0;
                else
                 grad_syn_w = 2*TF.C2*SynData(Syn_Ih,:)'*loss_syn(Syn_Ih);
                 grad_syn_b = 2*TF.C2*sum(loss(Syn_Ih)); 
                end
        else
            grad_syn_w = 0;
            grad_syn_b = 0;
        end
                
         %% Stochastic Newton Method update                           
               % Final gradient
                %gradw_part = loss(Ih).*miniTLabel(Ih)/nIh;
                gradw_part = loss(Ih).*miniTLabel(Ih);
                grad_w = (TF.C*w(1:end-1) - grad_prox_w - grad_syn_w - 2*TF.C1*zKTInst(Ih,:)'*(gradw_part) ) ;
                grad_b = (TF.C*w(end) - grad_prox_b - grad_syn_b - 2*TF.C1*sum(gradw_part)); 
 
                grad_final = [grad_w;grad_b];

               % Hessian matrix
               % w = w - eta * (g/C1 - 2/(C1^2) * Q' * (I+2QQ')^(-1) * Q * g))
               % Q:(batch size + synthestic data size) X (feature dimension + 1)
               
                if TF.C2>0 && ~isempty(SynData)
                   Q = [zKTInst(Ih,:)*(TF.C1),ones(nIh,1)*(TF.C1);SynData(Syn_Ih,:)*(TF.C2),ones(Syn_nIh,1)*(TF.C2)]; 
                   Nt_direct = grad_final/TF.C - 2*(1/(TF.C))*Q'*((eye(nIh+Syn_nIh)+2*Q*Q')\(Q*grad_final));
                else
                   %Q = [zKTInst(Ih,:)*(TF.C1/nIh),ones(nIh,1)*(TF.C1/nIh)];
                   %Nt_direct = grad_final/TF.C - 2*(1/(TF.C))*Q'*((eye(nIh)+2*Q*Q')\(Q*grad_final));
                   Q = [zKTInst(Ih,:),ones(nIh,1)];                    
                   Nt_direct = grad_final/TF.C - 2*TF.C1*(1/(TF.C))*Q'*((eye(nIh)+2*TF.C1*Q*Q')\(Q*grad_final));
                end
               %% Step size
               if Opt.N == 0
                  % Newton Step (stepsize == 1)
                   w = w - Nt_direct;
                   Report.eta(round,part) = 1; 

               elseif Opt.N == 1
                    %Optional : Armijo condition (for stepsize)
                    if  (grad_final'*grad_final/nDim > 1E-5)
                        if TF.C2>0
                            eta = Armijo(zKTInst,miniTLabel,SynData,SynLabel,w,[wp;bp],TF,Nt_direct,grad_final);
                        else
                            eta = Armijo(zKTInst,miniTLabel,0,0,w,[wp;bp],TF,Nt_direct,grad_final);
                        end
                        w = w - eta * Nt_direct;
                        Report.eta(round,part) = eta;   
                    end
               elseif Opt.N == 2
                   % Hypergradient
                   H          = Hyper_grad' * grad_final/(Hyper_grad' * Hyper_grad * grad_final' * grad_final+eps)^(1/2); 
                   eta        = eta + Opt.beta * H;
                   Hyper_grad = Nt_direct;
                   Report.eta(round,part) = eta;                   
                   w = w - eta.*Nt_direct; 
               elseif Opt.N == 3
                   w = w - eta  *Nt_direct;
                   Report.eta(round,part) = eta;   
               end % Stepsize end           
              
        end %Update Part end
        
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
        
    end %end of a minibatch 
    elapsedtime = toc(time);
    Report.time(round,1) = elapsedtime;
    Model.W(:,round) = w;
end % end of a sigle epoch
        
        
end % end of function
    









function [Stat_Info,n_p,n_n] = Stat(zKTInst,miniTLabel,Stat_Info,n_p,n_n)
%% ========================================================================
% Statistical information
% Outputs :
%        (1) Stat_Info    
%            Stat_Info(1,:) : Mean of Postive  Data X Number of Postive 
%            Stat_Info(1,:) : Mean of Negative Data X Number of Postive
%            Stat_Info(1,:) : Mean of Postive  Data Square X Number of Postive  
%            Stat_Info(1,:) : Mean of Postive  Data Square X Number of Postive 
%
%        (2) 
%            n_p            : Number of Postive  Data
%            n_n            : Number of Negative Data
%=========================================================================%
%%
    
    ind_p = find(miniTLabel>0);
    ind_n = find(miniTLabel<0);
    KT_p = zKTInst(ind_p,:);
    KT_n = zKTInst(ind_n,:);
    Stat_Info(1,:) = Stat_Info(1,:) + sum(KT_p,1);
    Stat_Info(2,:) = Stat_Info(2,:) + sum(KT_n,1);
    Stat_Info(3,:) = Stat_Info(3,:) + sum(KT_p.^2,1);
    Stat_Info(4,:) = Stat_Info(4,:) + sum(KT_n.^2,1);

    n_p = n_p + size(ind_p,1);
    n_n = n_n + size(ind_n,1);
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

%{
function stepsize = armijo(Tdata,Tlabel,Tdata_syn,Tlabel_syn,w,wp,TF,direct,gap,obj1)
%% ========================================================================
% Armijo Stepsize
%
% Inputs
%   w1, b1: Current model
%   C     : Weight parameter
%   gap   : Defined in ssvm code
%   obj1  : The object function value of current model
%   diff  : The difference of objective function values between current
%           and next model
%==========================================================================
%%
diff=0;
temp=0.5; % we start to test with setpsize=0.5
count = 1;
while diff  < -0.05*temp*gap
    temp = 0.5*temp;
    w2 = w - temp*direct;
    obj2 = objf(Tdata,Tlabel,Tdata_syn,Tlabel_syn,w2,wp,TF);
    diff = obj1 - obj2;
    count = count+1;
    if (count>20)
        break;
    end
end;

stepsize = temp;
end
%}

function stepsize = Armijo(Tdata,Tlabel,Tdata_syn,Tlabel_syn,w,wp,TF,direct,grad)
%% ========================================================================
% Armijo Stepsize
%
% Inputs
%   w1, b1: Current model
%   C     : Weight parameter
%   gap   : Defined in ssvm code
%   obj1  : The object function value of current model
%   diff  : The difference of objective function values between current
%           and next model
%==========================================================================
%%
% Check the First Order Opt. condition
% stepsize = 1; % The default stepsize is 1
w2 = w - direct;
obj1 = objf(Tdata,Tlabel,Tdata_syn,Tlabel_syn,w,wp,TF);
obj2 = objf(Tdata,Tlabel,Tdata_syn,Tlabel_syn,w2,wp,TF);
   if (obj1 - obj2) <= 1E-8
        % Use the Armijo's rule
        gap = direct'*grad; % Compute the gap
        % Find the step size & Update to the new point
        diff=obj1 - obj2;
        temp=0.5; % we start to test with setpsize=0.5
        count = 1;
        while diff  < -0.05*temp*gap
            temp = 0.5*temp;
            w2 = w - temp*direct;
            obj2 = objf(Tdata,Tlabel,Tdata_syn,Tlabel_syn,w2,wp,TF);
            diff = obj1 - obj2;
            count = count+1;
            if (count>30)
                break;
            end
        end;
        stepsize = temp; 
    else
        stepsize = 1;
    end

end



function value = objf(Tdata,Tlabel,Tdata_syn,Tlabel_syn,w,wp,TF)
%% ========================================================================
% Evaluate the objective function value
%==========================================================================
temp     = 1 - Tlabel.*(Tdata*w(1:end-1,1)+w(end));
v   = max(temp,0);
if TF.C2>0
    temp_syn = 1 - Tlabel_syn.*(Tdata_syn*w(1:end-1,1)+w(end));
    v_s = max(temp_syn,0);
else
    v_s = 0;
end
    value = 0.5*TF.C*(w'*w) + TF.C1*(v'*v)/length(v) - TF.C3*w'*wp + TF.C2*(v_s'*v_s)/length(v_s);
end


