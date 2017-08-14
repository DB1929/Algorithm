function [Result,Model] = Train_grid_search(Filename,Method,TF,Opt,Set,RatiofRS,gamma)
%% 
% ========================================================================%
% ! Input the sequence of parameter to search the best combination  !
% 
% Read Me ================================================================%
% Inputs :
%        (1) Filename     : The file of Training data & Testing data
%
%        (2) Method       : The Opt method.
%                           0->SGD      : Stochastic Gradient Descent
%                           1->SGDM     : SGD with Momentum
%
%                           3->SN       : Stochastic Newton Method
%                           4->Adadelta : Adaptive method
%                           5->Adam     : Adaptive method
%                           6->PSA      : Periodic Step-Size Adaptation
%
%        (3) TF           : Trade-off in objective function
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
%            Opt."method".: The parameter of each method (See [%%Training])
%
%        (4) Set          : Setting the learning.
%            Set.Minibatch: Training data size in every iteration
%            Set.Epoch    : Number of epoch
%                           [+] -> Previous model will be next initial
%                           [-] -> Each model are independent
%            Set.Overlap  : Overlapping times
%
%        (6) RatiofRS     : Size ratio between training data and reduce set
%
%        (7) gamma        : RBF kernel parameter.
% Outputs :
%        (1) Result
%            Result.train  : Result in training phrase
%             - Result.train.time   : time comsuption for each epoch.
%             - Result.train.loss   : Training loss for every iteration
%             - Result.train.eta    : Learning rate in every iteration
%
%            Result.test   : Result in testing phrase
%             - Result.test.PVLabel : Prediction of each model in testing data
%             - Result.test.Testing_error
%             - Result.test.[min\max\avg\std]_testing_err
%             - Result.test.ModelRelate : Model relative for each models
%
%        (2) Model        : Initial of the Model
%            Model.W      : classifiers (w,b)   (rs1+1 x RoundNum) 
%            Model.RS     : Reduce set of Kernel
%            Model.gamma  : Parameter of RBF Kernel.
%=========================================================================%
%% Preprocessing 
% load training and testing dataset
load(['dataset/',Filename,'.mat'],'TInst','TLabel');
load(['dataset/',Filename,'.mat'],'VInst','VLabel');
[InstNum,~] = size(TInst);

 
%% Initial Setting
%% Reduced set selection
if RatiofRS <= 1
    SizeofRS = round(InstNum*RatiofRS);
else
    SizeofRS = RatiofRS;
end
%RS = TInst(1:SizeofReducedset,:);
l_P = find(TLabel>0);
l_N = find(TLabel<0);
RS_P = TInst(l_P,:);
RS_N = TInst(l_N,:);
%RS_l = floor(SizeofReducedset*(length(l_N)/length(TLabel)));
RS_l = floor(SizeofRS/2);
if RS_l>length(l_N)
    Model.RS = [RS_N;RS_P(1:SizeofRS-length(l_N),:)];
elseif RS_l>length(l_P)
    Model.RS = [RS_N(1:SizeofRS-length(l_P),:);RS_P];
else
    Model.RS = [RS_N(1:RS_l,:);RS_P(1:SizeofRS-RS_l,:)];
end


%% Checkerboard RS
%%
%{
%Model.RS=[-1,-1; -1,0; -1,1; 0,-1; 0,0; 0,1; 1,-1; 1,0; 1,1];
%Model.RS =[-2,-2; -2,-1; -2,0; -2,1; -2,2; -1,-2; -1,-1; -1,0; -1,1; -1,2; 0,-2; 0,-1; 0,0; 0,1; 0,2; 1,-2; 1,-1; 1,0; 1,1; 1,2; 2,-2; 2,-1; 2,0; 2,1; 2,2];
rs = -2:0.5:2;
rs2 = rs'*ones(1,length(rs));
rs=rs';
Model.RS = [rs2(1,:)',rs; rs2(2,:)',rs;rs2(3,:)',rs;rs2(4,:)',rs;rs2(5,:)',rs;rs2(6,:)',rs;rs2(7,:)',rs;rs2(8,:)',rs;rs2(9,:)',rs];
SizeofReducedset = length(Model.RS);
%}





%Epoch Setting
if Set.Epoch > 0
    TrainTimes = 1;
elseif Set.Epoch < 0
    TrainTimes = abs(Set.Epoch);
    Set.Epoch = 1;
    WW = zeros(SizeofRS+1,TrainTimes);
end
%% Loop initial

count = 1;
CC    = TF.C;
CC1   = TF.C1;
CC2   = TF.C2;
CC2_1 = TF.C2_1;
CC3   = TF.C3;
ETA   = Opt.eta;
BETA  = Opt.beta;
for ii_1 = gamma
    Model.gamma = ii_1;
    for ii_2 = CC
        TF.C = ii_2;
        for ii_3 = CC1
            TF.C1 = ii_3;
            for ii_4 = CC2
                TF.C2 = ii_4;
                for ii_5 = CC2_1
                    TF.C2_1 = ii_5;
                    for ii_6 = CC3
                        TF.C3 = ii_6;
                        for ii_7 = ETA
                            Opt.eta =ii_7;
                            for ii_8 = BETA
                                Opt.beta = ii_8;
    %% Training
    for tt = 1:TrainTimes
        Model.W = rand(SizeofRS+1,Set.Epoch);
        if Method == 0
            %[Model,Result.train] = RSVM_SGD_PHS(Model,TF,Opt,Set,TInst,TLabel);
            [Model,Result.train] = RSVM_SGD_PHS_v2(Model,TF,Opt,Set,TInst,TLabel);
            method = 'SGD';
        elseif Method == 1
           %% Opt.mmt
            % Opt.mmt.mu   : Parameter of momentum
            [Model,Result.train] = RSVM_SGDM_PHS_v2(Model,TF,Opt,Set,TInst,TLabel);
            method = 'SGDM';
        elseif Method == 2
           %% Opt.Nmmt
            % Opt.Nmmt.mu  : Parameter of Nesterov momentum
            [Model,Result.train] = RSVM_SGDNvM_PHS_v2(Model,TF,Opt,Set,TInst,TLabel);
            method = 'SGDNvM';
        elseif Method == 3 
            %[Model,Result.train] = RSVM_SN_PHS(Model,TF,Opt,Set,TInst,TLabel);
            [Model,Result.train] = RSVM_SN_PHS_v2(Model,TF,Opt,Set,TInst,TLabel);
            method = 'SN';
        elseif Method == 4
           %% Opt.ada
            % Opt.ada.delta: Decay rate in Adadelta algorithm
            % Opt.ada.e    : Adadelta parameter in RMS function
            [Model,Result.train] = RSVM_Adadelta_PHS_v2(Model,TF,Opt,Set,TInst,TLabel);
            method = 'Adadelta';
        elseif Method == 5
           %% Opt.adam
            % Opt.adam.d1  : Decay rate in Adam 
            % Opt.adam.d2  : Decay rate in Adam 
            % Opt.adam.e   : Adam parameter in RMS function
            [Model,Result.train] = RSVM_Adam_PHS_v2(Model,TF,Opt,Set,TInst,TLabel); 
            method = 'Adam';
        elseif Method == 6
           %% Opt.psa
            % Opt.psa.eta  : Initial of PSA vector
            % Opt.psa.b    : Step size of update PSA vector 
            % Opt.psa.alpha: PSA bounded parameter
            % Opt.psa.beta : PSA bounded parameter
            % Opt.psa.kai  : PSA bounded parameter
            [Model,Result.train] = RSVM_PSA_PHS_v2(Model,TF,Opt,Set,TInst,TLabel);
            method = 'PSA';
        end
        if TrainTimes > 1
            WW(:,tt) = Model.W; 
            if tt == TrainTimes
                Model.W = WW;
            end
        end
    end
    
    %% Testing
    [Result.test] = Prediction(Model,VInst,VLabel);
    if count == 1
        fileID = fopen([method,'_',Filename,'_report','.txt'],'w');
    end    
    %% Report
    fprintf(fileID,'%s%d\n','#',count);
    fprintf(fileID,'%s\n','#### Setting ##############');
    fprintf(fileID,'%15s%d\n','Minibatch = ',Set.Minibatch);
    fprintf(fileID,'%15s%d\n','Epoch = ',Set.Epoch);
    fprintf(fileID,'%15s%d\n','Kernel size = ',length(Model.RS(:,1)));
    fprintf(fileID,'%s\n','#### Trade-off ############');
    fprintf(fileID,'%15s%f\n','Regular = ',TF.C);
    fprintf(fileID,'%15s%f\n','Empricial = ',TF.C1);
    fprintf(fileID,'%15s%f\n','Proximal = ',TF.C3);
    fprintf(fileID,'%15s%f\n','Synthestic = ',TF.C2);
    fprintf(fileID,'%15s%f\n','gamma = ',Model.gamma);
    fprintf(fileID,'%s\n','#### Opt ##################');
    fprintf(fileID,'%18s%d\n','Step Type = ',Opt.N);
    fprintf(fileID,'%18s%f\n','Learning rate = ',Opt.eta);
    fprintf(fileID,'%18s%f\n','Hypergradient = ',Opt.beta);
    fprintf(fileID,'%s\n','#### Result ###############');
    fprintf(fileID,'%s\n','Training Time = ');
    fprintf(fileID,'%9f\b%9f\b%9f\b\n',Result.train.time);
    fprintf(fileID,'\n%s\n','xxx');
    fprintf(fileID,'%s\n','Testing error = ');
    fprintf(fileID,'%10f\b%10f\n',Result.test.Testing_error);
    fprintf(fileID,'\n%s\n','xxx');
    fprintf(fileID,'%s%f\n','min_testing_err = ',Result.test.min_testing_err);
    fprintf(fileID,'%s%f\n','max_testing_err = ',Result.test.max_testing_err);
    fprintf(fileID,'%s%f\n','avg_testing_err = ',Result.test.avg_testing_err);
    fprintf(fileID,'%s%f\n','std_testing_err = ',Result.test.std_testing_err);
    fprintf(fileID,'%s\n','Model Relate = ');
    fprintf(fileID,'%9f\b%9f\b%9f\b\n',Result.test.ModelRelate);
    fprintf(fileID,'%s\n','#### end ##################');
    fprintf(fileID,'%s%d\n','#',count);
    fprintf(fileID,'%s\n','xxxxxxxxxxxxxxxxxxxxxxxxx');
    count = count+1;
    %clear Result;

                            end
                        end
                    end
                end
            end
        end
    end
end
fclose(fileID);
end