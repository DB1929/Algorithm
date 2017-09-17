function [Result,Model] = Train_YellowFin(Filename,TF,Opt,Set,RatiofRS,gamma)
%%
% 
% Read Me ================================================================%
% Inputs :
%        (1) Filename     : The file of Training data & Testing data
%
%        (3) TF           : Trade-off in objective function
%            TF.C         : Trade-off with Regularization
%            TF.C1        : Trade-off with Training  data loss
%            TF.C3        : Trade-off with Proximal Model
%
%        (3) Opt          : Parameter of optimization algorithm
%            Opt.yf.beta  : Decay rate of YellowFin
%            Opt.yf.width : Width of sliding window of YellowFin
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
idx = randperm(InstNum);
TInst = TInst(idx,:);
TLabel = TLabel(idx);
 
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

%% Fix the reduce set
%% a9a RS
%{
load('a9aRS');
Model.RS = a9aRS;
%}

%% svmguide1 RS
%{
load('svmguide1RS');
Model.RS = svmguide1RS;
%}

%% ijcnn1 RS
%{
load('ijcnn1RS');
Model.RS = ijcnn1RS;
%}

%% Checkerboard RS
%%
%{
%Model.RS=[-1,-1; -1,0; -1,1; 0,-1; 0,0; 0,1; 1,-1; 1,0; 1,1];
%Model.RS =[-2,-2; -2,-1; -2,0; -2,1; -2,2; -1,-2; -1,-1; -1,0; -1,1; -1,2; 0,-2; 0,-1; 0,0; 0,1; 0,2; 1,-2; 1,-1; 1,0; 1,1; 1,2; 2,-2; 2,-1; 2,0; 2,1; 2,2];
rs = -2:0.5:2;
rs2 = rs'*ones(1,length(rs));
rs=rs';
Model.RS = [rs2(1,:)',rs; rs2(2,:)',rs;rs2(3,:)',rs;rs2(4,:)',rs;rs2(5,:)',rs;rs2(6,:)',rs;rs2(7,:)',rs;rs2(8,:)',rs;rs2(9,:)',rs];
SizeofRS = length(Model.RS);
%}





%Epoch Setting
if Set.Epoch > 0
    TrainTimes = 1;
elseif Set.Epoch < 0
    TrainTimes = abs(Set.Epoch);
    Set.Epoch = 1;
    WW = zeros(SizeofRS+1,TrainTimes);
    TotalTime = zeros(1,TrainTimes);
end
Model.gamma = gamma;

%% Training
for tt = 1:TrainTimes
    %Model.W = 0.01*rand(SizeofRS+1,Set.Epoch);
    Model.W = zeros(SizeofRS+1,Set.Epoch);
    %[Model,Result.train] = RSVM_YFin_PHS_v3(Model,TF,Opt,Set,TInst,TLabel);
    [Model,Result.train] = RSVM_YFin_Prox_v4(Model,TF,Opt,Set,TInst,TLabel);
    method = 'YellowFin';
    if TrainTimes > 1
        TotalTime(1,tt) = Result.train.time;
        WW(:,tt) = Model.W;        
        if tt == TrainTimes
            Model.W = WW;
            Result.train.time = TotalTime;
        end
    end
end
%% Testing
[Result.test] = Prediction_v2(Model,VInst,VLabel);
[Result.train2] = Prediction_v2(Model,TInst,TLabel);
%% Print Kernel
%Kernelprint([VInst(VLabel<0,:);VInst(VLabel>0,:)],Model.RS,gamma);