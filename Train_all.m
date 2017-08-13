function [Result,Model] = Train_all(filename,method,TF,Opt,Set,SizeoRatiofReducedset,gamma)
%%
%
% method 0->SGD , 1->SN , 2->Adadelta , 3->Adam
%% Preprocessing 
% load training and testing dataset
load(['dataset/',filename,'.mat'],'TInst','TLabel');
load(['dataset/',filename,'.mat'],'VInst','VLabel');
[InstNum,~] = size(TInst);

 
%% Initial Setting
%% Reduced set selection
if SizeoRatiofReducedset <= 1
    SizeofReducedset = round(InstNum*SizeoRatiofReducedset);
else
    SizeofReducedset = SizeoRatiofReducedset;
end
%RS = TInst(1:SizeofReducedset,:);
l_P = find(TLabel>0);
l_N = find(TLabel<0);
RS_P = TInst(l_P,:);
RS_N = TInst(l_N,:);
%RS_l = floor(SizeofReducedset*(length(l_N)/length(TLabel)));
RS_l = floor(SizeofReducedset/2);
if RS_l>length(l_N)
    Model.RS = [RS_N;RS_P(1:SizeofReducedset-length(l_N),:)];
elseif RS_l>length(l_P)
    Model.RS = [RS_N(1:SizeofReducedset-length(l_P),:);RS_P];
else
    Model.RS = [RS_N(1:RS_l,:);RS_P(1:SizeofReducedset-RS_l,:)];
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


Model.gamma = gamma;
Model.W = rand(SizeofReducedset+1,Set.Epoch);


%% Training
if method == 0
    %[Model,Result.train] = RSVM_SGD_PHS(Model,TF,Opt,Set,TInst,TLabel);
     [Model,Result.train] = RSVM_SGD_PHS_v2(Model,TF,Opt,Set,TInst,TLabel);
elseif method == 1 
    %[Model,Result.train] = RSVM_SN_PHS(Model,TF,Opt,Set,TInst,TLabel);
    [Model,Result.train] = RSVM_SN_PHS_v2(Model,TF,Opt,Set,TInst,TLabel);
elseif method == 2
    %[Model,Result.train] = RSVM_Adadelta_PHS(Model,TF,Opt,Set,TInst,TLabel);
    [Model,Result.train] = RSVM_Adadelta_PHS_v2(Model,TF,Opt,Set,TInst,TLabel);
elseif method == 3
    %[Model,Result.train] = RSVM_Adam_PHS(Model,TF,Opt,Set,TInst,TLabel);
    [Model,Result.train] = RSVM_Adam_PHS_v2(Model,TF,Opt,Set,TInst,TLabel);    
end

%% Testing
[Result.test] = Prediction(Model,VInst,VLabel);
