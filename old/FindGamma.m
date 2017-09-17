clc
clear all


%
filename = 'svmguide1';
%
method = 0;
%% Set
Set.Minibatch = 1;   %BatchSize
Set.Epoch     = -30;   %Epoch
Set.Overlap   = 1 ;   %Overlap

%% Trade-Off
%C = 5;
TF.C  = 0.01;
TF.C1 = 100;       %TrainLoss
TF.C2 = 0;      %Syn
TF.C2_1 = 0.2;
TF.C3 = 0;      %Prox

%% Opt
Opt.eta  = 0.0001;      %LearningRate
Opt.beta = 0.0001;         %Hyper 
Opt.N = 3; 
%% Momentum
Opt.mmt.mu = 0.9;
%% Nesterov
Opt.Nmmt.mu = 0.9;
%% Adadelta
Opt.ada.delta = 0.95;
Opt.ada.e     = 1e-6;
%% Adam
Opt.adam.d1 = 0.9;
Opt.adam.d2 = 0.999;
Opt.adam.e = 1e-8;
%% PSA
Opt.psa.eta  = 0.0001;
Opt.psa.b    = 10;
Opt.psa.alpha= 0.999;
Opt.psa.beta = 0.99;
Opt.psa.kai  = 0.9;
%gamma = 0.00001;
%gamma = 45;
gamma = 0.000625;
%Reduce kernel subset size
SizeoRatiofReducedset = 0.1;

%profile on
[Result,Model] = Train_all(filename,method,TF,Opt,Set,SizeoRatiofReducedset,gamma)
%profile viewer
Result.train.time
Result.test
%Kernelprint(Model.RS,Model.RS,gamma);
%
%}
%{
RatiofRS = 0.003;
%gamma = 0.355;
gamma = 0.00625;
KSGamma(filename,RatiofRS,gamma)
%}