clc
clear all


%
filename = 'usps01';
%{
method = 6;
%% Set
Set.Minibatch = 100;   %BatchSize
Set.Epoch     = 5;   %Epoch
Set.Overlap   = 1 ;   %Overlap

%% Trade-Off
%C = 5;
TF.C  = 0.1;
TF.C1 = 100;       %TrainLoss
TF.C2 = 0.01;      %Syn
TF.C2_1 = 0.2;
TF.C3 = 0.01;      %Prox

%% Opt
Opt.eta  = 0.001;      %LearningRate
Opt.beta = 0.005;         %Hyper 
Opt.N = 3; 
%% Adam
Opt.adam.d1 = 0.9;
Opt.adam.d2 = 0.999;
Opt.adam.e = 1e-8;
%% PSA
Opt.psa.eta  = 0.001;
Opt.psa.b    = 1;
Opt.psa.alpha= 0.999;
Opt.psa.beta = 0.99;
Opt.psa.kai  = 0.9;
%gamma = 0.00001;
%gamma = 45;
gamma = 0.734085;
%Reduce kernel subset size
SizeoRatiofReducedset = 0.1;

profile on
[Result,Model] = Train_all(filename,method,TF,Opt,Set,SizeoRatiofReducedset,gamma)
profile viewer
Kernelprint(Model.RS,Model.RS,gamma);
%
%}
%{
RatiofRS = 0.003;
%gamma = 0.355;
gamma = 0.00625;
KSGamma(filename,RatiofRS,gamma)
%}