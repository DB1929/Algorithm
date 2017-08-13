%%Numerical test
clc
clear all


%Data set : svmguide1
%{
filename = 'svmguide1';

%% Set
Set.Minibatch = 10;   %BatchSize
Set.Epoch     = 10;   %Epoch
Set.Overlap   = 1 ;   %Overlap

%% Trade-Off
%C = 5;
TF.C  = 0.1;
TF.C1 = 100;       %TrainLoss
TF.C2 = 0.1;      %Syn
TF.C3 = 10;      %Prox

%% Opt
Opt.eta  = 0.0003;      %LearningRate
Opt.beta = 0.0001;         %Hyper 
Opt.N = 3; 
%gamma = 0.00001;
gamma = 1e-3;

%Reduce kernel subset size
SizeoRatiofReducedset = 0.05;

profile on
[Result,Model] = Train_SGD(filename,TF,Opt,Set,SizeoRatiofReducedset,gamma)
profile viewer
Kernelprint(Model.RS,Model.RS,gamma);
%}

%Data set : w3a 
%{
filename = 'w3a';

%% Set
Set.Minibatch = 1500;   %BatchSize
Set.Epoch     = 10;   %Epoch
Set.Overlap   = 1 ;   %Overlap

%% Trade-Off
%C = 5;
TF.C  = 0.001;
TF.C1 = 10;       %TrainLoss
TF.C2 = 0;      %Syn
TF.C3 = 0.01;      %Prox

%% Opt
Opt.eta  = 0.003;      %LearningRate
Opt.beta = 0.0001;         %Hyper 

%gamma = 0.00001;
%gamma = 0.09;
gamma = 0.075;
%Reduce kernel subset size
SizeoRatiofReducedset = 0.0125;

profile on
[Result_training,Result_testing,Model] = Train_SGD(filename,TF,Opt,Set,SizeoRatiofReducedset,gamma)
profile viewer
Kernelprint(Model.RS,Model.RS,gamma);
%}

%Data set : Checkerboard
% With special RS is ok
%{
filename = 'Checkerboard';

%% Set
Set.Minibatch = 5000;   %BatchSize
Set.Epoch     = 5;   %Epoch
Set.Overlap   = 1 ;   %Overlap

%% Trade-Off
%C = 5;
TF.C  = 0.001;
TF.C1 = 0.1;       %TrainLoss
TF.C2 = 0;      %Syn
TF.C3 = 0.005;      %Prox

%% Opt
Opt.eta  = 0.9;      %LearningRate
Opt.beta = 0.9;         %Hyper 

%gamma = 0.00001;
%gamma = 15;
gamma = 5;
%Reduce kernel subset size
SizeoRatiofReducedset = 0.001;

profile on
[Result,Model] = Train_SGD(filename,TF,Opt,Set,SizeoRatiofReducedset,gamma)
profile viewer
Kernelprint(Model.RS,Model.RS,gamma);
%}






%Data set : svmguide1 -SN
%{
filename = 'svmguide1';

%% Set
Set.Minibatch = 100;   %BatchSize
Set.Epoch     = 10;   %Epoch
Set.Overlap   = 1 ;   %Overlap

%% Trade-Off
%C = 5;
TF.C  = 1;
TF.C1 = 100;       %TrainLoss
TF.C2 = 0;      %Syn
TF.C3 = 0.1;      %Prox

%% Opt
Opt.eta  = 0.03;      %LearningRate
Opt.beta = 0.001;         %Hyper 
Opt.N = 1;
%gamma = 0.00001;
gamma = 1e-3;

%Reduce kernel subset size
SizeoRatiofReducedset = 0.05;

profile on
[Result,Model] = Train_SN(filename,TF,Opt,Set,SizeoRatiofReducedset,gamma)
profile viewer
Kernelprint(Model.RS,Model.RS,gamma);
%}

%Data set : Checkerboard -SN
% With special RS is ok
%{
filename = 'Checkerboard';

%% Set
Set.Minibatch = 1000;   %BatchSize
Set.Epoch     = 5;   %Epoch
Set.Overlap   = 1 ;   %Overlap

%% Trade-Off
%C = 5;
TF.C  = 0.001;
TF.C1 = 0.1;       %TrainLoss
TF.C2 = 0;      %Syn
TF.C3 = 0.005;      %Prox

%% Opt
Opt.eta  = 0.9;      %LearningRate
Opt.beta = 0.9;         %Hyper 
Opt.N = 1;

%gamma = 0.00001;
%gamma = 15;
gamma = 5;
%Reduce kernel subset size
SizeoRatiofReducedset = 0.001;

profile on
[Result_training,Result_testing,Model] = Train_SGD(filename,TF,Opt,Set,SizeoRatiofReducedset,gamma)
profile viewer
Kernelprint(Model.RS,Model.RS,gamma);
%}

%Data set : a9a -SN
%{
filename = 'a9a';

%% Set
Set.Minibatch = 100;   %BatchSize
Set.Epoch     = 10;   %Epoch
Set.Overlap   = 1 ;   %Overlap

%% Trade-Off
%C = 5;
TF.C  = 10;
TF.C1 = 100;       %TrainLoss
TF.C2 = 0;      %Syn
TF.C3 = 0.1;      %Prox

%% Opt
Opt.eta  = 0.03;      %LearningRate
Opt.beta = 0.001;         %Hyper 
Opt.N = 1;
%gamma = 0.00001;
gamma = 0.05;

%Reduce kernel subset size
SizeoRatiofReducedset = 0.005;

profile on
[Result,Model] = Train_SN(filename,TF,Opt,Set,SizeoRatiofReducedset,gamma)
profile viewer
Kernelprint(Model.RS,Model.RS,gamma);
%}


%Data set : svmguide1 -SN -script1
%{
filename = 'svmguide1';
method   = 'RSVM_SN';
%% Set
Set.Minibatch = 100;   %BatchSize
Set.Epoch     = 5;   %Epoch
Set.Overlap   = 1 ;   %Overlap

Type = 1;
%gamma = 0.00001;
gamma = 1e-3;

%Reduce kernel subset size
SizeoRatiofReducedset = 0.05;

script1(filename,method,SizeoRatiofReducedset,gamma,Set,Type)
%}

%Data set : a9a -SN -script
%{
filename = 'a9a';
method   = 'RSVM_SN';
%% Set
Set.Minibatch = 100;   %BatchSize
Set.Epoch     = 5;   %Epoch
Set.Overlap   = 1 ;   %Overlap

Type = 1;
%gamma = 0.00001;
gamma = 0.05;

%Reduce kernel subset size
SizeoRatiofReducedset = 0.005;

script1(filename,method,SizeoRatiofReducedset,gamma,Set,Type)
%}


%Data set : w3a -SN -script
%
filename = 'w3a';
method   = 'RSVM_SN';
%% Set
Set.Minibatch = 100;   %BatchSize
Set.Epoch     = 5;   %Epoch
Set.Overlap   = 1 ;   %Overlap

Type = 1;
%gamma = 0.00001;
gamma = 0.075;

%Reduce kernel subset size
SizeoRatiofReducedset = 0.0125;

script1(filename,method,SizeoRatiofReducedset,gamma,Set,Type)
%}