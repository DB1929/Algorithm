%%Numerical test
clc
clear all


%Data set : svmguide1
%{
filename = 'svmguide1';
method = 0;
%% Set
Set.Minibatch = 100;   %BatchSize
Set.Epoch     = 10;   %Epoch
Set.Overlap   = 1 ;   %Overlap

%% Trade-Off
%C = 5;
TF.C  = 1;
TF.C1 = 100;       %TrainLoss
TF.C2 = 0.1;      %Syn
TF.C3 = 0.1;      %Prox

%% Opt
Opt.eta  = 0.0003;      %LearningRate
Opt.beta = 0.0001;         %Hyper 
Opt.N = 2; 
%gamma = 0.00001;
gamma = 1e-3;

%Reduce kernel subset size
SizeoRatiofReducedset = 0.05;

profile on
[Result,Model] = Train_all(filename,method,TF,Opt,Set,SizeoRatiofReducedset,gamma)
profile viewer
Kernelprint(Model.RS,Model.RS,gamma);
%}

%Data set : w3a 
%{
filename = 'w3a';
method = 0;
%% Set
Set.Minibatch = 10;   %BatchSize
Set.Epoch     = 10;   %Epoch
Set.Overlap   = 1 ;   %Overlap

%% Trade-Off
%C = 5;
TF.C  = 1;
TF.C1 = 10;       %TrainLoss
TF.C2 = 0;      %Syn
TF.C3 = 0.1;      %Prox

%% Opt
Opt.eta  = 0.00003;      %LearningRate
Opt.beta = 0.00001;         %Hyper 
Opt.N = 2;
%gamma = 0.00001;
%gamma = 0.09;
gamma = 0.075;
%Reduce kernel subset size
SizeoRatiofReducedset = 0.0625;

profile on
[Result,Model] = Train_all(filename,method,TF,Opt,Set,SizeoRatiofReducedset,gamma)
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
%
filename = 'svmguide1';
method = 1;
%% Set
Set.Minibatch = 10;   %BatchSize
Set.Epoch     = 10;   %Epoch
Set.Overlap   = 1 ;   %Overlap

%% Trade-Off
%C = 5;
TF.C  = 0.1;
TF.C1 = 1000;       %TrainLoss
TF.C2 = 0;      %Syn
TF.C3 = 0.1;      %Prox

%% Opt
Opt.eta  = 0.03;      %LearningRate
Opt.beta = 0.01;         %Hyper 
Opt.N = 2;
%gamma = 0.00001;
gamma = 1e-3;

%Reduce kernel subset size
SizeoRatiofReducedset = 0.05;

profile on
[Result,Model] = Train_all(filename,method,TF,Opt,Set,SizeoRatiofReducedset,gamma)
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
[Result,Model] = Train_SGD(filename,TF,Opt,Set,SizeoRatiofReducedset,gamma)
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
%{
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



%Data set : svmguide1 - Adadelta
%{
filename = 'svmguide1';
method = 2;
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
Opt.eta  = 0.3;      %LearningRate
Opt.beta = 0.1;         %Hyper 
Opt.N = 2; 
Opt.delta = 0.95;
Opt.e = 1e-6;
%gamma = 0.00001;
gamma = 1e-3;

%Reduce kernel subset size
SizeoRatiofReducedset = 0.05;

profile on
[Result,Model] = Train_all(filename,method,TF,Opt,Set,SizeoRatiofReducedset,gamma)
profile viewer
Kernelprint(Model.RS,Model.RS,gamma);
%}


%Data set : w3a - Adadelta
%{
filename = 'w3a';
method = 2;
%% Set
Set.Minibatch = 100;   %BatchSize
Set.Epoch     = 10;   %Epoch
Set.Overlap   = 1 ;   %Overlap

%% Trade-Off
%C = 5;
TF.C  = 100;
TF.C1 = 100000;       %TrainLoss
TF.C2 = 0;      %Syn
TF.C3 = 1;      %Prox

%% Opt
Opt.eta  = 0.3;      %LearningRate
Opt.beta = 0.1;         %Hyper 
Opt.N = 2; 
Opt.delta = 0.95;
Opt.e = 1e-6;
%gamma = 0.00001;
gamma = 0.075;

%Reduce kernel subset size
SizeoRatiofReducedset = 0.0625;

profile on
[Result,Model] = Train_all(filename,method,TF,Opt,Set,SizeoRatiofReducedset,gamma)
profile viewer
Kernelprint(Model.RS,Model.RS,gamma);
%}



%Data set : svmguide1 - Adam
%{
filename = 'svmguide1';
method = 3;
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
Opt.eta  = 0.001;      %LearningRate
Opt.beta = 0.1;         %Hyper 
Opt.N = 2; 
Opt.d1 = 0.9;
Opt.d2 = 0.999;
Opt.e = 1e-8;
%gamma = 0.00001;
gamma = 1e-3;

%Reduce kernel subset size
SizeoRatiofReducedset = 0.05;

profile on
[Result,Model] = Train_all(filename,method,TF,Opt,Set,SizeoRatiofReducedset,gamma)
profile viewer
Kernelprint(Model.RS,Model.RS,gamma);
%}


%Data set : w3a - Adam
%{
filename = 'w3a';
method = 3;
%% Set
Set.Minibatch = 100;   %BatchSize
Set.Epoch     = 10;   %Epoch
Set.Overlap   = 1 ;   %Overlap

%% Trade-Off
%C = 5;
TF.C  = 0.01;
TF.C1 = 100;       %TrainLoss
TF.C2 = 0;      %Syn
TF.C3 = 0.01;      %Prox

%% Opt
Opt.eta  = 0.001;      %LearningRate
Opt.beta = 0.1;         %Hyper 
Opt.N = 2; 
Opt.d1 = 0.9;
Opt.d2 = 0.999;
Opt.e = 1e-8;
%gamma = 0.00001;
gamma = 0.075;

%Reduce kernel subset size
SizeoRatiofReducedset = 0.0625;

profile on
[Result,Model] = Train_all(filename,method,TF,Opt,Set,SizeoRatiofReducedset,gamma)
profile viewer
Kernelprint(Model.RS,Model.RS,gamma);
%}
