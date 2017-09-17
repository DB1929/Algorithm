clc
clear all
method = 'RSVM_SN'
filename = 'svmguide1';

%% Set
Set.Minibatch = 100;   %BatchSize
Set.Epoch     = 5;   %Epoch
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

%==========================================================================
% Report
%
fileID = fopen([method,'_',filename,'_report','.txt'],'w');
fprintf(fileID,'%s\n','#### Setting ##############');
fprintf(fileID,'%15s%d\n','Minibatch = ',Set.Minibatch);
fprintf(fileID,'%15s%d\n','Epoch = ',Set.Epoch);
fprintf(fileID,'%15s%d\n','Kernel size = ',length(Model.RS(:,1)));
fprintf(fileID,'%s\n','#### Trade-off ############');
fprintf(fileID,'%15s%f\n','Regular = ',TF.C);
fprintf(fileID,'%15s%f\n','Empricial = ',TF.C1);
fprintf(fileID,'%15s%f\n','Proximal = ',TF.C3);
fprintf(fileID,'%15s%f\n','Synthestic = ',TF.C2);
fprintf(fileID,'%15s%f\n','gamma = ',gamma);
fprintf(fileID,'%s\n','#### Opt ##################');
fprintf(fileID,'%18s%d\n','Step Type = ',Opt.N);
fprintf(fileID,'%18s%f\n','Learning rate = ',Opt.eta);
fprintf(fileID,'%18s%f\n','Hypergradient = ',Opt.beta);
fprintf(fileID,'%s\n','#### Result ###############');
fprintf(fileID,'%s\n','Training Time = ');
fprintf(fileID,'%9f\b%9f\b%9f\b\n',Result.train.time);
fprintf(fileID,'%\ns\n','---');
fprintf(fileID,'%s\n','Testing error = ');
fprintf(fileID,'%9f\b%9f\b%9f\b\n',Result.test.Testing_error);
fprintf(fileID,'%\ns\n','---');
fprintf(fileID,'%s%f\n','min_testing_err = ',Result.test.min_testing_err);
fprintf(fileID,'%s%f\n','max_testing_err = ',Result.test.max_testing_err);
fprintf(fileID,'%s%f\n','avg_testing_err = ',Result.test.avg_testing_err);
fprintf(fileID,'%s%f\n','std_testing_err = ',Result.test.std_testing_err);
fprintf(fileID,'%s\n','#### end ##################');
fclose(fileID);