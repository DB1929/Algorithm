function [] = script1(filename,method,SizeoRatiofReducedset,gamma,Set,type)



    fileID = fopen([method,'_',filename,'_report','.txt'],'w');
    count = 1; 


    %% Fixed Setting
    %% Preprocessing 
    % load training and testing dataset
    load(['dataset/',filename,'.mat'],'TInst','TLabel');
    load(['dataset/',filename,'.mat'],'VInst','VLabel');
    [InstNum,~] = size(TInst);

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
    Opt.N = type;
    Model.gamma = gamma;

    %%
    TF.C  = 1;
    TF.C2 = 0;      %Syn



    %TF.C1 = 1000;      %TrainLoss
    %TF.C3 = 1000;      %Prox
    CC = [1000,100,10,1,0.1,0.01,0.001];
    %% Opt
    %Opt.eta  = 0.1;      %LearningRate
    %Opt.beta = 0.1;      %Hyper 
    Step = [0.5,0.1,0.01,0.001,0.0001] ;
    if Opt.N == 2
        Step1 = Step;
    else
        Step1= 1;
    end
    for c1 = CC 
        TF.C1 = c1;
        for c3 = CC  
            TF.C3 = c3;
            for eta = Step   
                Opt.eta = eta;
                for beta= Step1            
                    Opt.beta = beta;  
    Model.W = rand(SizeofReducedset+1,Set.Epoch);
    %% Training
    [Model,Result.train] = RSVM_SN_PHS(Model,TF,Opt,Set,TInst,TLabel);

    %% Testing
    [Result.test] = Prediction(Model,VInst,VLabel);


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
    fprintf(fileID,'%15s%f\n','gamma = ',gamma);
    fprintf(fileID,'%s\n','#### Opt ##################');
    fprintf(fileID,'%18s%d\n','Step Type = ',Opt.N);
    fprintf(fileID,'%18s%f\n','Learning rate = ',Opt.eta);
    fprintf(fileID,'%18s%f\n','Hypergradient = ',Opt.beta);
    fprintf(fileID,'%s\n','#### Result ###############');
    fprintf(fileID,'%s\n','Training Time = ');
    fprintf(fileID,'%9f\b%9f\b%9f\b\n',Result.train.time);
    fprintf(fileID,'\n%s\n','xxx');
    fprintf(fileID,'%s\n','Testing error = ');
    fprintf(fileID,'%9f\b%9f\b%9f\b\n',Result.test.Testing_error);
    fprintf(fileID,'\n%s\n','xxx');
    fprintf(fileID,'%s%f\n','min_testing_err = ',Result.test.min_testing_err);
    fprintf(fileID,'%s%f\n','max_testing_err = ',Result.test.max_testing_err);
    fprintf(fileID,'%s%f\n','avg_testing_err = ',Result.test.avg_testing_err);
    fprintf(fileID,'%s%f\n','std_testing_err = ',Result.test.std_testing_err);
    fprintf(fileID,'%s\n','#### end ##################');
    fprintf(fileID,'%s%d\n','#',count);
    fprintf(fileID,'%s\n','xxxxxxxxxxxxxxxxxxxxxxxxx');
    count = count+1;
    clear Result;
              
                end                
            end            
        end       
    end


    fclose(fileID);
end