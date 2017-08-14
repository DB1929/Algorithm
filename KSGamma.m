function [] = KSGamma(Filename,RatiofRS,gamma)

load(['dataset/',Filename,'.mat'],'TInst','TLabel');
load(['dataset/',Filename,'.mat'],'VInst','VLabel');
[InstNum,~] = size(TInst);
idx = randperm(InstNum);
TInst = TInst(idx,:);
%TLabel = TLabel(idx);

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


Kernelprint([VInst(VLabel<0,:);VInst(VLabel>0,:)],Model.RS,gamma);
%Kernelprint([TInst(TLabel<0,:);TInst(TLabel>0,:)],Model.RS,gamma);

end