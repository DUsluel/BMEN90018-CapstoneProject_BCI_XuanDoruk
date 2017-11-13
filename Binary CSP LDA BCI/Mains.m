% change channel index; filter order; eTrain without label?
%% Main page
clear; clc;
% 
load 251017_CSPLDABin_ng.mat % train: 5, test: 15 
lblTrain = cuedata_trn;
eTrain = rawdata_trn(:,1:8);
lblTest = cuedata_tst;
eTest = rawdata_tst(:,1:8);

%% Training:
Fs = 250;
wnd_test = 0.5*Fs; % classification length (sec *Fs)
[Frfilt,Spfilt,lda_W,lda_B0,means_k,train_epocAdj] = trainBCI(eTrain,lblTrain,wnd_test,Fs,1);

%% Offline Testing for LDA update parameter:
[paramU,features_tst,lda_Bt] = offtestBCI(eTest,lblTest,wnd_test,Frfilt,Spfilt,lda_W,means_k,train_epocAdj,1);

%% Online Testing:
paramU = 0; %% LDA udpate parameter
% assumes that 'eTest_t' is in chunk of 'wnd' length
% lblTest_t is the current class label
count = 0; correct = 0;
features_tst = []; % store output features for later analysis
lda_Bt = []; % store updated bias for later analysis
if lblTest_t ~= 0 % cued intervals can be used to update LDA parameter
    [out,fea,means_k] = testBCI(eTest_t,Frfilt,Spfilt,lda_W,means_k,paramU);
    features_tst = vertcat(features_tst,[fea lblTest_t]);
    lda_Bt = vertcat(lda_Bt,(means_k{1}+means_k{2})*lda_W/2);
    count = count+1;
    if out == lblTest_t
       correct = correct+1; 
    end
else % during rest 
    [out,~,~] = testBCI(eTest_t,Frfilt,Spfilt,lda_W,means_k,paramU);
end
acc = correct/count; % test accuracy
msgbox(num2str(acc*100),'Accuracy(%):');

%% (Analysis: test classification output visualise)
period = 1:size(features_tst,1); % period to be analysed 
features_period = features_tst(period,:);  
pca_data = pca(features_period(:,1:end-1)); 
for k = 1:2
    fea_k = features_period(features_period(:,end)==k,1:end-1);
    x_axs{k} = fea_k *lda_W;
    y_axs{k} = fea_k *pca_data(:,1);
end
figure; hold on
scatter(x_axs{1},y_axs{1},'r','filled');
scatter(x_axs{2},y_axs{2},'b','filled'); 
yy = ylim; 
line([lda_B0 lda_B0],[yy(1) yy(2)],'LineStyle',':'); 
line([lda_Bt(period) lda_Bt(period)],[yy(1) yy(2)]); 
% xlim([0 8]);


%% Checking bandpass filter
Fs = 250;
freqRange = [7 30];
Frfilt = designfilt('bandpassiir','FilterOrder',30,...
         'HalfPowerFrequency1',freqRange(1),'HalfPowerFrequency2',freqRange(2),...
         'DesignMethod','butter','SampleRate',Fs);  
fvtool(Frfilt); % to analyse filter     
%%
a=3001:3125; c=1;     
all = eTrain(1:length(eTrain)/2,c);
sec = eTrain(a,c);
all_f = bpfilt(all,Frfilt);
all2_f = FilteringT(all,Fs,7,30);
sec_f = bpfilt(sec,Frfilt);
sec2_f = FilteringT(sec,Fs,7,30);
figure; hold on
plot(sec-mean(sec),'g'); 
plot(all_f(a),'b-.'); plot(sec_f,'b'); 
plot(all2_f(a),'r-.');plot(sec2_f,'r');
%%
fr_sec = fft(sec);
fr_sec_f = fft(sec_f);
f = linspace(0,Fs/2,length(sec)/2);
figure; hold on
plot(f,abs(fr_sec(1:length(sec)/2)),'g');
plot(f,abs(fr_sec_f(1:length(sec)/2)),'b');
xlim([3 125]);