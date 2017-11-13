function [Frfilt,Spfilt,lda_W,lda_B,means_k,train_epocAdj] = trainBCIX( eTrain,lblTrain,wnd_test,Fs,fig )
% ( note: lblTrain must be {0,1,2}, 
%         must have at least 15 trials for each class & trials divisible by 5 )

% OPTIONS:
freqRange = [7 30]; % bandpass range
nof = 4; % number of CSP filter pairs
left_chn = [5 6 7 8]; % left hemisphere channel indexes
right_chn = [1 2 3 4]; % right hemisphere channel indexes


%% Filter design
Frfilt = designfilt('bandpassiir','FilterOrder',30,...
         'HalfPowerFrequency1',freqRange(1),'HalfPowerFrequency2',freqRange(2),...
         'DesignMethod','butter','SampleRate',Fs);
%     fvtool(Frfilt); to analyse filter

%% Configure for start and length of trials
mrk = [];
trial_len = [];
chk = 0;
for i = 2:length(lblTrain)
    if lblTrain(i-1)==0 && lblTrain(i)~=0 % cue trial starts
        mrk = [mrk; i]; % store the starting index
        tmp_len = 0;
        chk = 1;
    end
    if lblTrain(i-1)~=0 && (lblTrain(i)==0 || i == length(lblTrain)) % cue trial ends
        trial_len = [trial_len; tmp_len];
        chk = 0;
    end
    if chk == 1
        tmp_len = tmp_len+1;
    end
end
train_epoc = [0:min(trial_len)-1]; % make sure all trials has same size

%% Epoching
nclass{1} = 0; nclass{2} = 0; % number of class trials
for i = 1:numel(mrk)
    k = lblTrain(mrk(i)); % class label
    nclass{k} = nclass{k}+1;
    tmp_data = eTrain(mrk(i)+train_epoc,:);
    train_data{k,nclass{k}} = tmp_data;
    train_data_CAR{k,nclass{k}} = tmp_data-repmat(mean(tmp_data,2),1,size(tmp_data,2)); % common average ref
end

%% Selection of channel based on discriminative frequency band
freq = [5:35];
for chn = 1:size(eTrain,2) % channel number
    psd_chn = [];
    for k = 1:2
        for trial = 1:nclass{k}
            clear tmp_psd
            for f = 1:length(freq)
                tmp_psd(f) = bandpower(train_data_CAR{k,trial}(:,chn),Fs,[freq(f)-1 freq(f)+1]);
            end
            psd_chn = vertcat(psd_chn, [(tmp_psd) k]);
        end
    end
    psd_data{chn} = psd_chn;
    for f = 1:length(freq)
        tmp_corr = corrcoef([psd_chn(:,f) psd_chn(:,end)]); % correlation coeff
        scoreFreq(chn,f) = tmp_corr(1,2); % score of freq
    end
end
tot_scoreChn = sqrt(sum(scoreFreq.^2,2));
[~,chn] = max(tot_scoreChn(left_chn)); best_chn(1) = left_chn(chn);
[~,chn] = max(tot_scoreChn(right_chn)); best_chn(2) = right_chn(chn);

if fig == 1 %(Analysis: Spectra curve compare)
    figure; 
    for c = 1:2
        subplot(2,1,c)
        hold on
        plot(freq,mean(psd_data{best_chn(c)}(1:nclass{1},1:end-1),1),'b');
        plot(freq,mean(psd_data{best_chn(c)}(nclass{1}+1:end,1:end-1),1),'r');
        xlabel('Frequency (Hz)'); ylabel('Band-power');
        legend('class 1','class 2');
    end
end

%% Selection of discriminative time epoch
for c = 1:2
    trial_data = [];
    for k = 1:2
        for trial = 1:nclass{k}
            tmp_data = bpfilt(train_data_CAR{k,trial}(:,best_chn(c)),Frfilt); % bandpass filter
            env = envelope(tmp_data,Fs/5,'rms'); % envelope transform
            trial_data = vertcat(trial_data, [env' k]);
        end
    end
    env_data{c} = trial_data;
    for t = 1:length(train_epoc)
        tmp_corr = corrcoef([trial_data(:,t) trial_data(:,end)]); % correlation coeff
        scoreTime(c,t) = tmp_corr(1,2); % score of time instance
    end
end
tot_scoreTime = sum(abs(scoreTime),1);
[~,max_idx] = max(tot_scoreTime);
for c = 1:2
    if sum(scoreTime(c,[max_idx-1 : max_idx+1])) < 0
        scoreTime(c,:) = -scoreTime(c,:);
    end
end
tot_scoreTime = sum(scoreTime,1);
[~,max_idx] = max(tot_scoreTime);
% find time interval
thresh = 0;
for i = 1:length(tot_scoreTime)
    if tot_scoreTime(i) >0
        thresh = thresh+tot_scoreTime(i);
    end
end
thresh = 0.8*thresh;
i1 = max_idx; i2 = max_idx;
while sum(tot_scoreTime(i1:i2)) < thresh
    if i1 == 1 || i2 == length(tot_scoreTime) %%
        break;
    end
    if sum(tot_scoreTime(1:i1-1)) > sum(tot_scoreTime(i2+1:end))
        i1 = i1-1;
    else
        i2 = i2+1;
    end
end
train_epocAdj = train_epoc(i1):train_epoc(i2); % selected epoch
extras = mod(length(train_epocAdj),wnd_test);
train_epocAdj = train_epocAdj(1+ceil(extras/2):end-floor(extras/2)); % make it divisible by wnd_test

if fig == 1 %(Analysis: Time signal curve compare)
    figure; 
    for c = 1:2
        subplot(2,1,c)
        hold on
        plot(train_epoc/Fs,mean(env_data{c}(1:nclass{1},1:end-1),1),'b');
        plot(train_epoc/Fs,mean(env_data{c}(nclass{1}+1:end,1:end-1),1),'r');
        yy = ylim; 
        line([train_epocAdj(1) train_epocAdj(1)]/Fs,[yy(1) yy(2)],'Color','k');
        line([train_epocAdj(end) train_epocAdj(end)]/Fs,[yy(1) yy(2)],'Color','k');
        xlabel('Trial interval (s)'); ylabel('Envelope signal (uV)');
        legend('class 1','class 2');
    end
end

%% Bandpass filtering & Epoching discriminative time interval
for k = 1:2
    for trial = 1:nclass{k}
        tmp_data = bpfilt(train_data{k,trial},Frfilt); % bandpass filter
        train_data{k,trial} = tmp_data(train_epocAdj,:); % epoching
    end
end

%% Stationary CSP 
for k = 1:2
    for trial = 1:nclass{k}
        tmp_data = train_data{k,trial};
        tmp_data = tmp_data'*tmp_data; % sample covariance
        C{k}(:,:,trial) = tmp_data./trace(tmp_data); % normalise
    end
    avrC{k} = mean(C{k}(:,:,1:nclass{k}),3); % class average covariance
end

% Cross-validation
cv_size = 5;
CHUNK = [1 5];
PARAM_A = [0 2^-8 2^-7 2^-6 2^-5 2^-4 2^-3 2^-2 2^-1 1];
error_data = zeros(numel(CHUNK),numel(PARAM_A));
for fold1 = 1:nclass{1}/cv_size
    for fold2 = 1:nclass{2}/cv_size
        clear tmp_trainC tmp_avrC
        test_trials{1} = (fold1-1)*cv_size+[1:cv_size];
        test_trials{2} = (fold2-1)*cv_size+[1:cv_size];
        for k = 1:2
            train_trials{k} = [1:(test_trials{k}(1)-1) (test_trials{k}(end)+1):nclass{k}];
            tmp_trainC{k} = C{k}(:,:,train_trials{k});
            tmp_avrC{k} = mean(tmp_trainC{k},3); % class average covariance
        end
        
        for i = 1:numel(CHUNK)
            clear tmp_chunkavrC
            for k = 1:2
                tmp_chunkC = [];
                for c = 1:size(tmp_trainC{k},3)/CHUNK(i)
                    interval =(c-1)*CHUNK(i)+[1:CHUNK(i)];
                    tmp_chunkC(:,:,c) = mean(tmp_trainC{k}(:,:,interval),3);
                    tmp_chunkC(:,:,c) = tmp_chunkC(:,:,c)-tmp_avrC{k};
                    [tmp_vec,tmp_val] = eig(tmp_chunkC(:,:,c));
                    tmp_chunkC(:,:,c) = tmp_vec*abs(tmp_val)/tmp_vec; % positise its eig value
                    tmp_chunkC(:,:,c) = tmp_chunkC(:,:,c)./trace(tmp_chunkC(:,:,c)); % normalise
                end
                tmp_chunkavrC{k} = mean(tmp_chunkC,3); % class average covariance
            end
            tmp_penaltyC = (tmp_chunkavrC{1}+tmp_chunkavrC{2}); % penalty term
            
            for j = 1:numel(PARAM_A)
                [V1,D1] = eig(tmp_avrC{1},tmp_avrC{1}+tmp_avrC{2}+PARAM_A(j)*tmp_penaltyC);
                [V2,D2] = eig(tmp_avrC{2},tmp_avrC{1}+tmp_avrC{2}+PARAM_A(j)*tmp_penaltyC);
                [~,idxs] = sort(diag(D1),'descend');
                V1 = V1(:,idxs);
                [~,idxs] = sort(diag(D2),'ascend');
                V2 = V2(:,idxs);
                tmp_Spfilt = [V1(:,1:nof) V2(:,end-nof+1:end)];
                
                % Log-variance feature extraction
                clear features outputs
                for k = 1:2
                    features{k} = [];
                    for trial = train_trials{k}
                        for section = 1:length(train_epocAdj)/wnd_test
                            interval = (section-1)*wnd_test+[1:wnd_test];
                            features{k} = vertcat(features{k},log(var(train_data{k,trial}(interval,:)*tmp_Spfilt)));
                        end
                    end
                end
                
                % LDA Classification
                lda_W = ((mean(features{2})-mean(features{1}))/(cov(features{1})+cov(features{2})))';
                lda_B = (mean(features{1})+mean(features{2}))*lda_W/2;
                temp_error = 0;
                n_test = 0;
                for k = 1:2
                    outputs{k} = [];
                    for trial = test_trials{k}
                        for section = 1:length(train_epocAdj)/wnd_test
                            interval = (section-1)*wnd_test+[1:wnd_test];
                            temp_output = log(var(train_data{k,trial}(interval,:)*tmp_Spfilt))*lda_W - lda_B;
                            outputs{k} = vertcat(outputs{k},temp_output);
                            temp_error = temp_error +(sign(temp_output)~=k*2-3);
                            n_test = n_test+1;
                        end
                    end
                end
                error_data(i,j) = error_data(i,j) +temp_error/n_test;
                fishscore(i,j) = (mean(outputs{1})-mean(outputs{2}))^2/(var(outputs{1}+var(outputs{2})));
                
            end
        end
    end
end
error_data = error_data./(fold1*fold2);
% Determine the best parameters
[i,j] = find(error_data==min(error_data(:)));
idx_i = i(1); idx_j = j(1);
for n = 1:numel(i)
    if fishscore(i(n),j(n)) > fishscore(idx_i,idx_j)
        idx_i = i(n); idx_j = j(n);
    end
end
chunk = CHUNK(idx_i);
paramA = PARAM_A(idx_j);

if fig == 1 % (Analysis: cross validation error)
    figure; image(error_data.*100,'CDataMapping','scaled'); colorbar; colormap('gray')
    xlabel('penalty constant: [0 2^{-8} 2^{-7} 2^{-6} 2^{-5} 2^{-4} 2^{-3} 2^{-2} 2^{-1} 1]');
    ylabel('chunk size: [5 1]'); title('Cross validation (error)')
end

% chunk=1; paramA=0; %remove this
clear features
for k = 1:2
    tmp_chunkC = [];
    for c = 1:nclass{k}/chunk
        interval =(c-1)*chunk+[1:chunk];
        tmp_chunkC(:,:,c) = mean(C{k}(:,:,interval),3);
        tmp_chunkC(:,:,c) = tmp_chunkC(:,:,c)-avrC{k};
        [tmp_vec,tmp_val] = eig(tmp_chunkC(:,:,c));
        tmp_chunkC(:,:,c) = tmp_vec*abs(tmp_val)/tmp_vec; % positise its eig value
        tmp_chunkC(:,:,c) = tmp_chunkC(:,:,c)./trace(tmp_chunkC(:,:,c)); % normalise
    end
    chunkavrC{k} = mean(tmp_chunkC,3); % class average covariance
end
penaltyC = chunkavrC{1}+chunkavrC{2}; % penalty term
[V1,D1] = eig(avrC{1},avrC{1}+avrC{2}+paramA*penaltyC);
[V2,D2] = eig(avrC{2},avrC{1}+avrC{2}+paramA*penaltyC);
[~,idxs] = sort(diag(D1),'descend');
V1 = V1(:,idxs);
[~,idxs] = sort(diag(D2),'ascend');
V2 = V2(:,idxs);
Spfilt0 = [V1(:,1:length(V1)/2) V2(:,end-length(V1)/2+1:end)];
Spfilt = [V1(:,1:nof) V2(:,end-nof+1:end)];

% Log-variance feature extraction
for k = 1:2
    features{k} = [];
    for trial = 1:nclass{k}
        for section = 1:length(train_epocAdj)/wnd_test
            interval = (section-1)*wnd_test+[1:wnd_test];
            features{k} = vertcat(features{k},log(var(train_data{k,trial}(interval,:)*Spfilt)));
        end
    end
end

%% LDA Classification
covr = cov(features{1})+cov(features{2});
mean_diff = mean(features{2})-mean(features{1}); covr = covr + mean_diff'*mean_diff./4; 
lda_W = ((mean(features{2})-mean(features{1}))/covr)';
means_k{1} = mean(features{1}); means_k{2} = mean(features{2});
lda_B = (means_k{1}+means_k{2})*lda_W/2;

if fig == 1 % (Analysis: training classification output visualise)
    pca_data = pca([features{1};features{2}]);
    for k = 1:2
        x_axs{k} = features{k}*lda_W;
        y_axs{k} = features{k}*pca_data(:,1);
    end
    figure; hold on
    scatter(x_axs{1},y_axs{1},'r','filled');
    scatter(x_axs{2},y_axs{2},'b','filled');    
    yy = ylim; line([lda_B lda_B],[yy(1) yy(2)]);
    title('Training'); legend('class 1','class 2');
end

end