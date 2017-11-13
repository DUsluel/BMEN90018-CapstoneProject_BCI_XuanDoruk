function [paramU,features_tst,lda_Bt] = offtestBCI(eTest,lblTest,wnd_test,Frfilt,Spfilt,lda_W,means_k,train_epocAdj,fig)
% ( note: assumes the test data has same trial duration as the training )

%% Obtain start of trials
mrk = [];
for i = 2:length(lblTest)
    if lblTest(i-1)==0 && lblTest(i)~=0 % cue trial starts
        mrk = [mrk; i]; % store the starting index
    end
end

%% Off-line testing for LDA update parameter
PARAM_U = [0:0.05:0.5]; %%

features_tst = []; % store output features for later analysis
lda_Bt = []; % store updated bias for later analysis
intervals = reshape(train_epocAdj,wnd_test,[]);
for u = 1:numel(PARAM_U)
    tmp_means_k = means_k;
    tmp_lda_Bt = [];
    count = 0; correct = 0;
    for i = 1:numel(mrk) % testing is done on the epoch chosen from training
        for j = 1:size(intervals,2)
            [out,fea,tmp_means_k] = testBCIX(eTest(mrk(i)+intervals(:,j),:),Frfilt,Spfilt,lda_W,tmp_means_k,PARAM_U(u));
            tmp_lda_Bt = vertcat(tmp_lda_Bt,(tmp_means_k{1}+tmp_means_k{2})*lda_W/2);
            count = count+1;
            if out == lblTest(mrk(i))
                correct = correct+1;
            end
            if u == 1
                features_tst = vertcat(features_tst,[fea lblTest(mrk(i))]);
            end
        end
    end
    lda_Bt = horzcat(lda_Bt,tmp_lda_Bt);
    acc_U(u) = correct/count; % test accuracy
end
% Determine the best parameter
[~,idx] = max(acc_U);
paramU = PARAM_U(idx);
lda_Bt = lda_Bt(:,idx);

if fig == 1 % (Analysis: classification accuracy of LDA update) 
    figure; stem(PARAM_U,acc_U*100); 
    xlabel('LDA bias update parameter'); ylabel('Accuracy (%)');
    figure; plot(lda_Bt); % visualise change in the bias
    xlabel('Time (no scale)'); ylabel('LDA bias'); 
end

end
