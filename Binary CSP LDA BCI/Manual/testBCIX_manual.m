function [output,features,means_k] = testBCIX_manual(data,Frfilt,Spfilt,lda_W,means_k,paramU)
    chns = evalin('base','channels1');
    data = data(:,chns);
    % Spatial filtered
    data_filt = data*Spfilt;
    % Bandpass filtered
    data_filt = bpfilt(data_filt,Frfilt);
    features = log(var(data_filt));
    
    % Update LDA's bias on each trial
    means_k{1} = (1-paramU)*means_k{1} + paramU*features;
    means_k{2} = (1-paramU)*means_k{2} + paramU*features;
    lda_B_t = (means_k{1}+means_k{2})*lda_W/2; 
    
    % LDA classifier output
    output = features*lda_W - lda_B_t;
    if sign(output)<0
        output = 1;
    else
        output = 2;
    end
    
end