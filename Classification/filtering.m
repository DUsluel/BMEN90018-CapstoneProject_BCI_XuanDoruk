function [Res, powers] = filtering(input, Fs)
%This function takes an input signal, the signals time information,
%sampling frequency, desired type of filtering (cheby2 bandpass 4-120Hz 
%or notch 47-53Hz), and a choice of printing two sets of figures:

    %% Filter Creation
    notchFilt = designfilt('bandstopiir','FilterOrder',20,'HalfPowerFrequency1',49,...
    'HalfPowerFrequency2',51,'SampleRate',Fs);    

    %bandFilt = designfilt('bandpassiir','DesignMethod','cheby2','StopbandFrequency1',2,'PassbandFrequency1',4,'PassbandFrequency2',100,...
    %'StopbandFrequency2',105,'StopbandAttenuation1',80,'StopbandAttenuation2',60,'SampleRate',Fs);

    %% Symmetric Padding
    if rem(length(input),2) == 0
        in1 = input((length(input)/2):-1:2,:);
        in2 = input((end-1):-1:(length(input)/2),:);
    else
        in1 = input(((length(input)+1)/2):-1:2,:);
        in2 = input((end-1):-1:((length(input)+1)/2),:);
    end
    padIn = [in1;input;in2];
    %% Filtering
    padRes = filtfilt(notchFilt,padIn);
    %padRes = filtfilt(bandFilt,padRes);
    Res = padRes((length(in1)+1):(length(in1)+length(input)),:);%Removing padded values    
%     %% Conditional Plotting of the result
%         [Finput,f] = pwelch(input,[],[],[1:125],Fs); %%%
%         [FRes,f] = pwelch(Res,[],[],[1:125],Fs); %%%
%         figure; plot(f,abs(Finput),f,abs(FRes)); xlabel('Frequency (Hz)');
%         title(['Signal PSD versus Frequency before/after filtering ' type]);
    powers = bandpowering(Res,Fs);
end