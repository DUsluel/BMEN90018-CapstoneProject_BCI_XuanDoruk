function [ features ] = feature_extract(data,start_index,end_index,window,Fs,average)
    
    j = 0;
%     for i = start_index:window:end_index %non-overlapping windows
    region_before = data(start_index-window:start_index-1);
    for i = start_index:window/2:end_index-window+1 % 50-percent overlapping windows
        j = j+1;
        region_current = data(i:i+window-1);

%         features(j,:) =
%         bandpowering(region_current/mean(region_before),Fs);
        %features(j,:) = bandpowering((region_current-mean(region_before))/std(region_before),Fs); % Z-score normalization
         features(j,:) = bandpowering(region_current,Fs); % no process
    end
    if average == 1
        features = mean(features,1);
    end
end
