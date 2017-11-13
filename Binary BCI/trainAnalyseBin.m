function [filtdata,powdata ] = trainAnalyseBin( EEGdata)
filtdata = [];
powdata = [];
gaps = 1:125:length(EEGdata);
obsrvs = EEGdata(gaps,9);
for n = 1:length(gaps)
    if n == length(gaps) 
        filtTbl = EEGdata(gaps(n):end,:);
    elseif n ~= length(gaps)
        filtTbl = EEGdata(gaps(n):gaps(n+1)-1,:);
    else
        break;
    end
    [filtRes, powRes] = filtering(filtTbl(:,1:8),250);
    filtdata = [filtdata; filtRes];
    powdata = [powdata; [powRes obsrvs(n)]];
end
end

