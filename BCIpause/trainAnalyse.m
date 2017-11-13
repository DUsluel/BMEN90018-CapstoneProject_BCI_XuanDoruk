function [filtdata,powdata ] = trainAnalyse( EEGdata )
filtdata = [];
powdata = [];
gaps = 1:400:length(EEGdata);
obsrvs = EEGdata(gaps,9);
for n = 1:length(gaps)
    if n == length(gaps) %&& (length(EEGdata)-gaps(n) >60)
        filtTbl = EEGdata(gaps(n):end,:);
    elseif n ~= length(gaps)
        filtTbl = EEGdata(gaps(n):gaps(n+1)-1,:);
    else
        break;
    end
    [filtRes, powRes] = filtering(filtTbl(:,1:8),250);
    filtdata = [filtdata; filtRes];
    powdata = [powdata; [powRes obsrvs(n)*ones(4,1)]];
end

end

