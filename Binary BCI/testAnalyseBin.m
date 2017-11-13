function [filtRes,powdata,chunk ] = testAnalyseBin( chunk )
powdata = [];
obsrvs = chunk(:,9);
      
[filtRes, powRes] = filtering(chunk(:,1:8),250);
powdata = [powdata; [powRes round(mean(obsrvs))]];
chunk = [];
end

