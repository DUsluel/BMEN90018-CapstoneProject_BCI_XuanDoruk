function [filtRes,powdata ] = testAnalyse( chunk )
powdata = [];
obsrvs = chunk(:,9);
      
[filtRes, powRes] = filtering(chunk(:,1:8),250);
powdata = [powdata; [powRes obsrvs(1:4)]];

end

