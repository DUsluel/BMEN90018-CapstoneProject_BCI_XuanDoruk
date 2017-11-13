function [obs_h,obs_side,slct]=setObsPosBin(q_r,obs_h,slct)
rng('shuffle')
%Updates the position of the obstacle on the figure
obs_xlen = 1/2; obs_ylen = 0.1;obs_ypos = 0.55;% might be 0.45
obs_pos = [];
if isequal(q_r,'r') % if resting period delete object, output 0 as obs_side (cue)
    delete(obs_h);
    obs_side = 0;
    
else % if cue period select random side, create obstacle
    sideslct = datasample(slct,1);
    obs_side = slct(find(slct == sideslct,1));
    slct(find(slct == sideslct,1)) = [];
    
    if obs_side == 2 % left
        obs_pos = [0 obs_ypos obs_xlen obs_ylen]; 
        obs_h = annotation('rectangle',obs_pos,'facecolor', [rand rand rand],'LineWidth',0.1);
    elseif obs_side == 1 % right
        obs_pos = [1-obs_xlen obs_ypos obs_xlen obs_ylen];
        obs_h = annotation('rectangle',obs_pos,'facecolor', [rand rand rand],'LineWidth',0.1);
    end
    
end
  
end

