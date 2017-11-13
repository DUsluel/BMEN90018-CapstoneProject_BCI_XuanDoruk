function [obs_h,obs_pos,obs_side,slct]=setObsPos(obs_h,obs_pos,obs_side,slct)
rng('shuffle')
%Updates the position of the obstacle on the figure
obs_xlen = 1/2; obs_ylen = 0.1;

if obs_pos(2)+obs_ylen > 0 
    obs_pos(2) = obs_pos(2)-0.110;

elseif obs_pos(2)+obs_ylen <= 0 
    delete(obs_h);
    sideslct = datasample(slct,1);
    obs_side = slct(find(slct == sideslct,1));
    slct(find(slct == sideslct,1)) = [];
    
    if obs_side == 1 
        obs_pos = [0 1 obs_xlen obs_ylen];

    elseif obs_side == 2 
        obs_pos = [1-obs_xlen 1 obs_xlen obs_ylen];

    elseif obs_side == 0 
        obs_pos = [1 1 obs_xlen obs_ylen];
    end            

    obs_h = annotation('rectangle',obs_pos,'facecolor', [rand rand rand],'LineWidth',0.1);   
end
set(obs_h,'Position',obs_pos);    
end

