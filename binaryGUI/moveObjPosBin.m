function moveObjPosBin(prd, obj_h)
    % if prediction is 1 right hand is pressed, if 2 left hand is pressed 
    pos = get(obj_h,'Position');
    if  prd == 1
        pos(1) = 0.8;
        
    elseif prd == 2
        pos(1) = 0.1;
    else
        pos(1) = 0.45;
    end
    set(obj_h,'Position',pos)
    drawnow()
