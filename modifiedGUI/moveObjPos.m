function moveObjPos(prd, obj_h)
    pos = get(obj_h,'Position');
    if  prd == 1
        pos(1) = 0.8;
        
    elseif prd == 2
        pos(1) = 0.1;        
    end
    set(obj_h,'Position',pos)
    drawnow()
