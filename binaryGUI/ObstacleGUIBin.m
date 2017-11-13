% GUI that randomly cretes rectangular obstacles which move towards the
% bottom of the figure
%clear;close all;clc;
function ObstacleGUIBin(h_fig,scheme)
    %h_fig = figure;
    h_fig.Units = 'normalized';
    h_fig.OuterPosition = [0.1 0.1 .8 .8];
    %create an control object 
    obj_pos = [0.45 0.02 0.1 0.1];
    obj_h = annotation('rectangle',obj_pos,'facecolor', 'k','LineWidth',0.1);
    
    scname = uicontrol('style','text','string',upper(scheme),...
                        'Units','normalized','Position',[.9 0 .1 0.1],...
                        'FontWeight','bold');
    
    countD = annotation('textbox','Units','normalized','Position',[0.35 0.35 .3 .3],...
                        'string', '3',...
                        'HorizontalAlignment','center',...
                        'VerticalAlignment', 'middle',...
                        'FontSize', 20,...
                        'BackgroundColor','r',...
                        'FontWeight','bold');
    pause(1);
    set(countD,'string',2);
    set(countD,'BackgroundColor','y');
    pause(1);
    set(countD,'string',1);
    set(countD,'BackgroundColor','g');
    pause(1);
    set(countD,'string','START');
    pause(1);
    countD.delete
    %create a "toogle button" user interface (UI) object
    but_h = uicontrol('style', 'togglebutton',...
                        'string', 'Stop',...
                        'units', 'normalized',...
                        'position', [0.01 0.9 0.1 0.1],...                        
                        'callback', {@toggleStart,obj_h,h_fig});
    
    while ~get(but_h,'Value')
        obs_side = round(rand*2);
        obs_xlen = 1/2; obs_ylen = 0.1;
        
        if obs_side == 1
            obs_pos = [0 1 obs_xlen obs_ylen];
        else 
            obs_pos = [1-obs_xlen 1 obs_xlen obs_ylen];
        end        
        obs_h = annotation('rectangle',obs_pos,'facecolor', [rand rand rand],'LineWidth',0.1);
        
        while (~get(but_h,'Value') && obs_pos(2)+obs_pos(4) >= 0)
            
            set(h_fig,'KeyPressFcn',@(h_obj,evt) moveObj(evt.Key,obj_h))
            obj_pos = get(obj_h,'Position');
            
            obs_bound = [obs_pos(1) obs_pos(1)+obs_pos(3) obs_pos(2)];
            obj_bound = [obj_pos(1) obj_pos(1)+obj_pos(3) obj_pos(2)+obj_pos(4)];
            
            % the boundries contain the x-axis boundries and the upper y limit
            % boundary of the object
            if obs_bound(3) <= obj_bound(3) && ((obs_bound(1)<obj_bound(1) && obj_bound(1)<obs_bound(2))...
                    || (obs_bound(1)<obj_bound(2) && obj_bound(2)<obs_bound(2)))
                disp('Hit')
            else
                disp('Pass')
            end
            obs_pos(2) = obs_pos(2)-0.01;
            set(obs_h,'Position',obs_pos);
            pause(0.03)
        end
        delete(obs_h);
    end
%     h_fig.delete
    
