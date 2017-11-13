% create a new figure and axes
function [fig,ax,lines] = create_figure(opts,on_key,on_close)
    if isempty(opts.parent_ax)
        if isempty(opts.parent_fig)
            fig = figure('Name',['LSL:Stream''' opts.streamname ''''], 'CloseRequestFcn',on_close, ...
                'KeyPressFcn',@(varargin)on_key(varargin{2}.Key));
        else
            fig = opts.parent_fig;
        end
        ax = axes('Parent',fig, 'YDir','reverse');
    else
        ax = opts.parent_ax;
    end       
    lines = [];
end
