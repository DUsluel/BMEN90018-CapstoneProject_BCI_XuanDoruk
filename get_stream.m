function get_stream(varargin)
%modified version of vis_stream from LSL to not only visualize but obtain
%the LSL stream

global datafileID 
datafileID = fopen('test_lsl2mat.txt','w'); % open file for writing EEG data

% make sure that dependencies are on the path and that LSL is loaded
if ~exist('arg_define','file')
    addpath(genpath(fileparts(mfilename('fullpath')))); end
try
    lib = lsl_loadlib(env_translatepath('dependencies:/liblsl-Matlab/bin'));
catch
    lib = lsl_loadlib();
end

% handle input arguments
streamnames = find_streams(lib);
opts = arg_define(varargin, ...
    arg({'streamname','StreamName'},streamnames{2},streamnames,'LSL stream that should be displayed. The name of the stream that you would like to display.'), ...
    arg({'bufferrange','BufferRange'},10,[0 Inf],'Maximum time range to buffer. Imposes an upper limit on what can be displayed.'), ...
    arg({'timerange','TimeRange'},5,[0 Inf],'Initial time range in seconds. The time range of the display window; can be changed with keyboard shortcuts (see help).'), ...
    arg({'datascale','DataScale'},150,[0 Inf],'Initial scale of the data. The scale of the data, in units between horizontal lines; can be changed with keyboard shortcuts (see help).'), ...
    arg({'channelrange','ChannelRange'},1:8,uint32([1 1000000]),'Channels to display. The channel range to display.'), ...
    arg({'samplingrate','SamplingRate'},250,[0 Inf],'Sampling rate for display. This is the sampling rate that is used for plotting; for faster drawing.'), ...
    arg({'refreshrate','RefreshRate'},10,[0 Inf],'Refresh rate for display. This is the rate at which the graphics are updated.'), ...
    arg({'freqfilter','FrequencyFilter','moving_avg','MovingAverageLength'},0,[0 Inf],'Frequency filter. The parameters of a bandpass filter [raise-start,raise-stop,fall-start,fall-stop], e.g., [7 8 14 15] for a filter with 8-14 Hz pass-band and 1 Hz transition bandwidth between passband and stop-bands; if given as a single scalar, a moving-average filter is designed (legacy option).'), ...
    arg({'reref','Rereference'},false,[],'Common average reference. Enable this to view the data with a common average reference filter applied.'), ...
    arg({'standardize','Standardize'},false,[],'Standardize data.'), ...
    arg({'zeromean','ZeroMean'},true,[],'Zero-mean data.'), ...
    arg({'fullData','fData'},[],[],'Contains the full data of the stream duration'),... % created afterwards for data streaming in to matlab workspace
    arg_nogui({'parent_fig','ParentFigure'},[],[],'Parent figure handle.'), ...
    arg_nogui({'parent_ax','ParentAxes'},[],[],'Parent axis handle.'), ...    
    arg_nogui({'pageoffset','PageOffset'},0,uint32([0 100]),'Channel page offset. Allows to flip forward or backward pagewise through the displayed channels.'), ...
    arg_nogui({'position','Position'},[],[],'Figure position. Allows to script the position at which the figures should appear.','shape','row'));

if isempty(varargin)
    % bring up GUI dialog if no arguments were passed (calls the function again)
    arg_guidialog;
else
    % create stream inlet, figure and stream buffer
    inlet = create_inlet(lib,opts);
    stream = create_streambuffer(opts,inlet.info());
    
    %stream.data = []; %for colData function purpose
    
    [fig,ax,lines] = create_figure(opts,@on_key,@on_close);
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    % start a timer that reads from LSL and updates the display
    th = timer('TimerFcn',@on_timer,'Period',1.0/opts.refreshrate,'ExecutionMode','fixedRate');
    % Including the GUI to data streaming
    h_fig = figure;
    ObstacleGUI(h_fig,'trial')
    start(th);    
%     tic %timer value obtaining and manipulation of cue timing
%     while toc ~= 10
%     end
%     disp(toc)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end      

    % handle key presses
    function on_key(key)
        switch lower(key)
            case 'uparrow' % decrease datascale                
                opts.datascale = opts.datascale*0.9;
            case 'downarrow' % increase datascale                
                opts.datascale = opts.datascale*1.1;
            case 'rightarrow' % increase timerange                
                opts.timerange = opts.timerange*1.1;                
            case 'leftarrow' % decrease timerange                
                opts.timerange = opts.timerange*0.9;                
            case 'pagedown' % shift display page offset down                
                opts.pageoffset = min(opts.pageoffset+1,ceil(size(stream.buffer,1)/numel(opts.channelrange))-1);
            case 'pageup' % shift display page offset up                
                opts.pageoffset = max(opts.pageoffset-1,0);
        end
    end

    % close figure, timer and stream
    function on_close(varargin)% should not be closed before stopping the stream i.e. colData usage
        
        try
            fclose(datafileID);
            delete(fig);
            stop(th);
            delete(th);            
        catch
        end
    end
    
    % update display with new data
    function on_timer(varargin)
        
        try 
            % pull a new chunk from LSL
            [chunk,timestamps] = inlet.pull_chunk();
            if isempty(chunk)                                                
                return; 
            end
            
            % append it to the stream buffer
            [stream.nsamples,stream.buffer(:,1+mod(stream.nsamples:stream.nsamples+size(chunk,2)-1,size(stream.buffer,2)))] = deal(stream.nsamples + size(chunk,2),chunk);

            % extract channels/samples to plot
            samples_to_get = min(size(stream.buffer,2), round(stream.srate*opts.timerange));
            channels_to_get = intersect(opts.channelrange + opts.pageoffset*length(opts.channelrange), 1:size(stream.buffer,1));
            stream.data = stream.buffer(channels_to_get,1+round(mod(stream.nsamples-samples_to_get: stream.srate/opts.samplingrate : stream.nsamples-1,size(stream.buffer,2))));                       
            %ty = 'Try writing into file'
            elecs = [stream.data(1,:);stream.data(2,:);stream.data(3,:);stream.data(4,:);stream.data(5,:);stream.data(6,:);stream.data(7,:);stream.data(8,:)]; %EEG electrodes
            fprintf(datafileID,'%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\r\n',elecs);
            
            %fclose(datafileID);
            
            [stream.nbchan,stream.pnts,stream.trials] = size(stream.data);
            stream.xmax = max(timestamps) - lsl_local_clock(lib);
            stream.xmin = stream.xmax - (samples_to_get-1)/stream.srate;

            % arrange for plotting
            plotoffsets = (0:stream.nbchan-1)'*opts.datascale;
            plotdata = bsxfun(@plus, stream.data, plotoffsets);           
            plottime = linspace(stream.xmin,stream.xmax,stream.pnts);
        
            % update graphics
            if isempty(lines)                        
                lines = plot(ax,plottime,plotdata);
                title(ax,opts.streamname);
                xlabel(ax,'Time (sec)','FontSize',12);
                ylabel(ax,'Activation','FontSize',12);
            else
                for k=1:min(length(lines),size(plotdata,1))
                    set(lines(k),'Xdata',plottime, 'Ydata',plotdata(k,:)); end
                for k = size(plotdata,1)+1:length(lines)
                    set(lines(k),'Ydata',nan(stream.pnts,1)); end
            end            
            
            % update the axis limit and tickmarks
            axis(ax,[stream.xmin stream.xmax -opts.datascale stream.nbchan*opts.datascale + opts.datascale]);
            set(ax, 'YTick',plotoffsets, 'YTickLabel',{stream.chanlocs(channels_to_get).labels});
            %stream.chanlocs(channels_to_get).labels %channel names
            drawnow;
        catch e
            % display error message
            fprintf('vis_stream error: %s\noccurred in:\n',e.message);
            for st = e.stack'
                if ~isdeployed
                    try
                        fprintf('   <a href="matlab:opentoline(''%s'',%i)">%s</a>: %i\n',st.file,st.line,st.name,st.line);
                    catch
                        fprintf('   <a href="matlab:edit %s">%s</a>: %i\n',st.file,st.name,st.line);
                    end
                else
                    fprintf('   %s: %i\n',st.file,st.line);
                end
            end
            on_close();
        end
    end


end

