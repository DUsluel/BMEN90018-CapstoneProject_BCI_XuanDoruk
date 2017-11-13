function BCI_train_fast(varargin)
%modified version of vis_stream from LSL to not only visualize but obtain
%the LSL stream

global EEGdata; EEGdata = [];
global cuedata; cuedata = [];
global powdata; powdata = []; 
global filtdata; filtdata = []; 

global filtTbl; filtTbl = []; % the temporary filtering table containing bandpowers of each electrode 

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
    arg({'refreshrate','RefreshRate'},500,[0 Inf],'Refresh rate for display. This is the rate at which the graphics are updated.'), ...
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
    %stream = create_streambuffer(opts,inlet.info());
    
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    % start a timer that reads from LSL and updates the display
    th = timer('TimerFcn',@on_timer,'Period',1.0/opts.refreshrate,'ExecutionMode','fixedRate');
    % Including the GUI to data streaming
    
    h_fig = figure('CloseRequestFcn',@on_close);obs_pos = [.4 .4 .1 .1];
    obs_h = annotation('rectangle',obs_pos,'facecolor', [rand rand rand],'LineWidth',0.1);
    obs_side = -1;
          
    start(th);    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end      
    function on_close(varargin)% should not be closed before stopping the stream i.e. colData usage        
        try
            fclose all;            
            stop(th);
            delete(th);
            delete(h_fig);            
            assignin('base','rawdata_trn',EEGdata);
            assignin('base','cuedata_trn',cuedata);
            assignin('base','filtdata_trn',filtdata);
            assignin('base','powdata_trn',powdata);
            Mdl = fitcsvm(powdata(:,1:8),powdata(:,9));
            assignin('base','Mdl',Mdl);
        catch
        end
    end    

    % update display with new data
    function on_timer(varargin)
        disp('bok')
        try
            % pull a new chunk from LSL
            [chunk,timestamps] = inlet.pull_chunk();
            if isempty(chunk)                                                
                return;
            end
            
                       
            [obs_h,obs_pos,obs_side] = setObsPos(obs_h,obs_pos,obs_side);            
            drawnow;                           
                
            if obs_side ~= -1
                
                t = datetime;
                [hr,mr,sr] = hms(t);cuedata = [cuedata;[obs_side mr*60+sr]];
                
                t = datetime;samp = inlet.pull_sample();          
                [hs,ms,ss] = hms(t); EEGdata = [EEGdata;[samp ms*60+ss]];
                                
                if length(filtTbl) <= 110               
                    filtTbl = [filtTbl; [samp obs_side ms*60+ss]];
                else
                    filtTbl = [filtTbl; [samp obs_side ms*60+ss]];                    
                    [filtRes, powRes] = filtering(filtTbl(:,1:8),250);
                    disp('Filtered')
                    filtdata = [filtdata; filtRes];
                    powdata = [powdata; [powRes obs_side*ones(4,1)]];                                    
                    filtTbl = [];                    
                end                             
            end
            
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

    