function BCI_testBinX(varargin)
%modified version of vis_stream from LSL to not only visualize but obtain
%the LSL stream
global Fs; Fs = 250;
global EEGdata; EEGdata = [];
global tempdat; tempdat = [];
global cuedata; cuedata = [];
global predicdata; predicdata = [];

global dtr;
global dtq;
global slct; slct = [ones(1,15) ones(1,15)*2]; slct = slct(randperm(length(slct)));
global loop; loop = true;
global endt;
global q_r; q_r = 'r';
global filtTbl; filtTbl = []; % the temporary filtering table containing bandpowers of each electrode 

global chunkcount;chunkcount = 0;
global chunkcheck;chunkcheck = 0;
global lda_Bt; lda_Bt = [];
global features_tst;features_tst=[]; 

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
    arg({'datascale','DataScale'},1,[0 Inf],'Initial scale of the data. The scale of the data, in units between horizontal lines; can be changed with keyboard shortcuts (see help).'), ...
    arg({'channelrange','ChannelRange'},1:8,uint32([1 1000000]),'Channels to display. The channel range to display.'), ...
    arg({'samplingrate','SamplingRate'},250,[0 Inf],'Sampling rate for display. This is the sampling rate that is used for plotting; for faster drawing.'), ...
    arg({'refreshrate','RefreshRate'},250,[0 Inf],'Refresh rate for display. This is the rate at which the graphics are updated.'), ...
    arg({'freqfilter','FrequencyFilter','moving_avg','MovingAverageLength'},0,[0 Inf],'Frequency filter. The parameters of a bandpass filter [raise-start,raise-stop,fall-start,fall-stop], e.g., [7 8 14 15] for a filter with 8-14 Hz pass-band and 1 Hz transition bandwidth between passband and stop-bands; if given as a single scalar, a moving-average filter is designed (legacy option).'), ...
    arg({'reref','Rereference'},false,[],'Common average reference. Enable this to view the data with a common average reference filter applied.'), ...
    arg({'standardize','Standardize'},false,[],'Standardize data.'), ...
    arg({'zeromean','ZeroMean'},false,[],'Zero-mean data.'), ...    
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
    % variables for prediction model
    Frfilt = evalin('base','Frfilt2');
    Spfilt = evalin('base','Spfilt2');
    lda_W = evalin('base','lda_W2');
    means_k = evalin('base','means_k2');
    paramU = evalin('base','paramU');
    
    h_fig = figure('CloseRequestFcn',@on_close,'units','normalized','outerposition',[0 0 1 1]);obs_pos = [.4 .4 .1 .1];obj_pos = [0.45 0.02 0.1 0.1];
    obj_h = annotation('rectangle',obj_pos,'facecolor', 'k','LineWidth',0.1);
    obs_h = annotation('rectangle',obs_pos,'facecolor', [rand rand rand],'LineWidth',0.1);
    obs_side = -1;drawnow;
    dtr = 3;dtq = 4; % duration of rest and cues
    on_timer()
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end      
    function on_close(varargin)        
        try
            endt = toc            
            fclose all;
            loop = false;
            delete(h_fig);
            cuedata = EEGdata(:,9);
            
            L = 1:length(cuedata);
            stem(L(cuedata==1),cuedata(cuedata==1),'x')
            hold on;
            stem(L(cuedata==2),cuedata(cuedata==2),'ro')
            stem(L(cuedata==0),cuedata(cuedata==0),'kd')
            
            sampfreq = (length(EEGdata)/endt)
            
            acc = chunkcheck/chunkcount; % test accuracy
            msgbox(num2str(acc*100),'Accuracy(%):');
            
            assignin('base','rawdata_tst',EEGdata);
            assignin('base','cuedata_tst',cuedata);            
            assignin('base','predict_tst',predicdata);
            assignin('base','features_tst',features_tst);
            assignin('base','lda_W',lda_W);
            assignin('base','lda_Bt',lda_Bt);
                                   
        catch
        end
    end    

    % update display with new data
    function on_timer(varargin)
        
        try            
            while obs_side == -1
                % pull a new chunk from LSL
                [chunk,~] = inlet.pull_chunk();                
                if isempty(chunk)
                    %disp("No incoming data")
                    continue;
                end
                obs_side = 0;  
            end
            
            tic; toc2 = 0;slctE = 0;
            while loop
                toc1 = toc;
                if isempty(slct) && (toc1-toc2) >= dtq
                    slctE = 1;
                else
                    samp = inlet.pull_sample();          
                    EEGdata = [EEGdata;[samp obs_side]];
                    tempdat = [tempdat;[samp obs_side]];
                    if length(tempdat) >= 125                        
                        if obs_side ~= 0 % cued intervals can be used to update LDA parameter
                            [P,features,means_k] = testBCIX(tempdat(:,1:8),Frfilt,Spfilt,lda_W,means_k,paramU);                                           
                            features_tst = vertcat(features_tst,[features obs_side]);
                            lda_Bt = vertcat(lda_Bt,(means_k{1}+means_k{2})*lda_W/2);
                            chunkcount = chunkcount+1;
                            if P == obs_side
                               chunkcheck = chunkcheck+1; 
                            end
                        else
                            [P,~,~] = testBCIX(tempdat(:,1:8),Frfilt,Spfilt,lda_W,means_k,paramU);
                        end                        
                        disp('Predicted:')
                        P
                        moveObjPosBin(P,obj_h);
                        predicdata = [predicdata;P];
                        tempdat = [];
                        
                    end
                end
                
                if isequal(q_r,'r') && (toc1-toc2) >= dtr
                    q_r = 'q';
                    toc2 = toc1;
                    [obs_h,obs_side,slct] = setObsPosBin(q_r,obs_h,slct);
                    if slctE == 1                        
                        on_close();
                    end
                
                elseif isequal(q_r,'q') && (toc1-toc2) >= dtq
                    q_r = 'r';
                    toc2 = toc1;
                    [obs_h,obs_side,slct] = setObsPosBin(q_r,obs_h,slct);
                    %restObjBin(obj_h);
                end
                               
            end%;on_close();
            
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

    