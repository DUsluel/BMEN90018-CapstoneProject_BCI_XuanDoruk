% derive a list of channel labels for the given stream info
function channels = derive_channel_labels(info)
    channels = {};
    ch = info.desc().child('channels').child('channel');
    while ~ch.empty()
        name = ch.child_value_n('label');
        if name
            channels{end+1} = name; end %#ok<AGROW>
        ch = ch.next_sibling_n('channel');
    end
    if length(channels) ~= info.channel_count()
        disp('The number of channels in the steam does not match the number of labeled channel records. Using numbered labels.');
        channels = cellfun(@(k)['Ch' num2str(k)],num2cell(1:info.channel_count(),1),'UniformOutput',false);
    end
end