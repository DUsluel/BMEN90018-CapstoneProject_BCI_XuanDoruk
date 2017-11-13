% create a new stream buffer to hold our data
function stream = create_streambuffer(opts,info)
    stream.srate = info.nominal_srate();
    stream.chanlocs = struct('labels',derive_channel_labels(info));
    stream.buffer = zeros(length(stream.chanlocs),max(max(opts.bufferrange,opts.timerange)*stream.srate,100));
    [stream.nsamples,stream.state] = deal(0,[]);
end