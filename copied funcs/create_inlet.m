% create an inlet to read from the stream with the given name
function inlet = create_inlet(lib,opts)
    % look for the desired device
    result = {};
    disp(['Looking for a stream with name=' opts.streamname ' ...']);
    while isempty(result)
        result = lsl_resolve_byprop(lib,'name',opts.streamname); end
    % create a new inlet
    disp('Opening an inlet...');
    inlet = lsl_inlet(result{1},opts.bufferrange);
end