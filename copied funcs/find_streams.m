% find names of streams on the lab network
function names = find_streams(lib)
    streams = lsl_resolve_all(lib,0.3);
    names = unique(cellfun(@(s)s.name(),streams ,'UniformOutput',false));
    if isempty(names)
        error('There is no stream visible on the network.'); end
end