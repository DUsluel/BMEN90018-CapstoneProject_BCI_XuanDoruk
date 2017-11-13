function [ pband ] = bandpowering( data,Fs )
    
    pband = [bandpower(data,Fs,[5 13]) bandpower(data,Fs,[15 23]) bandpower(data,Fs,[28 35]) bandpower(data,Fs,[38 50])];
    
        
%     pband(1) = bandpower(data,Fs,[8 28]);
%     pband(2) = bandpower(data,Fs,[30 47]);
%     pband(3) = bandpower(data,Fs,[53 60]);
    
%     window = length(data);
%     overlap = window*0.2;
%     freq = [4:120];
%     [psd_data,f] = pwelch(data,window,overlap,freq,Fs);
%     pband(1) = bandpower(psd_data,f,[8 28],'psd');
%     pband(2) = bandpower(psd_data,f,[30 47],'psd');
%     pband(3) = bandpower(psd_data,f,[53 60],'psd');

%     bp_tot = bandpower(data,Fs,[4 120]);
%     pband(1) = bandpower(data,Fs,[8 28])/bp_tot*100;
%     pband(2) = bandpower(data,Fs,[30 47])/bp_tot*100;
%     pband(3) = bandpower(data,Fs,[53 60])/bp_tot*100;

end
