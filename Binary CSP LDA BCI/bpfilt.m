function [output] = bpfilt(input,Frfilt)
    % input takes a matrix whose rows are a time series data & columns are variables
    
    for c = 1:size(input,2)
        input_c = input(:,c);
        
        % Symmetric Padding
        if rem(length(input_c),2) == 0
            in1 = input_c((length(input_c)/2):-1:2);
            in2 = input_c((end-1):-1:(length(input_c)/2));
        else
            in1 = input_c(((length(input_c)+1)/2):-1:2);
            in2 = input_c((end-1):-1:((length(input_c)+1)/2));
        end
        padd = [in1;input_c;in2];        

        % Filtering
        outpad_c = filtfilt(Frfilt,padd);

        output(:,c) = outpad_c((length(in1)+1):(length(in1)+length(input_c))); % Removing padded values
    end

end