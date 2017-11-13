%  https://au.mathworks.com/help/stats/support-vector-machines-for-binary-classification.html
function svmModel = svm_train( data1,data2,class1,class2 )
    
    data_in = [data1;data2];
    class = [class1.*ones(size(data1,1),1);class2.*ones(size(data2,1),1)];
    svmModel = fitclinear(data_in,class);
   
end
 
                  