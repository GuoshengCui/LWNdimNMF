
function [knn,alpha,beta,layers] = ...
    get_paras_3layers_lwn_init_nmf(dataset,percent,nlayer)

if strcmp(dataset,'wisconsin')
    nClass = 5;
    if percent==0.7
        if nlayer==1
            knn = 31;
            alpha = 1e-1;
            beta = 1e1;
            nSubSpace = 4;
            layers = [nSubSpace*nClass];
        elseif nlayer==2
            knn = 28;
            alpha = 1e-1;
            beta = 1e0;
            nSubSpace = 4;
            layers = [100,nSubSpace*nClass];% reported
        elseif nlayer==3
            knn = 28;
            alpha = 1e-2;
            beta = 1e0;
            nSubSpace = 4;
            layers = [200,100,nSubSpace*nClass]; 
        else
           error('wrong parameter nlayer!'); 
        end
    end
elseif strcmp(dataset,'yalea')
    nClass = 15;
    if percent==0.3
        if nlayer==1
            knn = 1;
            alpha = 1e0;
            beta = 1e0;
            nSubSpace = 4;
            layers = [nSubSpace*nClass];
        elseif nlayer==2
            knn = 2;
            alpha = 1e0;
            beta = 1e0;
            nSubSpace = 4;
            layers = [400,nSubSpace*nClass]; 
        elseif nlayer==3
            knn = 2;
            alpha = 1e0;
            beta = 1e0;
            nSubSpace = 4;
            layers = [300,200,nSubSpace*nClass];% reported
        else
           error('wrong parameter nlayer!'); 
        end
    end
else
    error('wrong data set name!!!!');
end

end
