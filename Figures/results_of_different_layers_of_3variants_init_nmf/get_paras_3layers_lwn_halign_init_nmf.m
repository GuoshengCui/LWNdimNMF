function [knn,alpha,layers] = ...
    get_paras_3layers_lwn_halign_init_nmf(dataset,percent,nlayer)

if strcmp(dataset,'wisconsin')
    nClass = 5;
    if percent==0.7
        if nlayer==1
            knn = 15;
            alpha = 1e-4;
            nSubSpace = 1;
            layers = [nSubSpace*nClass];
        elseif nlayer==2
            knn = 18;
            alpha = 1e-5;
            nSubSpace = 1;
            layers = [100,nSubSpace*nClass];% reported
        elseif nlayer==3
            knn = 25;
            alpha = 1e-7;
            nSubSpace = 1;
            layers = [200,100,nSubSpace*nClass]; 
        else
           error('wrong parameter nlayer!'); 
        end
    end
elseif strcmp(dataset,'yalea')
    nClass = 15;
    if percent==0.3
        if nlayer==1
            knn = 3;
            alpha = 1e-3;
            nSubSpace = 4;
            layers = [nSubSpace*nClass];
        elseif nlayer==2
            knn = 2;
            alpha = 1e-3;
            nSubSpace = 4;
            layers = [50,nSubSpace*nClass]; 
        elseif nlayer==3
            knn = 2;
            alpha = 1e-1;
            nSubSpace = 4;
            layers = [100,100,nSubSpace*nClass];% reported
        else
           error('wrong parameter nlayer!'); 
        end
    end
else
    error('wrong data set name!!!!');
end

end
