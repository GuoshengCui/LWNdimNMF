function [knn,alpha,layers] = ...
    get_paras_3layers_lwn_halign(dataset,percent,nlayer)

if strcmp(dataset,'washington')
    nClass = 5;
    if percent==0.3 
        if nlayer==1
            knn = 19;
            alpha = 1e-1;
            nSubSpace = 1;
            layers = [nSubSpace*nClass]; 
        elseif nlayer==2
            knn = 13;
            alpha = 1e0;
            nSubSpace = 4;
            layers = [100,nSubSpace*nClass];% reported
        elseif nlayer==3
            knn = 15;
            alpha = 1e0;
            nSubSpace = 4;
            layers = [100,50,nSubSpace*nClass]; 
        else
           error('wrong parameter nlayer!'); 
        end
    elseif percent==0.5
        if nlayer==1
            knn = 33;
            alpha = 1e3;
            nSubSpace = 4;
            layers = [nSubSpace*nClass]; 
        elseif nlayer==2
            knn = 17;
            alpha = 1e2;
            nSubSpace = 4;
            layers = [100,nSubSpace*nClass];% reported
        elseif nlayer==3
            knn = 18;
            alpha = 1e0;
            nSubSpace = 1;
            layers = [100,50,nSubSpace*nClass]; 
        else
           error('wrong parameter nlayer!'); 
        end
    elseif percent==0.7
        if nlayer==1
            knn = 33;
            alpha = 1e3;
            nSubSpace = 4;
            layers = [nSubSpace*nClass]; 
        elseif nlayer==2
            knn = 18;
            alpha = 1e3;
            nSubSpace = 4;
            layers = [50,nSubSpace*nClass];% reported
        elseif nlayer==3
            knn = 18;
            alpha = 1e0;
            nSubSpace = 4;
            layers = [200,50,nSubSpace*nClass]; 
        else
           error('wrong parameter nlayer!'); 
        end
    elseif percent==0.9
        if nlayer==1
            knn = 11;
            alpha = 1e0;
            nSubSpace = 4;
            layers = [nSubSpace*nClass]; 
        elseif nlayer==2
            knn = 42;
            alpha = 1e3;
            nSubSpace = 4;
            layers = [100,nSubSpace*nClass];% reported
        elseif nlayer==3
            knn = 19;
            alpha = 1e2;
            nSubSpace = 4;
            layers = [50,50,nSubSpace*nClass]; 
        else
           error('wrong parameter nlayer!'); 
        end
    end
elseif strcmp(dataset,'wisconsin')
    nClass = 5;
    if percent==0.3
        if nlayer==1
            knn = 15;
            alpha = 1e-1;
            nSubSpace = 1;
            layers = [nSubSpace*nClass];% reported
        elseif nlayer==2
            knn = 15;
            alpha = 1e0;
            nSubSpace = 1;
            layers = [100,nSubSpace*nClass];
        elseif nlayer==3
            knn = 22;
            alpha = 1e0;
            nSubSpace = 1;
            layers = [200,100,nSubSpace*nClass]; 
        else
           error('wrong parameter nlayer!'); 
        end
    elseif percent==0.5
        if nlayer==1
            knn = 16;
            alpha = 1e-1;
            nSubSpace = 1;
            layers = [nSubSpace*nClass];
        elseif nlayer==2
            knn = 16;
            alpha = 1e0;
            nSubSpace = 1;
            layers = [100,nSubSpace*nClass];% reported
        elseif nlayer==3
            knn = 20;
            alpha = 1e0;
            nSubSpace = 1;
            layers = [200,100,nSubSpace*nClass]; 
        else
           error('wrong parameter nlayer!'); 
        end
    elseif percent==0.7
        if nlayer==1
            knn = 15;
            alpha = 1e-1;
            nSubSpace = 1;
            layers = [nSubSpace*nClass];
        elseif nlayer==2
            knn = 18;
            alpha = 1e3;
            nSubSpace = 1;
            layers = [100,nSubSpace*nClass];% reported
        elseif nlayer==3
            knn = 17;
            alpha = 1e1;
            nSubSpace = 1;
            layers = [200,100,nSubSpace*nClass]; 
        else
           error('wrong parameter nlayer!'); 
        end
    elseif percent==0.9
        if nlayer==1
            knn = 12;
            alpha = 1e-1;
            nSubSpace = 1;
            layers = [nSubSpace*nClass];
        elseif nlayer==2
            knn = 19; 
            alpha = 1e1;
            nSubSpace = 1;
            layers = [100,nSubSpace*nClass];
        elseif nlayer==3
            knn = 19;
            alpha = 1e1;
            nSubSpace = 1;
            layers = [200,100,nSubSpace*nClass]; 
        else
           error('wrong parameter nlayer!'); 
        end
    end
elseif strcmp(dataset,'3sources3vbig')
    nClass = 6;
    if percent==0.1
        if nlayer==1
            knn = 3;
            alpha = 1e0;
            nSubSpace = 4;
            layers = [nSubSpace*nClass];
        elseif nlayer==2
            knn = 19;
            alpha = 1e3;
            nSubSpace = 4;
            layers = [50,nSubSpace*nClass];% reported
        elseif nlayer==3
            knn = 10;
            alpha = 1e0;
            nSubSpace = 4;
            layers = [300,100,nSubSpace*nClass]; 
        else
           error('wrong parameter nlayer!'); 
        end
    elseif percent==0.3
        if nlayer==1
            knn = 4;
            alpha = 1e0;
            nSubSpace = 4;
            layers = [nSubSpace*nClass];
        elseif nlayer==2
            knn = 17;
            alpha = 1e1;
            nSubSpace = 4;
            layers = [50,nSubSpace*nClass];% reported
        elseif nlayer==3
            knn = 8;
            alpha = 1e0;
            nSubSpace = 4;
            layers = [200,100,nSubSpace*nClass]; 
        else
           error('wrong parameter nlayer!'); 
        end
    elseif percent==0.5
        if nlayer==1
            knn = 3;
            alpha = 1e0;
            nSubSpace = 4;
            layers = [nSubSpace*nClass];
        elseif nlayer==2
            knn = 2;
            alpha = 1e0;
            nSubSpace = 4;
            layers = [50,nSubSpace*nClass];% reported
        elseif nlayer==3
            knn = 3;
            alpha = 1e0;
            nSubSpace = 4;
            layers = [200,100,nSubSpace*nClass]; 
        else
           error('wrong parameter nlayer!'); 
        end
    end
elseif strcmp(dataset,'yalea')
    nClass = 15;
    if percent==0.1
        if nlayer==1
            knn = 1;
            alpha = 1e-3;
            nSubSpace = 4;
            layers = [nSubSpace*nClass];
        elseif nlayer==2
            knn = 4;
            alpha = 1e-1;
            nSubSpace = 4;
            layers = [100,nSubSpace*nClass]; 
        elseif nlayer==3
            knn = 5;
            alpha = 1e-1;
            nSubSpace = 1;
            layers = [200,100,nSubSpace*nClass];% reported
        else
           error('wrong parameter nlayer!'); 
        end
    elseif percent==0.3
        if nlayer==1
            knn = 1;
            alpha = 1e-3;
            nSubSpace = 4;
            layers = [nSubSpace*nClass];
        elseif nlayer==2
            knn = 4;
            alpha = 1e-1;
            nSubSpace = 4;
            layers = [50,nSubSpace*nClass]; 
        elseif nlayer==3
            knn = 4;
            alpha = 1e-1;
            nSubSpace = 4;
            layers = [100,100,nSubSpace*nClass];% reported
        else
           error('wrong parameter nlayer!'); 
        end
    elseif percent==0.5
        if nlayer==1
            knn = 2;
            alpha = 1e-3;
            nSubSpace = 4;
            layers = [nSubSpace*nClass];
        elseif nlayer==2
            knn = 3;
            alpha = 1e-1;
            nSubSpace = 1;
            layers = [50,nSubSpace*nClass]; 
        elseif nlayer==3
            knn = 5;
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
