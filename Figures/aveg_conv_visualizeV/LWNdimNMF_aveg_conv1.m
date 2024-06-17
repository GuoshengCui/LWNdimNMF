% dbstop if error

clear;
clc
dataname_list = {'yalea'};% wisconsin msrcv1
for idata = 1:length(dataname_list)
    Dataname = dataname_list{idata};
    % >= 2 views
    %{
    if strcmp(Dataname,'yalea')
    percentDel_list = [0.3,0.5];% 0.1,0.3,0.5
    else
    percentDel_list = [0.1,0.3,0.5];
    end
    for ipercentDel = 1:length(percentDel_list)
        % percentDel = 0.3;
        percentDel = percentDel_list(ipercentDel);
        percent = percentDel;
    %}
    % = 2 views
    % {
    if strcmp(Dataname,'yalea')
        percent_pair_list = {0.3};% 0.3,0.5,0.7,0.9
    else
        percent_pair_list = {0.5};% 0.3,0.5,0.7,0.9
    end
    for iper_pair = 1:length(percent_pair_list)
        % percent_pair = 0.5;
        percent_pair = percent_pair_list{iper_pair};
        percent = percent_pair;
        %}
        [Datafold,Data] = getData(Dataname,percent);
        
        %% Parameters for the model
        [knn,nClass] = ...
            get_paras(Dataname,percent);
        options.k = knn;% 8 6 4 10
        options.alpha = 1e-1;% 1e-2
        options.beta = 1e1;% 1e-1
        nSubSpace = 4;
%         layerstack = {[nSubSpace*nClass];...
%             [100,nSubSpace*nClass];...
%             [200,100,nSubSpace*nClass]};
        layerstack = {[200,100,nSubSpace*nClass]};
        
        obj_layers = [];
        for ilayers = 1:length(layerstack)
        options.layers = layerstack{ilayers};
        options.max_iter = 300;
        num_folds = 1;% fixed as 1
        
            for f = 1:num_folds
                if f > 1
                    clear folds X truth
                end
                load(Data);
                load(Datafold);
                disp([Dataname,' start fold: ',num2str(f)]);

                num_view = length(X);
                numClust = length(unique(truth));
                numInst  = length(truth);
                options.n_layer = length(options.layers);

                if strcmp(Dataname,'bbcsport4vbig')
                    %{
                            ind_folds = folds(f,:,:);
                            ind_folds = squeeze(ind_folds);
                            disp(['There are ',num2str(size(folds,1)),' folds']);
                    %}
                    ind_folds = folds{f};
                    disp(['There are ',num2str(length(folds)),' folds']);
                else
                    ind_folds = folds{f};
                    disp(['There are ',num2str(length(folds)),' folds']);
                end
                gnd = truth;
                if strcmp(Dataname,'handwritten')
                    gnd = gnd + 1;
                end

                for iv = 1:num_view
                    ind_0 = find(ind_folds(:,iv) == 0);
                    %         X0{iv} = X{iv}';
                    %         X0{iv}(ind_0,:) = randn(1);
                    %         X0{iv} = abs(X{iv}');
                    %         X0{iv}(ind_0,:) = rand(1);
                    %         X0{iv} = NormalizeFea(X0{iv},1);
                    %         X1 = X{iv}';
                    %         X{iv} = X_remo_min(X{iv});
                    X1 = abs(X{iv}');% + 1e-1*rand(size(X{iv}))'
                    X1 = NormalizeFea(X1,1);
                    % % % %                         X1 = normX(X1);
                    X1(ind_0,:) = [];% 去掉 缺失样本
                    Y{iv} = X1;
                    % ------------- 构造缺失视角的索引矩阵 ----------- %
                    W0 = eye(numInst);
                    W0(ind_0,:) = [];
                    G{iv} = W0;
                    G{iv} = sparse(G{iv});
                    ind_1 = find(ind_folds(:,iv) == 1);
                    W1 = eye(numInst);
                    W1(ind_1,:) = [];
                    Gp{iv} = W1;
                    Gp{iv} = sparse(Gp{iv});
                end
                clear X X1 W1 ind_0
                X = Y;
                clear Y

                rng(f*666,'v5normal');

                [W,H,Ht,Hc,obj_final] = LWNdimNMF_aveg_conv2(X,gnd,G,options);
                
            end
           obj_layers(ilayers) = mean(obj_final(1,:)); 
           obj_rec1_layers(ilayers) = mean(obj_final(2,:)); 
           obj_rec2_layers(ilayers) = mean(obj_final(3,:)); 
        end
        
%         path = ['D:\my-work-space\11-imcomplete-mv-deepIMC\',...
%             'Figures\aveg_conv_visualizeV\not_init_nmf\'];
        path = ['D:\my-work-space\11-imcomplete-mv-deepIMC\',...
            'Figures\aveg_conv_visualizeV\init_nmf\'];
        if ~exist(path,'dir')
            mkdir(path);
        end
% % % %         save([path,num2str(percent),'_',Dataname,...
% % % %             '_obj_3layers_LWNdimNMF_',num2str(options.max_iter),'.mat'],...
% % % %             'obj_layers','obj_rec1_layers','obj_rec2_layers');
% % % %         save([path,num2str(percent),'_',Dataname,...
% % % %             '_obj_3layers_LWNdimNMF_HA_',num2str(options.max_iter),'.mat'],...
% % % %             'obj_layers','obj_rec1_layers','obj_rec2_layers');
% % % %         save([path,num2str(percent),'_',Dataname,...
% % % %             '_obj_3layers_LWNdimNMF_NN_',num2str(options.max_iter),'.mat'],...
% % % %             'obj_layers','obj_rec1_layers','obj_rec2_layers');
%         save([path,num2str(percent),'_',Dataname,...
%             '_obj_3layers_DeepNMF_',num2str(options.max_iter),'.mat'],...
%             'obj_layers','obj_rec1_layers','obj_rec2_layers');
    end
end

function [Datafold,Data] = getData(Dataname,percent)

if strcmp(Dataname,'bbcsport4vbig')
    percentDel = percent;
    Datafold = ['MV_datasets/',Dataname,'/',Dataname,...
        'RnSp_percentDel_',num2str(percentDel),'_new','.mat'];
    Data = ['MV_datasets/',Dataname,'/',Dataname,'RnSp'];
elseif strcmp(Dataname,'3sources3vbig')||...
        strcmp(Dataname,'yale')||...
        strcmp(Dataname,'caltech6v')||...
        strcmp(Dataname,'skin100')||...
        strcmp(Dataname,'test100')||...
        strcmp(Dataname,'skin1005v')||...
        strcmp(Dataname,'test1005v')||...
        strcmp(Dataname,'bankuai3v')||...
        strcmp(Dataname,'bankuai3v_2of3')||...
        strcmp(Dataname,'BDGP4v')||...
        strcmp(Dataname,'BDGP4v200')||...
        strcmp(Dataname,'caltech20')||...
        strcmp(Dataname,'msrcv1')||...
        strcmp(Dataname,'yalea')||...
        strcmp(Dataname,'adni3_9v')||...
        strcmp(Dataname,'adni3_6v')||...
        strcmp(Dataname,'adni3_4v')||...
        strcmp(Dataname,'adni3_5v')||...
        strcmp(Dataname,'adni3_6v2')||...
        strcmp(Dataname,'adni3_5v2')||...
        strcmp(Dataname,'adni3_5v3')||...
        strcmp(Dataname,'adni3_7v')||...
        strcmp(Dataname,'adni3_4v2')||...
        strcmp(Dataname,'adni3_4v3')||...
        strcmp(Dataname,'adni3_5v4')||...
        strcmp(Dataname,'adni3_5v5')||...
        strcmp(Dataname,'adni3_6v3')||...
        strcmp(Dataname,'adni3_5v6')||...
        strcmp(Dataname,'adni3_5v7')||...
        strcmp(Dataname,'adni3_6v4')||...
        strcmp(Dataname,'adni3_5v8')||...
        strcmp(Dataname,'adni3_5v9')||...
        strcmp(Dataname,'adni3_5v10')||...
        strcmp(Dataname,'adni3_5v11')||...
        strcmp(Dataname,'adni3_5v12')||...
        strcmp(Dataname,'adni3_5v13')||...
        strcmp(Dataname,'adni3_5v14')||...
        strcmp(Dataname,'breast7v_7v')||...
        strcmp(Dataname,'breast7v_6v')||...
        strcmp(Dataname,'breast7v_6v2')||...
        strcmp(Dataname,'breast7v_5v')||...
        strcmp(Dataname,'breast7v_5v2')||...
        strcmp(Dataname,'breast7v_5v3')||...
        strcmp(Dataname,'breast7v_5v4')||...
        strcmp(Dataname,'breast7v_5v5')||...
        strcmp(Dataname,'breast7v_5v6')||...
        strcmp(Dataname,'breast7v_4v')||...
        strcmp(Dataname,'breast8v_6v')||...
        strcmp(Dataname,'breast10v_5v')||...
        strcmp(Dataname,'breast10v_5v2')||...
        strcmp(Dataname,'adni3_5v15')||...
        strcmp(Dataname,'adni3_5v16')||...
        strcmp(Dataname,'adni3_6v5')||...
        strcmp(Dataname,'caltech20p50')
    percentDel = percent;
    Datafold = ['MV_datasets/',Dataname,'/',Dataname,...
        'RnSp_percentDel_',num2str(percentDel),'.mat'];
    Data = ['MV_datasets/',Dataname,'/',Dataname,'RnSp'];
else
    percent_pair = percent;
    dataroot = ['MV_datasets/',Dataname,'/',Dataname,'_with_G/'];
    Datafold = [dataroot,Dataname,'_Folds_with_G_paired_',...
        num2str(percent_pair),'.mat'];
    Data = [dataroot,Dataname,'_RnSp_with_G'];
end
end

function X = norm_max(X,normD)

if normD==1 % row max norm
    xrow_max = max(X,[],2);% row max
    X = X./xrow_max;
else
    xcol_max = max(X,[],1);% collumn max
    X = X./xcol_max;
end

end

function X = normX(X)


X = X/sum(sum(X));


end

function X = X_remo_min(X)

X = X-min(X,[],2);



end

function [Ht] = get_Ht(G,H,n_view)
M = 0;
for i = 1:n_view
    M = M + sum(G{i}',2);
end
Ht = 0;
for i = 1:n_view
    Ht = Ht + H{i}/(G{i}')*diag(1./M);
end
end

function [options] = get_k(Dataname)
% '3sources3vbig','bbcsport4vbig','bankuai3v_2of3','caltech6v'
% 'cornell','wisconsin','texas','washington'
% 'handwritten','cora'
switch Dataname
    case '3sources3vbig'
        options.k = 12;
    case 'bbcsport4vbig'
        options.k = 10;
    case 'bankuai3v_2of3'
        options.k = 2;
    case 'caltech6v'
        options.k = 4;
    case 'cornell'
        options.k = 8;
    case 'wisconsin'
        options.k = 20;
    case 'texas'
        options.k = 7;
    case 'washington'
        options.k = 18;
    case 'handwritten'
        options.k = 8;
    case 'cora'
        options.k = 12;
    case 'BDGP4v'
        options.k = 15;
    case 'skin100'
        options.k = 6;
    case 'test100'
        options.k = 6;
    otherwise
        options.k = 8;
end

end

function [knn,nSubSpace,nClass] = get_paras(dataset,percent)

if strcmp(dataset,'washington')
    nClass = 5;
    if percent==0.3
        knn = 25;
        nSubSpace = 4;
    elseif percent==0.5
        knn = 25;
        nSubSpace = 4;
    elseif percent==0.7
        knn = 25;
        nSubSpace = 4;
    elseif percent==0.9
        knn = 25;
        nSubSpace = 4;
    end
elseif strcmp(dataset,'wisconsin')
    nClass = 5;
    if percent==0.3
        knn = 30;
        nSubSpace = 4;
    elseif percent==0.5
        knn = 30;
        nSubSpace = 4;
    elseif percent==0.7
        knn = 30;
        nSubSpace = 4;
    elseif percent==0.9
        knn = 30;
        nSubSpace = 4;
    end
elseif strcmp(dataset,'handwritten')
    nClass = 10;
    if percent==0.3
        knn = 9;
        nSubSpace = 4;
    elseif percent==0.5
        knn = 9;
        nSubSpace = 4;
    elseif percent==0.7
        knn = 9;
        nSubSpace = 4;
    elseif percent==0.9
        knn = 9;
        nSubSpace = 4;
    end
elseif strcmp(dataset,'cora')
    nClass = 7;
    if percent==0.3
        knn = 9;
        nSubSpace = 4;
    elseif percent==0.5
        knn = 9;
        nSubSpace = 4;
    elseif percent==0.7
        knn = 9;
        nSubSpace = 4;
    elseif percent==0.9
        knn = 9;
        nSubSpace = 4;
    end
elseif strcmp(dataset,'3sources3vbig')
    nClass = 6;
    if percent==0.1
        knn = 9;
        nSubSpace = 4;
    elseif percent==0.3
        knn = 9;
        nSubSpace = 4;
    elseif percent==0.5
        knn = 9;
        nSubSpace = 4;
    end
elseif strcmp(dataset,'msrcv1')
    nClass = 7;
    if percent==0.1
        knn = 3;
        nSubSpace = 4;
    elseif percent==0.3
        knn = 3;
        nSubSpace = 4;
    elseif percent==0.5
        knn = 3;
        nSubSpace = 4;
    end
elseif strcmp(dataset,'yalea')
    nClass = 15;
    if percent==0.1
        knn = 2;
        nSubSpace = 4;
    elseif percent==0.3
        knn = 2;
        nSubSpace = 4;
    elseif percent==0.5
        knn = 2;
        nSubSpace = 4;
    end
elseif strcmp(dataset,'adni3_5v')
    nClass = 4;
    if percent==0.1
        knn = 2;
        nSubSpace = 4;
    elseif percent==0.3
        knn = 2;
        nSubSpace = 4;
    elseif percent==0.5
        knn = 2;
        nSubSpace = 4;
    end
else
    error('wrong data set name!!!!');
end

end

