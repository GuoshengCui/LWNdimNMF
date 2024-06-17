% dbstop if error

clear;
clc
% '3sources3vbig' 'bbcsport4vbig' 'citeseer','cora','cornell',...
% 'handwritten','texas','washington','wisconsin','buaa90'
% adni3 adni3_9v adni3_6v adni3_4v 【adni3_5v】
% adni3_6v2 adni3_5v2 adni3_5v3 adni3_7v adni3_4v2 adni3_4v3 
% adni3_5v4 adni3_5v5 adni3_6v3 adni3_5v6 adni3_5v7 
% adni3_6v4 adni3_5v9 adni3_5v10 adni3_5v11 adni3_5v12 
% adni3_5v13 adni3_5v14 adni3_5v15 adni3_5v16 adni3_6v5 
% breast7v_7v breast7v_6v breast7v_6v2 breast7v_5v2 breast7v_5v3 
% breast7v_5v4 【breast7v_5v5】 breast10v_5v breast10v_5v2
% breast7v_5v6 breast7v_4v breast8v_6v 
% test1002v skin1002v msrcv1 wikipa yalea 
% '3sources3vbig','adni3_5v','yalea','msrcv1'
% 'yale','caltech2v','caltech6v','skin100','test100','ecg','bankuai'
% ,'BDGP4v','caltech20','caltech20p50','BDGP4v200'
% dataname_list = {'cora','wisconsin','washington','handwritten'};
% dataname_list = {'washington','wisconsin','handwritten','cora'};% ,'wisconsin'
dataname_list = {'wisconsin'};
% dataname_list = {'washington','wisconsin'};%'3sources3vbig','bbcsport4vbig'
% dataname_list = {'3sources3vbig','adni3_5v','yalea','msrcv1'};
for idata = 1:length(dataname_list)
    % Dataname = 'caltech6v';
    Dataname = dataname_list{idata};
    % >= 2 views
    %{ 
    percentDel_list = [0.1,0.3,0.5];% 0.1,0.3,0.5
    for ipercentDel = 1:length(percentDel_list)
        % percentDel = 0.3;
        percentDel = percentDel_list(ipercentDel);
        percent = percentDel;
        %}
    % = 2 views
    % {
%     percent_pair_list = {0.3};
    percent_pair_list = {0.7};% {0.3,0.5,0.7,0.9} {0.1,0.3,0.5}
    for iper_pair = 1:length(percent_pair_list)
        % percent_pair = 0.5;
        percent_pair = percent_pair_list{iper_pair};
        percent = percent_pair;
        %}
        [Datafold,Data] = getData(Dataname,percent);
        
       %% Parameters for the model
        for nlayer = 1:3
            [knn,alpha,beta,layers] = ...
                get_paras_3layers_lwn_nnorm_init_nmf(Dataname,percent,nlayer);
            options.k = knn; 
            alpha_list = [alpha]; 
            beta_list = [beta];
            options.layers = layers;
            options.n_layer = length(options.layers);
            options.max_iter = 300;
            num_folds = 15; 
            repeat = 5; 
        
        for ialpha = 1:length(alpha_list)
            options.alpha = alpha_list(ialpha);
            for ibeta = 1:length(beta_list)
                options.beta = beta_list(ibeta);
                ACC = [];NMI = [];PUR = [];ARI = [];
                Fscore = [];Precision = [];Recall = [];
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
                    
                    [W,H,Ht,Hc] = LWNdimNMF_NNorm_warped_options(X,gnd,G,options);
                    
%                     U = Ht';
                    U = Hc';
                    U(isnan(U)) = 0;
                    U(isinf(U)) = 1e5;
                    new_F = U;
                    % {
                    norm_mat = repmat(sqrt(sum(new_F.*new_F,2)),1,size(new_F,2));
%                     norm_mat = repmat(sum(new_F,2),1,size(new_F,2));
                    %%avoid divide by zero
                    for i = 1:size(norm_mat,1)
                        if (norm_mat(i,1)==0)
                            norm_mat(i,:) = 1;
                        end
                    end
                    new_F = new_F./norm_mat;
                    %}
                    for iter_c = 1:repeat% cosine sqEuclidean
                        %         pre_labels = litekmeans(new_F ,numClust, 'Replicates',20,'Distance','sqEuclidean');
                        pre_labels    = kmeans(new_F,numClust,'emptyaction','singleton',...
                            'replicates',20,'display','off',...
                            'Distance','sqeuclidean');
                        % cosine correlation cityblock hamming sqeuclidean
                        %         pre_labels = kmeans(new_F,numClust, 'Replicates',20,'Distance','correlation');
                        result_LatLRR = ClusteringMeasure(gnd, pre_labels);
                        %         AC(iter_c)    = result_LatLRR(1);
                        %         MIhat(iter_c) = result_LatLRR(2);
                        Purity(iter_c)= result_LatLRR(3);
                        [AC(iter_c),MI(iter_c),~] = result(pre_labels,gnd);
                        [AR(iter_c),~,~,~] = RandIndex(gnd,pre_labels);
                        [Fs(iter_c),Pre(iter_c),Rec(iter_c)] = compute_f(gnd,pre_labels);
                    end
                    %     mean_ACC = mean(AC);
                    %     mean_NMI = mean(MI);
                    %     mean_PUR = mean(Purity);
                    %     disp(strcat('mean_ACC:',num2str(mean_ACC)));
                    %     disp(strcat('mean_NMI:',num2str(mean_NMI)));
                    %     disp(strcat('mean_PUR:',num2str(mean_PUR)));
                    
                    ac = mean(AC);
                    nmi = mean(MI);
                    pur = mean(Purity);
                    ar = mean(AR);
                    fs = mean(Fs);
                    p = mean(Pre);
                    r = mean(Rec);
                    clear AC MI Purity AR Fs Pre Rec
                    
                    ACC = [ACC,ac];
                    NMI = [NMI,nmi];
                    PUR = [PUR,pur];
                    ARI = [ARI,ar];
                    Fscore = [Fscore, fs];
                    Precision = [Precision, p];
                    Recall = [Recall, r];
                    
                    disp(strcat('(',num2str(roundn(ac,-3)),',',num2str(roundn(nmi,-3)),...
                        ',',num2str(roundn(pur,-3)),',',num2str(roundn(ar,-3)),...
                        ',',num2str(roundn(fs,-3)),',',num2str(roundn(p,-3)),...
                        ',',num2str(roundn(r,-3)),')'));
                    
                end
                disp(['alpha:1e',num2str(log10(options.alpha))]);
                disp(['beta:1e',num2str(log10(options.beta))]);
                disp(strcat('Final results of',[' ',Dataname],':'));
                disp(strcat('(',num2str(roundn(mean(ACC)*100,-2)),...
                    ',',num2str(roundn(mean(NMI)*100,-2)),...
                    ',',num2str(roundn(mean(PUR)*100,-2)),...
                    ',',num2str(roundn(mean(ARI)*100,-2)),...
                    ',',num2str(roundn(mean(Fscore)*100,-2)),...
                    ',',num2str(roundn(mean(Precision)*100,-2)),...
                    ',',num2str(roundn(mean(Recall)*100,-2)),')'));
                LWNdimNMF_NNorm{ialpha,ibeta} = ...
                    [roundn(mean(ACC)*100,-2),roundn(mean(NMI)*100,-2),...
                    roundn(mean(PUR)*100,-2),roundn(mean(ARI)*100,-2);
                    roundn(std(ACC)*100,-2),roundn(std(NMI)*100,-2),...
                    roundn(std(PUR)*100,-2),roundn(std(ARI)*100,-2)];
                clear ACC NMI PUR ARI Fscore Precision Recall
            end
        end
        root = ['F:\我的工作空间\11-imcomplete-mv-deepIMC\',...
            'Figures\results_of_different_layers_of_3variants_init_nmf'];
        path = [root,'\LWNdimNMF_NNorm\',Dataname];
        if ~exist(path,'dir')
            mkdir(path);
        end
        % {
        if length(options.layers)==1
            save(strcat(path,'\',...% one layer
                num2str(percent),'_',num2str(nlayer),'layer',...
                '_k=',num2str(options.k),...
                '_[',num2str(log10(options.alpha)),',',...
                num2str(log10(options.beta)),']',...
                '_[',num2str(options.layers(1)),'].mat'),'LWNdimNMF_NNorm');
        elseif length(options.layers)==2
            save(strcat(path,'\',...% two
                num2str(percent),'_',num2str(nlayer),'layer',...
                '_k=',num2str(options.k),...
                '_[',num2str(log10(options.alpha)),',',...
                num2str(log10(options.beta)),']',...
                '_[',num2str(options.layers(1)),',',...
                num2str(options.layers(2)),'].mat'),'LWNdimNMF_NNorm');
        elseif length(options.layers)==3
            save(strcat(path,'\',...% three
                num2str(percent),'_',num2str(nlayer),'layer',...
                '_k=',num2str(options.k),...
                '_[',num2str(log10(options.alpha)),',',...
                num2str(log10(options.beta)),']',...
                '_[',num2str(options.layers(1)),...
                ',',num2str(options.layers(2)),...
                ',',num2str(options.layers(3)),'].mat'),'LWNdimNMF_NNorm');
        end
        %}
        end
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
