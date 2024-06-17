% dbstop if error

clear;
clc
dataname_list = {'yalea'};% wisconsin msrcv1
for idata = 1:length(dataname_list)
    % Dataname = 'caltech6v';
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
                get_paras_3layers_deepnmf_init_nmf(Dataname,percent,nlayer);
            options.k = knn;% 8 6 4 10
            alpha_list = [alpha];% 1e-2
            beta_list = [beta];% 1e-1
            options.layers = layers;
            options.max_iter = 300;
            num_folds = 15;% length(folds)
            repeat = 5;% before date:2020.12.20 is set as 10.
            
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
                        
                        [W,H,Ht,Hc] = DeepNMF_warped_options(X,gnd,G,options);
                        
                        U = Ht';
                        U(isnan(U)) = 0;
                        U(isinf(U)) = 1e5;
                        new_F = U;
                        % {
                        norm_mat = repmat(sqrt(sum(new_F.*new_F,2)),1,size(new_F,2));
                        %%avoid divide by zero
                        for i = 1:size(norm_mat,1)
                            if (norm_mat(i,1)==0)
                                norm_mat(i,:) = 1;
                            end
                        end
                        new_F = new_F./norm_mat;
                        %}
                        for iter_c = 1:repeat
                            pre_labels    = kmeans(new_F,numClust,'emptyaction','singleton',...
                                'replicates',20,'display','off',...
                                'Distance','sqeuclidean');
                            result_LatLRR = ClusteringMeasure(gnd, pre_labels);
                            Purity(iter_c)= result_LatLRR(3);
                            [AC(iter_c),MI(iter_c),~] = result(pre_labels,gnd);
                            [AR(iter_c),~,~,~] = RandIndex(gnd,pre_labels);
                            [Fs(iter_c),Pre(iter_c),Rec(iter_c)] = compute_f(gnd,pre_labels);
                        end
                        
                        ac = mean(AC);
                        nmi = mean(MI);
                        pur = mean(Purity);
                        ar = mean(AR);
                        fs = mean(Fs);
                        p = mean(Pre);
                        rec = mean(Rec);
                        clear AC MI Purity AR Fs Pre Rec
                        
                        ACC = [ACC,ac];
                        NMI = [NMI,nmi];
                        PUR = [PUR,pur];
                        ARI = [ARI,ar];
                        Fscore = [Fscore, fs];
                        Precision = [Precision, p];
                        Recall = [Recall, rec];
                        
                        disp(strcat('(',num2str(roundn(ac,-3)),',',num2str(roundn(nmi,-3)),...
                            ',',num2str(roundn(pur,-3)),',',num2str(roundn(ar,-3)),...
                            ',',num2str(roundn(fs,-3)),',',num2str(roundn(p,-3)),...
                            ',',num2str(roundn(rec,-3)),')'));
                        
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
                    res_DeepNMF{ialpha,ibeta} = ...
                        [roundn(mean(ACC)*100,-2),roundn(mean(NMI)*100,-2),...
                        roundn(mean(PUR)*100,-2),roundn(mean(ARI)*100,-2);
                        roundn(std(ACC)*100,-2),roundn(std(NMI)*100,-2),...
                        roundn(std(PUR)*100,-2),roundn(std(ARI)*100,-2)];
                    clear ACC NMI PUR ARI Fscore Precision Recall
                end
            end
            root = ['D:\my-work-space\11-imcomplete-mv-deepIMC\',...
                'Figures\results_of_different_layers_of_3variants_init_nmf'];
            path = [root,'\DeepNMF\',Dataname];
            if ~exist(path,'dir')
                mkdir(path);
            end
%             if length(options.layers)==1
%                 save(strcat(path,'\',...% one layer
%                     num2str(percent),'_',num2str(nlayer),'layer',...
%                     '_k=',num2str(options.k),...
%                     '_[',num2str(log10(options.alpha)),',',...
%                     num2str(log10(options.beta)),']',...
%                     '_[',num2str(options.layers(1)),'].mat'),'res_DeepNMF');
%             elseif length(options.layers)==2
%                 save(strcat(path,'\',...% two
%                     num2str(percent),'_',num2str(nlayer),'layer',...
%                     '_k=',num2str(options.k),...
%                     '_[',num2str(log10(options.alpha)),',',...
%                     num2str(log10(options.beta)),']',...
%                     '_[',num2str(options.layers(1)),',',...
%                     num2str(options.layers(2)),'].mat'),'res_DeepNMF');
%             elseif length(options.layers)==3
%                 save(strcat(path,'\',...% three
%                     num2str(percent),'_',num2str(nlayer),'layer',...
%                     '_k=',num2str(options.k),...
%                     '_[',num2str(log10(options.alpha)),',',...
%                     num2str(log10(options.beta)),']',...
%                     '_[',num2str(options.layers(1)),...
%                     ',',num2str(options.layers(2)),...
%                     ',',num2str(options.layers(3)),'].mat'),'res_DeepNMF');
%             end
        end
    end
end

function [Datafold,Data] = getData(Dataname,percent)

if strcmp(Dataname,'msrcv1')||...
        strcmp(Dataname,'yalea')
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

