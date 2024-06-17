function [fea,gnd] = ld_image_mv_dataset(dataset)
% output fea: cell type {n*m1,n*m2,...,n*mk}
% output gnd: n*1
%
% 'bbcsport': 2 views 5 classes (imbalance class)
% 'caltech': 6 views 7 classes (imbalance class) 
%            (view 2 have few(25) negative values)
% 'handwritten': 6 views 10 classes (even class) 
%            (label:0~9) (half of view 5 are negative values)
% 'orl': 3 views 40 classes (even class)
% 'webkb': 2 views 2 classes (imbalance class (230:821))
% 'yaleb': 3 views 10 classes (even class)

gt = [];
%% set root path
data_root = 'D:\my-work-space\new_dataset\mv_data\';
% for i = 1:length(unique(Y))
%     disp(sum(Y==i))
% end
%% load dataset 
switch lower(dataset)
    case 'stl10' % 3 views 1024 512 2048 
        load(['D:\my-work-space\new_dataset\CVPR22\',...
            'stl10\stl10_fea.mat']);
        gnd = Y;
        fea = X;
    case 'mnist1w' % 10,000 3 views 30 9 30 
        load(['D:\my-work-space\new_dataset\2022-TKDE-EMKMC'...
            '\mnist.mat']);
        gnd = Y;
        fea = X;
    case 'mnist4' % 4,000 3 views 30 9 30 
        load(['D:\my-work-space\new_dataset\2022-TKDE-EMKMC'...
            '\mnist4.mat']);
        gnd = Y;
        fea = X;
    case 'caltechall6v' % 9144 34,436,304
        dir = ['D:\my-work-space\new_dataset\mv_data\Caltech\'];
        load(strcat(dir,'Caltech101-all.mat'));
        fea{1} = X{1}; % 48 gabor
        fea{2} = X{2}; % 40 wavelet have negative values.
        fea{3} = X{3}; % 254 centrist
        fea{4} = X{4}; % 1984 hog
        fea{5} = X{5}; % 512 gist
        fea{6} = X{6}; % 928 lbp
        gnd = Y;
    case 'webkb2'% 2 views 1840 3000
        load(['D:\my-work-space\new_dataset\',...
            '2022-TKDE-EMKMC\WebKB.mat']);
        gnd = Y;
        fea = X;
    case 'coil20'% 3 views 30 19 30 
        load(['D:\my-work-space\new_dataset\',...
            '2022-TKDE-EMKMC\COIL20.mat']);
        gnd = Y;
        fea{1} = X{1}';
        fea{2} = X{2}';
        fea{3} = X{3}';
    case 'ucidigit'% 3 views 216 76 64 
        load(['D:\my-work-space\new_dataset\',...
            'DownloadFromGitee\uci-digit.mat']);
        gnd = truth;
        fea{1} = mfeat_fac;
        fea{2} = mfeat_fou;
        fea{3} = mfeat_kar;
    case 'handwritten' % 6 views use two views
%         load(strcat(data_root,'handwritten.mat'));
        % 'gnd','fourier'76,'zer'47,'mor'6,'profile'216,'pixel'240
        load(strcat(data_root,'handwritten.mat'));
        gnd = gnd + 1;% convert 0~9 to 1~10
        nClas = length(unique(gnd));
        fea{1} = pixel;
        fea{2} = fourier; 
    case 'fmnist3w' % 30,000
        load(['D:\my-work-space\new_dataset\CVPR22\fmnist\',...
            'fmnist3w.mat'],'');
        fea{1} = data{1}';
        fea{2} = data{2}'; 
        gnd = gnd;
    case 'mnist3w' % 30,000
        load(['D:\my-work-space\new_dataset\CVPR22\MNIST\',...
            'mnist3w.mat'],'');
        fea{1} = data{1}';
        fea{2} = data{2}'; 
        gnd = gnd;
    case 'youtube4w' % 40,021
        load(['D:\my-work-space\new_dataset\CVPR22\YoutubeFace\',...
            'youtube4w.mat'],'');
        fea{1} = data{1}';
        fea{2} = data{2}'; 
        gnd = gnd;
    case 'reuters2v' % 18,758
        load(['D:\my-work-space\new_dataset\CVPR22\Reuters\',...
            'Reuters2v.mat'],'');
        fea{1} = X{1};
        fea{2} = X{2}; 
        gnd = Y+1;
    case 'cornellrnsp'% 2 views 1840 3000
        load(['D:\my-work-space\new_dataset\mv_data\',...
            'cornellRnSp.mat']);
        gnd = truth;
        Xf1=readsparse(X1);
        X{1} = Xf1';
        Xf2=readsparse(X2);
        X{2} = Xf2';
        fea = X;
    case 'yale' % 3 views
        load(strcat(data_root,'fea_extr_func\Gabor\mv_yale_15.mat'));
        X{1} = pixel;X{2} = gabr;X{3} = lbp;
        n_view = length(X);
        nClas = length(unique(gnd));
        fea = X;
    case 'yalebmy' % 3 views
        load(strcat(data_root,'fea_extr_func\Gabor\mv_yaleb_8.mat'));
        X{1} = pixel;X{2} = gabr;X{3} = lbp;
        n_view = length(X);
        nClas = length(unique(gnd));
        fea = X;
    case 'yalebmy12' % 3 views
        load(strcat(data_root,'fea_extr_func\Gabor\mv_yaleb_12.mat'));
        X{1} = pixel;X{2} = gabr;X{3} = lbp;
        n_view = length(X);
        nClas = length(unique(gnd));
        fea = X;
    case 'orl30' % 3 views 
        load(strcat(data_root,'fea_extr_func\Gabor\mv_orl_30.mat'));
%         load(strcat(data_root,'fea_extr_func\Gabor\3scales-4orients\mv_orl_30.mat'));
        X{1} = pixel;X{2} = gabr;X{3} = lbp;
        n_view = length(X);
        nClas = length(unique(gnd));
        fea = X;
    case 'orl40' % 3 views
        load(strcat(data_root,'fea_extr_func\Gabor\mv_orl_40.mat'));
%         load(strcat(data_root,'fea_extr_func\Gabor\3scales-4orients\mv_orl_40.mat'));
        X{1} = pixel;X{2} = gabr;X{3} = lbp;
        n_view = length(X);
        nClas = length(unique(gnd));
        fea = X;
    case 'fei35' % 3 views
        load(strcat(data_root,'fea_extr_func\Gabor\mv_fei_35.mat'));
        X{1} = pixel;X{2} = gabr;X{3} = lbp;
        n_view = length(X);
        nClas = length(unique(gnd));
        fea = X;
    case 'fei30' % 3 views
        load(strcat(data_root,'fea_extr_func\Gabor\mv_fei_30.mat'));
        X{1} = pixel;X{2} = gabr;X{3} = lbp;
        n_view = length(X);
        nClas = length(unique(gnd));
        fea = X;
    case 'fei50' % 3 views
        load(strcat(data_root,'fea_extr_func\Gabor\mv_fei_50.mat'));
        X{1} = pixel;X{2} = gabr;X{3} = lbp;
        n_view = length(X);
        nClas = length(unique(gnd));
        fea = X;
    case 'umist15' % 3 views
        load(strcat(data_root,'fea_extr_func\Gabor\mv_umist_15.mat'));
%         load(strcat(data_root,'fea_extr_func\Gabor\3scales-4orients\mv_umist_15.mat'));
        X{1} = pixel;X{2} = gabr;X{3} = lbp;
        n_view = length(X);
        nClas = length(unique(gnd));
        fea = X;
    case 'umist20' % 3 views
        load(strcat(data_root,'fea_extr_func\Gabor\mv_umist_20.mat'));
%         load(strcat(data_root,'fea_extr_func\Gabor\3scales-4orients\mv_umist_20.mat'));
        X{1} = pixel;X{2} = gabr;X{3} = lbp;
        n_view = length(X);
        nClas = length(unique(gnd));
        fea = X;
    case 'pie68'
%         load(strcat(data_root,'fea_extr_func\Gabor\mv_pie_8.mat'));
        load(strcat(data_root,'fea_extr_func\Gabor\mv_pie_68.mat'));
        X{1} = pixel;X{2} = gabr;X{3} = lbp;
        n_view = length(X);
        nClas = length(unique(gnd));
        fea = X;
        for i = 1:length(unique(gnd))
            ind_dump_t = find(gnd==i);
            ind_dump = ind_dump_t(11:end);
            gnd(ind_dump) = [];
            fea{1}(ind_dump,:) = [];fea{2}(ind_dump,:) = [];fea{3}(ind_dump,:) = [];
            %fea{4}(ind_dump,:) = [];fea{5}(ind_dump,:) = [];fea{6}(ind_dump,:) = [];
        end
    case 'coil1008'
%         load(strcat(data_root,'fea_extr_func\Gabor\mv_coil100_10.mat'));
        load(strcat(data_root,'fea_extr_func\Gabor\mv_coil100_8.mat'));
        X{1} = pixel;X{2} = gabr;X{3} = lbp;
        n_view = length(X);
        nClas = length(unique(gnd));
        fea = X;
    case 'bbcsport' % 2 views 
        load(strcat(data_root,'bbcsport.mat'));
        n_view = length(X);
        gnd = Y;
        nClas = length(unique(gnd));
        fea = X;
    case 'caltech2v' % last 2 views
        load(strcat(data_root,'Caltech101-7.mat'));
        X{2} = [];% delete 1474x40, view 2 has negative values
        X{1} = [];X{3} = [];X{4} = [];
        X(cellfun(@isempty,X))=[];
        n_view = length(X);
        gnd = Y;
%         nClas = length(unique(gnd));
        fea = X;
%         for i = 1:2
%             ind_dump_t = find(gnd==i);
%             ind_dump = ind_dump_t(161:end);
%             gnd(ind_dump) = [];
%             fea{1}(ind_dump,:) = [];fea{2}(ind_dump,:) = [];fea{3}(ind_dump,:) = [];
%             %fea{4}(ind_dump,:) = [];fea{5}(ind_dump,:) = [];fea{6}(ind_dump,:) = [];
%         end
        % samples of each class: lenSmp
        % name of each class(cell type): categories(cateset)
        % name of features(cell type): feanames
    case 'caltech6v' % all 6 views
        load(strcat(data_root,'Caltech101-7.mat'));
        % view 2 has negative values
        gnd = Y;
        fea = X;
    case 'caltechall4v' % 9144 7,808,976
        dir = ['D:\my-work-space\new_dataset\mv_data\Caltech\'];
        load(strcat(dir,'Caltech101-all.mat'));
        fea{1} = X{1}; % 48 gabor
        fea{2} = X{2}; % 40 wavelet have negative values.
        fea{3} = X{3}; % 254 centrist
        fea{4} = X{4}; % 512 gist
        gnd = Y;
    case 'nus' % 30,000 19,050,000
        dir = ['D:\my-work-space\new_dataset\TKDE-R1\NUSWIDE\',...
            'NUS-WIDE-OBJECT\low level features\'];
        load(strcat(dir,'NUS.mat'));
        fea{1} = CH; % 64 gabor have negative values.
        fea{2} = CM55; % 225 wavelet have negative values.
        fea{3} = CORR; % 144 have negative values.
        fea{4} = EDH; % 74 have negative values.
        fea{5} = WT; % 128 have negative values.
        gnd = gnd;
    case 'nus6v' % 2,400 64, 144, 73, 128, 225, 500
        dir = 'D:\my-work-space\new_dataset\2022-TKDE-EMKMC\';
        load(strcat(dir,'NUS.mat'));
        fea = X;
        gnd = Y;
    case 'nus4k' % 4,030 2,559,050
        dir = ['D:\my-work-space\new_dataset\TKDE-R1\NUSWIDE\',...
            'NUS-WIDE-OBJECT\low level features\'];
        load(strcat(dir,'NUS.mat'));
        fea{1} = CH; % 64 gabor have negative values.
        fea{2} = CM55; % 225 wavelet have negative values.
        fea{3} = CORR; % 144 have negative values.
        fea{4} = EDH; % 74 have negative values.
        fea{5} = WT; % 128 have negative values.
        gnd = gnd;
        for i = 1:31
            ind_dump_t = find(gnd==i);
            ind_dump = ind_dump_t(131:end);
            gnd(ind_dump) = [];
            fea{1}(ind_dump,:) = [];
            fea{2}(ind_dump,:) = [];
            fea{3}(ind_dump,:) = [];
            fea{4}(ind_dump,:) = [];
            fea{5}(ind_dump,:) = [];
        end
    case 'nus4k4v' % 4,030 2,559,050
        dir = ['D:\my-work-space\new_dataset\TKDE-R1\NUSWIDE\',...
            'NUS-WIDE-OBJECT\low level features\'];
        load(strcat(dir,'NUS.mat'));
        fea{1} = CH; % 64 have negative values.
        fea{2} = CM55; % 225 have negative values.
        fea{3} = CORR; % 144 have negative values.
        fea{4} = EDH; % 74 have negative values.
%         fea{5} = WT; % 128 have negative values.
        gnd = gnd;
        for i = 1:31
            ind_dump_t = find(gnd==i);
            ind_dump = ind_dump_t(131:end);
            gnd(ind_dump) = [];
            fea{1}(ind_dump,:) = [];
            fea{2}(ind_dump,:) = [];
            fea{3}(ind_dump,:) = [];
            fea{4}(ind_dump,:) = [];
%             fea{5}(ind_dump,:) = [];
        end
    case 'nus4kpos' % 4,030 2,559,050
        dir = ['D:\my-work-space\new_dataset\TKDE-R1\NUSWIDE\',...
            'NUS-WIDE-OBJECT\low level features\'];
        load(strcat(dir,'NUS.mat'));
        fea{1} = CH; % 64 have negative values.
        fea{2} = CM55; % 225 have negative values.
        fea{3} = CORR; % 144 have negative values.
        fea{4} = EDH; % 74 have negative values.
        fea{5} = WT; % 128 have negative values.
        gnd = gnd;
        for i = 1:31
            ind_dump_t = find(gnd==i);
            ind_dump = ind_dump_t(131:end);
            gnd(ind_dump) = [];
            fea{1}(ind_dump,:) = [];
            fea{2}(ind_dump,:) = [];
            fea{3}(ind_dump,:) = [];
            fea{4}(ind_dump,:) = [];
            fea{5}(ind_dump,:) = [];
        end
    case 'nus3vcce' % 30,000 8,460,000
        dir = ['D:\my-work-space\new_dataset\TKDE-R1\NUSWIDE\',...
            'NUS-WIDE-OBJECT\low level features\'];
        load(strcat(dir,'NUS.mat'));
        fea{1} = CH; % 64 gabor have negative values.
%         fea{2} = CM55; % 225 wavelet have negative values.
        fea{2} = CORR; % 144 have negative values.
        fea{3} = EDH; % 74 have negative values.
%         fea{5} = WT; % 128 have negative values.
        gnd = gnd;
    case 'handwritten5v' % 6 views use two views
        load(strcat(data_root,'handwritten.mat'));
        % 'gnd','fourier'76,'zer'47,'mor'6,'profile'216,'pixel'240
%         load(strcat(data_root,'handwritten_each_40.mat'));
        gnd = gnd + 1;% convert 0~9 to 1~10
        nClas = length(unique(gnd));
        fea{1} = pixel;
        fea{2} = fourier;
        fea{3} = mor; 
        fea{4} = zer;
        fea{5} = profilee;      
%{        
        ind_base = randperm(200,40);% 1~200 40 unique nums
        fea_1 = [];fea_2 = [];fea_3 = [];fea_4 = [];fea_5 = [];
        gnd_40 = [];
        for i = 1:nClas
            ind = ind_base + (i-1)*200;
            fea_1 = [fea_1;fourier(ind,:)];
            fea_2 = [fea_2;zer(ind,:)];
            fea_3 = [fea_3;mor(ind,:)];
            fea_4 = [fea_4;profile(ind,:)];
            fea_5 = [fea_5;pixel(ind,:)];
            gnd_40 = [gnd_40;gnd(ind,:)];
        end
        fea{1} = fea_1;fea{2} = fea_2;fea{3} = fea_3;
        fea{4} = fea_4;fea{5} = fea_5;
        delete gnd fourier zer mor profile pixel
        gnd = gnd_40;
        fourier = fea{1};zer = fea{2};mor = fea{3};
        profile = fea{4};pixel = fea{5};
        save(strcat(data_root,'handwritten_each_40.mat'),...
            'gnd','fourier','zer','mor','profile','pixel');
        %}
    case 'orl'% 3 views
        load(strcat(data_root,'ORL_mtv.mat'));
        n_view = length(X);
        gnd = gt;
        nClas = length(unique(gnd));
        fea = cell(1,n_view);
        fea{1} = X{1}';
        fea{2} = X{2}';
        fea{3} = X{3}';
    case 'webkb'% 2 views
        load(strcat(data_root,'WebKB.mat'));
        n_view = length(X);
        gnd = gnd;
        nClas = length(unique(gnd));
        fea = X;
        for i = 2 
            ind_dump_t = find(gnd==i);
            ind_dump = ind_dump_t(241:end);
            gnd(ind_dump) = [];
            fea{1}(ind_dump,:) = [];fea{2}(ind_dump,:) = [];
        end
    case 'yaleb' % 3 views
        load(strcat(data_root,'YaleB_first10.mat'));
        n_view = 3;
        gnd = gt;
        nClas = length(unique(gnd));
        X = cell(1,3);
        X{1} = X1';
        X{2} = X2';
        X{3} = X3';
        fea = X;
    case 'ecg' % 2 views
        dir = 'D:\my-work-space\new_dataset\physionet_ECG_data\ECGData\';
        load(strcat(dir,'ECGData_294.mat'));
        fea{1} = time;
        fea{2} = fft_coefs;
    case 'bankuai' % 2 views x classes each 17 features
        dir = 'D:\my-work-space\new_dataset\mv_data\bankuai\';
        load(strcat(dir,'bankuai.mat'));
        fea{1} = X{1};
        fea{2} = X{2};
    case 'bankuai3v' % 3 views x classes
        dir = 'D:\my-work-space\new_dataset\mv_data\bankuai\';
        load(strcat(dir,'bankuai3v.mat'));
        fea{1} = X{1};
        fea{2} = X{2};
        fea{3} = X{3};
    case 'bankuai3v_2of3' % 3 views x classes each 34*2/3 features
        dir = 'F:\my-work-space\new_dataset\mv_data\bankuai\';
        load(strcat(dir,'bankuai3v_2of3.mat'));
        fea{1} = X{1};
        fea{2} = X{2};
        fea{3} = X{3};
    case 'skin100' % 4 views x classes 700 7 classes 
        dir = ['D:\my-work-space\new_dataset\mv_data\fea_extr_func\',...
            'Gabor\2scales-4orients\'];
        load(strcat(dir,'mv_skin100_7.mat'));
        fea{1} = pixel;
        fea{2} = gabr;
        fea{3} = lbp;
        fea{4} = grayhist;
%         for i = 1:7
%             ind_dump_t = find(gnd==i);
%             ind_dump = ind_dump_t(41:end);
%             gnd(ind_dump) = [];
%             fea{1}(ind_dump,:) = [];
%             fea{2}(ind_dump,:) = [];
%             fea{3}(ind_dump,:) = [];
%             fea{4}(ind_dump,:) = [];
%         end
    case 'skin1005v' % 5 views x classes 700 7 classes 
        dir = ['F:\my-work-space\new_dataset\mv_data\fea_extr_func\',...
            'Gabor\2scales-4orients\'];
        load(strcat(dir,'mv_skin1005v_7.mat'));
        fea{1} = pixel;
        fea{2} = gabr;
        fea{3} = lbp;
        fea{4} = grayhist;
        fea{5} = hog;
        for i = 1:7
            ind_dump_t = find(gnd==i);
            ind_dump = ind_dump_t(41:end);
            gnd(ind_dump) = [];
            fea{1}(ind_dump,:) = [];
            fea{2}(ind_dump,:) = [];
            fea{3}(ind_dump,:) = [];
            fea{4}(ind_dump,:) = [];
            fea{5}(ind_dump,:) = [];
        end
    case 'test100' % 4 views x classes 400 4 classes
        dir = ['F:\my-work-space\new_dataset\mv_data\fea_extr_func\',...
            'Gabor\2scales-4orients\'];
        load(strcat(dir,'mv_test100_4.mat'));
        fea{1} = pixel;
        fea{2} = gabr;
        fea{3} = lbp;
        fea{4} = grayhist;
        for i = 1:4
            ind_dump_t = find(gnd==i);
            ind_dump = ind_dump_t(51:end);
            gnd(ind_dump) = [];
            fea{1}(ind_dump,:) = [];
            fea{2}(ind_dump,:) = [];
            fea{3}(ind_dump,:) = [];
            fea{4}(ind_dump,:) = [];
        end
    case 'test1002v' % 4 views x classes 400 4 classes
        dir = ['F:\my-work-space\new_dataset\mv_data\Wang\test100\'];
        load(strcat(dir,'test1002v.mat'));
        fea{1} = edgeFea;
        fea{2} = im2double(pixelFea);
    case 'skin1002v' % 4 views x classes 700 7 classes
        dir = ['F:\my-work-space\new_dataset\mv_data\Wang\skin100\'];
        load(strcat(dir,'skin1002v.mat'));
        fea{1} = edgeFea/1.;
        fea{2} = im2double(pixelFea);
    case 'test1005v' % 5 views x classes 400 4 classes
        dir = ['F:\my-work-space\new_dataset\mv_data\fea_extr_func\',...
            'Gabor\2scales-4orients\'];
        load(strcat(dir,'mv_test1005v_4.mat'));
        fea{1} = pixel;
        fea{2} = gabr;
        fea{3} = lbp;
        fea{4} = grayhist;
        fea{5} = hog;
        for i = 1:4
            ind_dump_t = find(gnd==i);
            ind_dump = ind_dump_t(51:end);
            gnd(ind_dump) = [];
            fea{1}(ind_dump,:) = [];
            fea{2}(ind_dump,:) = [];
            fea{3}(ind_dump,:) = [];
            fea{4}(ind_dump,:) = [];
            fea{5}(ind_dump,:) = [];
        end
    case 'msrcv1' %  
        dir = ['D:\my-work-space\new_dataset\mv_data\',...
                'multi-view-dataset\'];
        load(strcat(dir,'MSRCV1.mat'));
        fea{1} = X{1};
        fea{2} = X{2};
        fea{3} = X{3};
        fea{4} = X{4};
        fea{5} = X{5};
        fea{6} = X{6};
        gnd = Y;
    case 'wiki' %  wiki 2866 128 10 
        dir = ['D:\my-work-space\new_dataset\mv_data\',...
                'multi-view-dataset\'];
        load(strcat(dir,'Wiki_fea.mat'));
        fea{1} = X{1};
        fea{2} = X{2};
        gnd = Y;
    case 'wikipa' %  WikipediaArticles
        dir = ['D:\my-work-space\new_dataset\mv_data\',...
                'multi-view-dataset\'];
        load(strcat(dir,'WikipediaArticles.mat'));
        fea{1} = X{1};
        fea{2} = X{2};
        gnd = Y;
    case 'yalea' %  yaleA
        dir = ['F:\my-work-space\new_dataset\mv_data\',...
                'multi-view-dataset\'];
        load(strcat(dir,'yaleA_3view.mat'));
        fea{1} = X{1};
        fea{2} = X{2};
        fea{3} = X{3};
        gnd = Y;
    case 'adni3' %  adni3
        dir = ['F:\my-work-space\new_dataset\mv_data\ANDI\'];
        load(strcat(dir,'adni3.mat'));
        fea{1} = dtisc;
        fea{2} = fmrisc;
        gnd = gnd';
    case 'adni3_9v' %  9 views
        dir = ['F:\my-work-space\new_dataset\mv_data\ANDI\'];
        load(strcat(dir,'adni3_16v.mat'));
        fea{1} = dtisc;
        fea{2} = fmrisc;
        fea{3} = gabor13;
        fea{4} = glcm;
        fea{5} = hog378;
        fea{6} = lbp18;
        fea{7} = orb1000;
        fea{8} = surf2k;
        fea{9} = sift2k;
        gnd = gnd';
    case 'adni3_6v' %  6 views
        dir = ['F:\my-work-space\new_dataset\mv_data\ANDI\'];
        load(strcat(dir,'adni3_16v.mat'));
        fea{1} = dtisc;
        fea{2} = fmrisc;
        fea{3} = gabor13;
        fea{4} = glcm;
%         fea{x} = hog378;
        fea{5} = lbp18;
%         fea{x} = orb1000;
%         fea{x} = surf2k;
        fea{6} = sift2k;
        gnd = gnd';
    case 'adni3_4v' %  4 views
        dir = ['D:\my-work-space\new_dataset\mv_data\ANDI\'];
        load(strcat(dir,'adni3_16v.mat'));
%         fea{x} = dtisc;
%         fea{x} = fmrisc;
        fea{1} = gabor13;
        fea{2} = glcm;
%         fea{x} = hog378;
        fea{3} = lbp18;
%         fea{x} = orb1000;
%         fea{x} = surf2k;
        fea{4} = sift2k;
        gnd = gnd';
    case 'adni3_5v' %  5 views
        dir = ['D:\my-work-space\new_dataset\mv_data\ANDI\'];
        load(strcat(dir,'adni3_16v.mat'));
        fea{1} = dtisc;
%         fea{x} = fmrisc;
        fea{2} = gabor13;
        fea{3} = glcm;
%         fea{x} = hog378;
        fea{4} = lbp18;
%         fea{x} = orb1000;
%         fea{x} = surf2k;
        fea{5} = sift2k;
        gnd = gnd';
    case 'adni3_6v2' %  6 views v2.0
        dir = ['F:\my-work-space\new_dataset\mv_data\ANDI\'];
        load(strcat(dir,'adni3_16v.mat'));
        fea{1} = dtisc;
%         fea{x} = fmrisc;
        fea{2} = gabor13;
        fea{3} = glcm;
        fea{4} = hog378;
        fea{5} = lbp18;
%         fea{x} = orb1000;
%         fea{x} = surf2k;
        fea{6} = sift2k;
        gnd = gnd';
    case 'adni3_5v2' %  5 views v2.0
        dir = ['F:\my-work-space\new_dataset\mv_data\ANDI\'];
        load(strcat(dir,'adni3_16v.mat'));
        fea{1} = dtisc;
%         fea{x} = fmrisc;
        fea{2} = gabor13;
        fea{3} = glcm;
%         fea{x} = hog378;
        fea{4} = lbp18;
        fea{5} = orb2000;
%         fea{x} = surf2k;
%         fea{x} = sift2k;
        gnd = gnd';
    case 'adni3_5v3' %  5 views v3.0
        dir = ['F:\my-work-space\new_dataset\mv_data\ANDI\'];
        load(strcat(dir,'adni3_16v.mat'));
        fea{1} = dtisc;
%         fea{x} = fmrisc;
        fea{2} = gabor13;
        fea{3} = glcm;
%         fea{x} = hog378;
        fea{4} = lbp18;
%         fea{x} = orb2000;
        fea{5} = surf2k;
%         fea{x} = sift2k;
        gnd = gnd';
    case 'adni3_7v' %  7 views
        dir = ['F:\my-work-space\new_dataset\mv_data\ANDI\'];
        load(strcat(dir,'adni3_16v.mat'));
        fea{1} = dtisc;
%         fea{x} = fmrisc;
        fea{2} = gabor13;
        fea{3} = glcm;
%         fea{x} = hog378;
        fea{4} = lbp18;
        fea{5} = orb2000;
        fea{6} = surf2k;
        fea{7} = sift2k;
        gnd = gnd';
    case 'adni3_4v2' %  4 views v2.0
        dir = ['F:\my-work-space\new_dataset\mv_data\ANDI\'];
        load(strcat(dir,'adni3_16v.mat'));
        fea{1} = dtisc;
%         fea{x} = fmrisc;
        fea{2} = gabor13;
        fea{3} = glcm;
%         fea{x} = hog378;
        fea{4} = lbp18;
%         fea{x} = orb2000;
%         fea{x} = surf2k;
        fea{x} = sift2k;
        gnd = gnd';
    case 'adni3_4v3' %  4 views v3.0
        dir = ['F:\my-work-space\new_dataset\mv_data\ANDI\'];
        load(strcat(dir,'adni3_16v.mat'));
        fea{1} = dtisc;
%         fea{x} = fmrisc;
        fea{2} = gabor13;
%         fea{x} = glcm;
%         fea{x} = hog378;
        fea{3} = lbp18;
%         fea{x} = orb2000;
%         fea{x} = surf2k;
        fea{4} = sift2k;
        gnd = gnd';
    case 'adni3_5v4' %  5 views v4.0
        dir = ['F:\my-work-space\new_dataset\mv_data\ANDI\'];
        load(strcat(dir,'adni3_16v.mat'));
        fea{1} = dtisc;
%         fea{x} = fmrisc;
        fea{2} = gabor13;
        fea{3} = glcm;
%         fea{x} = hog378;
        fea{4} = lbp24;
%         fea{x} = orb2000;
%         fea{x} = surf2k;
        fea{5} = sift2k;
        gnd = gnd';
    case 'adni3_5v5' %  5 views v5.0
        dir = ['F:\my-work-space\new_dataset\mv_data\ANDI\'];
        load(strcat(dir,'adni3_16v.mat'));
        fea{1} = dtisc;
%         fea{x} = fmrisc;
        fea{2} = gabor13;
        fea{3} = glcm;
%         fea{x} = hog378;
        fea{4} = lbp34;
%         fea{x} = orb2000;
%         fea{x} = surf2k;
        fea{5} = sift2k;
        gnd = gnd';
    case 'adni3_6v3' %  6 views v3.0
        dir = ['F:\my-work-space\new_dataset\mv_data\ANDI\'];
        load(strcat(dir,'adni3_16v.mat'));
        fea{1} = dtisc;
%         fea{x} = fmrisc;
        fea{2} = gabor9;
        fea{3} = gabor13;
        fea{4} = glcm;
%         fea{x} = hog378;
        fea{5} = lbp18;
%         fea{x} = orb2000;
%         fea{x} = surf2k;
        fea{6} = sift2k;
        gnd = gnd';
    case 'adni3_5v6' %  5 views v6.0
        dir = ['F:\my-work-space\new_dataset\mv_data\ANDI\'];
        load(strcat(dir,'adni3_16v.mat'));
        fea{1} = dtisc;
%         fea{x} = fmrisc;
        fea{2} = gabor9;
        fea{3} = glcm;
%         fea{x} = hog378;
        fea{4} = lbp18;
%         fea{x} = orb2000;
%         fea{x} = surf2k;
        fea{5} = sift2k;
        gnd = gnd';
    case 'adni3_5v7' %  5 views v7.0
        dir = ['F:\my-work-space\new_dataset\mv_data\ANDI\'];
        load(strcat(dir,'adni3_16v.mat'));
        fea{1} = dtisc;
%         fea{x} = fmrisc;
        fea{2} = gabor9;
        fea{3} = glcm;
%         fea{x} = hog378;
        fea{4} = lbp18;
        fea{5} = orb500;
%         fea{x} = surf2k;
%         fea{x} = sift2k;
        gnd = gnd';
    case 'adni3_6v4' %  6 views v4.0
        dir = ['F:\my-work-space\new_dataset\mv_data\ANDI\'];
        load(strcat(dir,'adni3_23v.mat'));
        fea{1} = dtisc;
%         fea{x} = fmrisc;
        fea{2} = gabor9;
        fea{3} = glcm;
%         fea{x} = hog378;
        fea{4} = lbp18;
%         fea{x} = orb500;
%         fea{x} = surf2k;
        fea{5} = sift2k;
        fea{6} = kaze2k;
        gnd = gnd';
    case 'adni3_5v8' %  5 views v8.0
        dir = ['F:\my-work-space\new_dataset\mv_data\ANDI\'];
        load(strcat(dir,'adni3_23v.mat'));
        fea{1} = dtisc;
%         fea{x} = fmrisc;
        fea{2} = gabor9;
        fea{3} = glcm;
%         fea{x} = hog378;
        fea{4} = lbp18;
%         fea{x} = orb500;
%         fea{x} = surf2k;
%         fea{x} = sift2k;
        fea{5} = kaze2k;
        gnd = gnd';
    case 'adni3_5v9' %  5 views v9.0
        dir = ['F:\my-work-space\new_dataset\mv_data\ANDI\'];
        load(strcat(dir,'adni3_23v.mat'));
        fea{1} = dtisc;
%         fea{x} = fmrisc;
        fea{2} = gabor13;
        fea{3} = glcm;
%         fea{x} = hog378;
        fea{4} = lbp18;
%         fea{x} = orb500;
%         fea{x} = surf2k;
%         fea{x} = sift2k;
        fea{5} = kaze2k;
        gnd = gnd';
    case 'adni3_5v10' %  5 views v10.0
        dir = ['F:\my-work-space\new_dataset\mv_data\ANDI\'];
        load(strcat(dir,'adni3_23v.mat'));
        fea{1} = dtisc;
%         fea{x} = fmrisc;
        fea{2} = gabor13;
        fea{3} = glcm;
%         fea{x} = hog378;
        fea{4} = lbp18;
%         fea{x} = orb500;
%         fea{x} = surf2k;
%         fea{x} = sift2k;
        fea{5} = akaze2k;
        gnd = gnd';
    case 'adni3_5v11' %  5 views v11.0
        dir = ['F:\my-work-space\new_dataset\mv_data\ANDI\'];
        load(strcat(dir,'adni3_23v.mat'));
        fea{1} = dtisc;
%         fea{x} = fmrisc;
        fea{2} = gabor13;
        fea{3} = glcm;
%         fea{x} = hog378;
        fea{4} = lbp18;
%         fea{x} = orb500;
%         fea{x} = surf2k;
%         fea{x} = sift2k;
        fea{5} = akaze500;
        gnd = gnd';
    case 'adni3_5v12' %  5 views v12.0
        dir = ['F:\my-work-space\new_dataset\mv_data\ANDI\'];
        load(strcat(dir,'adni3_23v.mat'));
        fea{1} = dtisc;
%         fea{x} = fmrisc;
        fea{2} = gabor13;
        fea{3} = glcm;
%         fea{x} = hog378;
        fea{4} = lbp18;
%         fea{x} = orb500;
%         fea{x} = surf2k;
%         fea{x} = sift2k;
        fea{5} = akaze1k;
        gnd = gnd';
    case 'adni3_5v13' %  5 views v13.0
        dir = ['F:\my-work-space\new_dataset\mv_data\ANDI\'];
        load(strcat(dir,'adni3_24v.mat'));
        fea{1} = dtisc;
%         fea{x} = fmrisc;
        fea{2} = gabor13;
        fea{3} = glcm;
%         fea{x} = hog378;
        fea{4} = lbp18;
%         fea{x} = orb500;
%         fea{x} = surf2k;
        fea{5} = sift3k;
        gnd = gnd';
    case 'adni3_5v14' %  5 views v14.0
        dir = ['F:\my-work-space\new_dataset\mv_data\ANDI\'];
        load(strcat(dir,'adni3_25v.mat'));
        fea{1} = dtisc;
%         fea{x} = fmrisc;
        fea{2} = gabor13;
        fea{3} = glcm;
%         fea{x} = hog378;
        fea{4} = lbp18;
%         fea{x} = orb500;
%         fea{x} = surf2k;
        fea{5} = sift1500;
        gnd = gnd';
    case 'adni3_5v15' %  5 views v15.0
        dir = ['F:\my-work-space\new_dataset\mv_data\ANDI\'];
        load(strcat(dir,'adni3_29v.mat'));
        fea{1} = dtisc;
%         fea{x} = fmrisc;
        fea{2} = gabor13;
        fea{3} = glcm;
%         fea{x} = hog378;
        fea{4} = lbp18uniform;
%         fea{x} = lbp18;
%         fea{x} = orb500;
        fea{5} = sift2k;
        gnd = gnd';
    case 'adni3_5v16' %  5 views v16.0
        dir = ['F:\my-work-space\new_dataset\mv_data\ANDI\'];
        load(strcat(dir,'adni3_29v.mat'));
        fea{1} = dtisc;
%         fea{x} = fmrisc;
        fea{2} = gabor13;
        fea{3} = glcm;
%         fea{x} = hog378;
        fea{4} = lbp48uniform;
%         fea{x} = lbp18;
%         fea{x} = orb500;
        fea{5} = sift2k;
        gnd = gnd';
    case 'adni3_6v5' %  6 views v5.0
        dir = ['F:\my-work-space\new_dataset\mv_data\ANDI\'];
        load(strcat(dir,'adni3_30v.mat'));
        fea{1} = dtisc;
%         fea{x} = fmrisc;
        fea{2} = gabor13;
        fea{3} = glcm;
%         fea{x} = hog378;
%         fea{x} = lbp48uniform;
        fea{4} = lbp18;
%         fea{x} = orb500;
        fea{5} = sift2k;
        fea{6} = pixel;
        gnd = gnd';
    case 'breast7v_7v' %  7 views
        dir = ['D:\my-work-space\new_dataset\medmnist\breastMNIST\',...
            'Dataset_BUSI\Dataset_BUSI_with_GT\'];
        load(strcat(dir,'breast7v.mat'));
        fea{1} = gabor9;
        fea{2} = gabor13;
        fea{3} = glcm;
        fea{4} = lbp18;
        fea{5} = orb2k;
        fea{6} = pixel;
        fea{7} = sift2k;
        gnd = gnd;
    case 'breast7v_6v' %  6 views
        dir = ['F:\my-work-space\new_dataset\medmnist\breastMNIST\',...
            'Dataset_BUSI\Dataset_BUSI_with_GT\'];
        load(strcat(dir,'breast7v.mat'));
%         fea{x} = gabor9;
        fea{1} = gabor13;
        fea{2} = glcm;
        fea{3} = lbp18;
        fea{4} = orb2k;
        fea{5} = pixel;
        fea{6} = sift2k;
        gnd = gnd;
    case 'breast7v_6v2' %  6 views v2.0
        dir = ['F:\my-work-space\new_dataset\medmnist\breastMNIST\',...
            'Dataset_BUSI\Dataset_BUSI_with_GT\'];
        load(strcat(dir,'breast7v.mat'));
        fea{1} = gabor9;
%         fea{x} = gabor13;
        fea{2} = glcm;
        fea{3} = lbp18;
        fea{4} = orb2k;
        fea{5} = pixel;
        fea{6} = sift2k;
        gnd = gnd;
    case 'breast7v_5v' %  5 views 
        dir = ['F:\my-work-space\new_dataset\medmnist\breastMNIST\',...
            'Dataset_BUSI\Dataset_BUSI_with_GT\'];
        load(strcat(dir,'breast7v.mat'));
        fea{1} = gabor9;
%         fea{x} = gabor13;
        fea{2} = glcm;
        fea{3} = lbp18;
%         fea{x} = orb2k;
        fea{4} = pixel;
        fea{5} = sift2k;
        gnd = gnd;
    case 'breast7v_5v2' %  5 views v2.0 
        dir = ['F:\my-work-space\new_dataset\medmnist\breastMNIST\',...
            'Dataset_BUSI\Dataset_BUSI_with_GT\'];
        load(strcat(dir,'breast7v.mat'));
%         fea{x} = gabor9;
        fea{1} = gabor13;
        fea{2} = glcm;
        fea{3} = lbp18;
%         fea{x} = orb2k;
        fea{4} = pixel;
        fea{5} = sift2k;
        gnd = gnd;
    case 'breast7v_5v3' %  5 views v3.0
        dir = ['F:\my-work-space\new_dataset\medmnist\breastMNIST\',...
            'Dataset_BUSI\Dataset_BUSI_with_GT\'];
        load(strcat(dir,'breast7v.mat'));
%         fea{x} = gabor9;
        fea{1} = gabor13;
%         fea{x} = glcm;
        fea{2} = lbp18;
        fea{3} = orb2k;
        fea{4} = pixel;
        fea{5} = sift2k;
        gnd = gnd;
    case 'breast7v_5v4' %  5 views v4.0
        dir = ['F:\my-work-space\new_dataset\medmnist\breastMNIST\',...
            'Dataset_BUSI\Dataset_BUSI_with_GT\'];
        load(strcat(dir,'breast7v.mat'));
%         fea{x} = gabor9;
        fea{1} = gabor13;
        fea{2} = glcm;
%         fea{x} = lbp18;
        fea{3} = orb2k;
        fea{4} = pixel;
        fea{5} = sift2k;
        gnd = gnd;
    case 'breast7v_5v5' %  5 views v5.0
        dir = ['F:\my-work-space\new_dataset\medmnist\breastMNIST\',...
            'Dataset_BUSI\Dataset_BUSI_with_GT\'];
        load(strcat(dir,'breast7v.mat'));
%         fea{x} = gabor9;
        fea{1} = gabor13;
        fea{2} = glcm;
        fea{3} = lbp18;
        fea{4} = orb2k;
%         fea{x} = pixel;
        fea{5} = sift2k;
        gnd = gnd;
    case 'breast7v_5v6' %  5 views v6.0
        dir = ['F:\my-work-space\new_dataset\medmnist\breastMNIST\',...
            'Dataset_BUSI\Dataset_BUSI_with_GT\'];
        load(strcat(dir,'breast7v.mat'));
%         fea{x} = gabor9;
        fea{1} = gabor13;
        fea{2} = glcm;
        fea{3} = lbp18;
        fea{4} = orb2k;
        fea{5} = pixel;
%         fea{x} = sift2k;
        gnd = gnd;
    case 'breast7v_4v' %  4 views 
        dir = ['F:\my-work-space\new_dataset\medmnist\breastMNIST\',...
            'Dataset_BUSI\Dataset_BUSI_with_GT\'];
        load(strcat(dir,'breast7v.mat'));
%         fea{x} = gabor9;
        fea{1} = gabor13;
        fea{2} = glcm;
        fea{3} = lbp18;
        fea{4} = orb2k;
%         fea{x} = pixel;
%         fea{x} = sift2k;
        gnd = gnd;
    case 'breast8v_6v' %  6 views 
        dir = ['F:\my-work-space\new_dataset\medmnist\breastMNIST\',...
            'Dataset_BUSI\Dataset_BUSI_with_GT\'];
        load(strcat(dir,'breast8v.mat'));
%         fea{x} = gabor9;
        fea{1} = gabor13;
        fea{2} = glcm;
        fea{3} = lbp18;
        fea{4} = orb2k;
%         fea{x} = pixel;
        fea{5} = hog;
        fea{6} = sift2k;
        gnd = gnd;
    case 'breast10v_5v' %  5 views 
        dir = ['F:\my-work-space\new_dataset\medmnist\breastMNIST\',...
            'Dataset_BUSI\Dataset_BUSI_with_GT\'];
        load(strcat(dir,'breast10v.mat'));
%         fea{x} = gabor9;
        fea{1} = gabor6;
%         fea{x} = gabor13;
        fea{2} = glcm;
        fea{3} = lbp18;
        fea{4} = orb2k;
%         fea{x} = pixel;
%         fea{x} = hog;
        fea{5} = sift2k;
        gnd = gnd;
    case 'breast10v_5v2' %  5 views v2.0
        dir = ['F:\my-work-space\new_dataset\medmnist\breastMNIST\',...
            'Dataset_BUSI\Dataset_BUSI_with_GT\'];
        load(strcat(dir,'breast10v.mat'));
%         fea{x} = gabor9;
        fea{1} = gabor17;
%         fea{x} = gabor13;
        fea{2} = glcm;
        fea{3} = lbp18;
        fea{4} = orb2k;
%         fea{x} = pixel;
%         fea{x} = hog;
        fea{5} = sift2k;
        gnd = gnd;
    case 'bdgp4v' %  
        dir = ['D:\my-work-space\new_dataset\mv_data\BDGP\'];
        load(strcat(dir,'BDGP4v.mat'));
        fea{1} = X{1};
        fea{2} = X{2};
        fea{3} = X{3};
        fea{4} = X{4};
        gnd = Y;
    case 'bdgp4v200' %  
        dir = ['F:\my-work-space\new_dataset\mv_data\BDGP\'];
        load(strcat(dir,'BDGP4v.mat'));
        fea{1} = X{1};
        fea{2} = X{2};
        fea{3} = X{3};
        fea{4} = X{4};
        gnd = Y;
        for i = 1:4
            ind_dump_t = find(gnd==i);
            ind_dump = ind_dump_t(201:end);
            gnd(ind_dump) = [];
            fea{1}(ind_dump,:) = [];
            fea{2}(ind_dump,:) = [];
            fea{3}(ind_dump,:) = [];
            fea{4}(ind_dump,:) = [];
        end
    case 'caltech20' %  
        dir = ['F:\my-work-space\new_dataset\mv_data\Caltech\'];
        load(strcat(dir,'Caltech101-20.mat'));
        fea{1} = X{1};
        fea{2} = X{2};
        fea{3} = X{3};
        fea{4} = X{4};
        fea{5} = X{5};
        fea{6} = X{6};
        gnd = Y;
    case 'caltech20p50' %  
        dir = ['F:\my-work-space\new_dataset\mv_data\Caltech\'];
        load(strcat(dir,'Caltech101-20.mat'));
        fea{1} = X{1};
        fea{2} = X{2};
        fea{3} = X{3};
        fea{4} = X{4};
        fea{5} = X{5};
        fea{6} = X{6};
        gnd = Y;
        for i = 1:4
            ind_dump_t = find(gnd==i);
            ind_dump = ind_dump_t(51:end);
            gnd(ind_dump) = [];
            fea{1}(ind_dump,:) = [];
            fea{2}(ind_dump,:) = [];
            fea{3}(ind_dump,:) = [];
            fea{4}(ind_dump,:) = [];
            fea{5}(ind_dump,:) = [];
            fea{6}(ind_dump,:) = [];
        end
    case 'aloi100' % 11025 2,403,450
        dir = ['D:\my-work-space\new_dataset\TKDE-R1\ALOI-100\'];
        load(strcat(dir,'ALOI100.mat'));
        fea{1} = RGBColorHist; % 64
        fea{2} = HSVColorHist; % 64
        fea{3} = colorsim; % 77
        fea{4} = haralick; % 13 have negative values.
        gnd = gnd;
    case 'proteinfold' % 12 views 27 27 ...
        load('D:\my-work-space\new_dataset\CVPR22\protein\proteinFold.mat');
        gnd = Y;
        fea = X;
    case 'ecg12leads' % 3 views 60,000: 342, 1024, 64 ...
        load(['D:\my-work-space\new_dataset\',...
            'ECG12Leads\ecg\ecg12leads.mat'],'X','gnd');
        fea = X;
    case '100leaves' % 3 views 1600: 64, 64, 64
        load(['D:\my-work-space\new_dataset\CVPR22\100Leaves\',...
            '100Leaves.mat']);
        gnd = truth;
        fea{1} = data{1}';
        fea{2} = data{2}';
        fea{3} = data{3}';
    case 'mnist'
        load(['D:\my-work-space\new_dataset\CVPR22\MNIST\',...
            'MNIST_fea_Per1.mat'],'truelabel','data');
        gnd = truelabel{1};
        fea{1} = data{1}';
        fea{2} = data{2}';
        fea{3} = data{3}';
    case 'awa' % 4000 35,760,000
        dir = ['D:\my-work-space\new_dataset\TKDE-R1\AwA\'];
        load(strcat(dir,'AwA4000.mat'));
        fea{1} = cq; % 2688
        fea{2} = lss; % 2000
        fea{3} = phog; % 252
        fea{4} = rgsift; % 2000
        fea{5} = sift; % 2000
        fea{6} = surf(1:4000,:); % 2000
        gnd = gnd;
    case 'dermamnist4v' % 4 views
        load(['D:\my-work-space\new_dataset\',...
            'medmnist\mv_dermamnist4v_7.mat']);
        X{1} = pixel;% 784
        X{2} = gabr;% 392
        X{3} = lbp;% 256
        X{4} = grayhist;% 11
        gnd = gnd + 1;
        n_view = length(X);
        nClas = length(unique(gnd));
        fea = X;
    case 'dermamnist3v' % 3 views
        load(['D:\my-work-space\new_dataset\',...
            'medmnist\mv_dermamnist4v_7.mat']);
        X{1} = pixel;% 784
        X{2} = gabr;% 392
        X{3} = lbp;% 256
        gnd = gnd + 1;
        n_view = length(X);
        nClas = length(unique(gnd));
        fea = X;
    case 'bbcsport4vbig' % 4 views
        load(['D:\my-work-space\12-semi-supervised-IMC\',...
            'MV_datasets\bbcsport4vbig\bbcsport4vbigRnSp.mat']);
        fea{1} = X{1}'; 
        fea{2} = X{2}'; 
        fea{3} = X{3}'; 
        fea{4} = X{4}'; 
        gnd = truth;
    case '3sources3vbig' % 4 views
        load(['D:\my-work-space\12-semi-supervised-IMC\',...
            'MV_datasets\3sources3vbig\3sources3vbigRnSp.mat']);
        fea{1} = X{1}'; 
        fea{2} = X{2}'; 
        fea{3} = X{3}'; 
        gnd = truth;
    otherwise
        error('wrong dataset!');
end
%% normalize dataset 
n_view = length(fea);
if strcmpi(dataset,'webkb')||strcmpi(dataset,'BDGP4v')
    fea = fea;
elseif strcmpi(dataset,'bankuai3v')||strcmpi(dataset,'bankuai3v_2of3')
%     fea = X_remo_min(fea);
%     fea = normalize_fea(fea,n_view,1);
    fea = normalize_fea(fea,n_view,0);
else
    fea = normalize_fea(fea,n_view,0);% maxmin:1;else:norm sample to unit vec
end
%% disp info of dataset 
disp(' ');
disp(strcat('''',dataset,'''',' dataset info: '));
disp('//------------------------------------------------//');
disp('nClas: ');
nClas = length(unique(gnd));
disp(strcat('-----',num2str(nClas)));
for i = 1:n_view
    disp(strcat('nSmp x mFea: '));
    disp(strcat('-----(',num2str(size(fea{i},1)),', ',...
        num2str(size(fea{i},2)),')'));
end
str = [];
for i = 1:length(unique(gnd))
    if i == 1
        str = strcat('-----(',num2str(i),':',num2str(sum(gnd==i)),')');
    else
        str = strcat(str,'---','(',num2str(i),':',num2str(sum(gnd==i)),')');
    end
end
disp('sample of each class: ');
disp(str);
disp('//------------------------------------------------//');
end
%% func to normalize dataset
function [X] = normalize_fea(X,n_view,maxmin)
    
    if maxmin
        for i = 1:n_view
            X_temp = full(X{i});
            max_col = max(X_temp,[],1);
            min_col = min(X_temp,[],1);
            range_col = max_col - min_col;
            % eliminate dummy features
            ind_dummy_f = find(range_col==0);
            X_temp(:,ind_dummy_f) = [];
            min_col(ind_dummy_f) = [];
            range_col(ind_dummy_f) = [];
            % 
            diff_mat = X_temp - min_col;
            X{i} = diff_mat./max(range_col,eps);
            
            % unit vector
%             X_temp_2 = X{i};
%             norm_col = sqrt(sum(X_temp_2.^2,2));
%             % 
%             X{i} = X_temp_2./norm_col;
        end
    else
        for i = 1:n_view
%             X_temp = full(X{i});
%             norm_col = sqrt(sum(X_temp.^2,1));
%             % eliminate dummy features
%             ind_dummy_f = norm_col==0;
%             X_temp(:,ind_dummy_f) = [];
%             % 
%             X{i} = X_temp./norm_col;
            
            % unit vector
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % %             X_temp = full(X{i});
% % % %             max_col = max(X_temp,[],1);
% % % %             min_col = min(X_temp,[],1);
% % % %             range_col = max_col - min_col;
% % % %             % eliminate dummy features
% % % %             ind_dummy_f = find(range_col==0);
% % % %             X_temp(:,ind_dummy_f) = [];
% % % %             min_col(ind_dummy_f) = [];
% % % %             range_col(ind_dummy_f) = [];
% % % %             diff_mat = X_temp - min_col;
% % % %             X{i} = diff_mat./range_col;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
            
            X_temp_2 = X{i};
            norm_col = sqrt(sum(X_temp_2.^2,2));
            % 
            X{i} = X_temp_2./max(norm_col,eps);
        end
    end
    
end

function X = X_remo_min(X)

    n_view = length(X);
    for i = 1:n_view
        X{i} = X{i}-min(X{i},[],1);
    end

end
