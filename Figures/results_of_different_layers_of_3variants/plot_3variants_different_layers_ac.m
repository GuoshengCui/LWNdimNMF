close all
clear
clc
%% 
% 1:'LWNdimNMF',2:'LWNdimNMF\_HA',3:'LWNdimNMF\_ON'
% AC (or NMI Purity) of 1layer, 2layer, 3layer 
% 【0.7】：'wisconsin',【0.3】：'yalea',
Dataname = 'wisconsin'; 
percent = 0.3; % 2v:{0.7,0.9}; 3v:{0.1,0.3};
root = ['D:\my-work-space\11-imcomplete-mv-deepIMC\',...
    'Figures\results_of_different_layers_of_3variants'];

for nlayer = 1:3
    for imethod = 1:3 % 1:'LWNdimNMF',2:'LWNdimNMF\_HA',3:'LWNdimNMF\_ON'
        if imethod==1
            [knn,alpha,beta,layers] = ...
                get_paras_3layers_lwn(Dataname,percent,nlayer);
            path = [root,'\LWNdimNMF\',Dataname];
            Results = load_lwn(path,percent,nlayer,knn,alpha,beta,layers);
        elseif imethod==2
            [knn,alpha,layers] = ...
                get_paras_3layers_lwn_halign(Dataname,percent,nlayer);
            path = [root,'\LWNdimNMF_HAlign\',Dataname];
            Results = load_lwn_halign(path,percent,nlayer,knn,alpha,layers);
        elseif imethod==3
            [knn,alpha,beta,layers] = ...
                get_paras_3layers_lwn_nnorm(Dataname,percent,nlayer);
            path = [root,'\LWNdimNMF_NNorm\',Dataname];
            Results = load_lwn_nnorm(path,percent,nlayer,knn,alpha,beta,layers);
        else
            error('imethod must be in {1,2,3}!');
        end
        Y(nlayer,imethod) = Results(1,1);% AC
    end
end
figure;
Bodlabel = 22;
X=1:3;% 1,2,3 layers
h=bar(X,Y);
% 设置条形图颜色
set(h(1),'FaceColor',[0.3,0.3,0.7])
set(h(2),'FaceColor',[0.3,0.8,0.2])
set(h(3),'FaceColor',[1.0,0.7,0.0])
legend('LWNdimNMF','LWNdimNMF\_HA','LWNdimNMF\_ON',...
    'Location','northwest'); % ,'FontSize',12
% set(gca,'xtick',1:3);% 3 layers
% set(gca,'XTickLabel',xticks,'FontSize',12); % 'FontName','Times New Roman'
set(gca,'xticklabel',{'1 layer','2 layer','3 layer'},...
    'FontSize',12);
ylim([0,min(max(max(Y))*1.4,99)]);% 110 95 99 
xlabel('#Layers','FontSize',Bodlabel);
ylabel('AC(%)','FontSize',Bodlabel);
DatanameOfficial = choose_official_dataname(Dataname);
title(DatanameOfficial,'FontSize',Bodlabel);
% set(gca,'xticklabel',{'1{\it{layer}}','2{\it{layer}}','3{\it{layer}}'},...
%     'FontSize',12,'FontName','Times New Roman');
Y_1=roundn(Y,-2);

function [dataname] = choose_official_dataname(dataset)

switch dataset
    case 'wisconsin'
        dataname = 'Wisconsin';
    case 'yalea'
        dataname = 'yaleA';
    case 'msrcv1'
        dataname = 'MSRC-v1';
    otherwise      
        error('wrong dataset!!!!');
end

end

function res = load_lwn(path,percent,nlayer,knn,alpha,beta,layers)

    if length(layers)==1
        load(strcat(path,'\',...% one layer
            num2str(percent),'_',num2str(nlayer),'layer',...
            '_k=',num2str(knn),...
            '_[',num2str(log10(alpha)),',',...
            num2str(log10(beta)),']',...
            '_[',num2str(layers(1)),'].mat'),'LWNdimNMF');
        res = LWNdimNMF{1};
    elseif length(layers)==2
        load(strcat(path,'\',...% two
            num2str(percent),'_',num2str(nlayer),'layer',...
            '_k=',num2str(knn),...
            '_[',num2str(log10(alpha)),',',...
            num2str(log10(beta)),']',...
            '_[',num2str(layers(1)),',',...
            num2str(layers(2)),'].mat'),'LWNdimNMF');
        res = LWNdimNMF{1};
    elseif length(layers)==3
        load(strcat(path,'\',...% three
            num2str(percent),'_',num2str(nlayer),'layer',...
            '_k=',num2str(knn),...
            '_[',num2str(log10(alpha)),',',...
            num2str(log10(beta)),']',...
            '_[',num2str(layers(1)),...
            ',',num2str(layers(2)),...
            ',',num2str(layers(3)),'].mat'),'LWNdimNMF');
        res = LWNdimNMF{1};
    end
end

function res = load_lwn_halign(path,percent,nlayer,knn,alpha,layers)

    if length(layers)==1
        load(strcat(path,'\',...% one layer
            num2str(percent),'_',num2str(nlayer),'layer',...
            '_k=',num2str(knn),...
            '_alpha=[',num2str(log10(alpha)),']',...
            '_[',num2str(layers(1)),'].mat'),'LWNdimNMF_HAlign');
        res = LWNdimNMF_HAlign{1};
    elseif length(layers)==2
        load(strcat(path,'\',...% two
            num2str(percent),'_',num2str(nlayer),'layer',...
            '_k=',num2str(knn),...
            '_alpha=[',num2str(log10(alpha)),']',...
            '_[',num2str(layers(1)),',',...
            num2str(layers(2)),'].mat'),'LWNdimNMF_HAlign');
        res = LWNdimNMF_HAlign{1};
    elseif length(layers)==3
        load(strcat(path,'\',...% three
            num2str(percent),'_',num2str(nlayer),'layer',...
            '_k=',num2str(knn),...
            '_alpha=[',num2str(log10(alpha)),']',...
            '_[',num2str(layers(1)),...
            ',',num2str(layers(2)),...
            ',',num2str(layers(3)),'].mat'),'LWNdimNMF_HAlign');
        res = LWNdimNMF_HAlign{1};
    end
end

function res = load_lwn_nnorm(path,percent,nlayer,knn,alpha,beta,layers)

    if length(layers)==1
        load(strcat(path,'\',...% one layer
            num2str(percent),'_',num2str(nlayer),'layer',...
            '_k=',num2str(knn),...
            '_[',num2str(log10(alpha)),',',...
            num2str(log10(beta)),']',...
            '_[',num2str(layers(1)),'].mat'),'LWNdimNMF_NNorm');
        res = LWNdimNMF_NNorm{1};
    elseif length(layers)==2
        load(strcat(path,'\',...% two
            num2str(percent),'_',num2str(nlayer),'layer',...
            '_k=',num2str(knn),...
            '_[',num2str(log10(alpha)),',',...
            num2str(log10(beta)),']',...
            '_[',num2str(layers(1)),',',...
            num2str(layers(2)),'].mat'),'LWNdimNMF_NNorm');
        res = LWNdimNMF_NNorm{1};
    elseif length(layers)==3
        load(strcat(path,'\',...% three
            num2str(percent),'_',num2str(nlayer),'layer',...
            '_k=',num2str(knn),...
            '_[',num2str(log10(alpha)),',',...
            num2str(log10(beta)),']',...
            '_[',num2str(layers(1)),...
            ',',num2str(layers(2)),...
            ',',num2str(layers(3)),'].mat'),'LWNdimNMF_NNorm');
        res = LWNdimNMF_NNorm{1};
    end
end



