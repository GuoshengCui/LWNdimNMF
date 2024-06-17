

function [Datafold,Data] = getData(Dataname,percent,isTwoView)

if ~isTwoView
    if strcmp(Dataname,'bbcsport4vbig')
        percentDel = percent;
        Datafold = ['MV_datasets/',Dataname,'/',Dataname,...
            'RnSp_percentDel_',num2str(percentDel),'_new','.mat'];
        Data = ['MV_datasets/',Dataname,'/',Dataname,'RnSp'];
%         percentDel = percent;
%         Datafold = ['MV_datasets/',Dataname,'/old_data_from_paper/',Dataname,...
%             'RnSp_percentDel_',num2str(percentDel),'_new_old','.mat'];
%         Data = ['MV_datasets/',Dataname,'/',Dataname,'RnSp'];
    else
        percentDel = percent;
        Datafold = ['MV_datasets/',Dataname,'/',Dataname,...
            'RnSp_percentDel_',num2str(percentDel),'.mat'];
        Data = ['MV_datasets/',Dataname,'/',Dataname,'RnSp'];
    end
else
    percent_pair = percent;
    dataroot = ['MV_datasets/',Dataname,'/',Dataname,'_with_G/'];
    Datafold = [dataroot,Dataname,'_Folds_with_G_paired_',...
        num2str(percent_pair),'.mat'];
    Data = [dataroot,Dataname,'_RnSp_with_G'];
end
end