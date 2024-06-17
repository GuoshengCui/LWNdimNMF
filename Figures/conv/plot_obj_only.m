close all
clear
clc
% 'yalea','msrcv1'----0.3
% 'wisconsin'----0.5
figure('color','w');
dataname_list = {'wisconsin','yalea','msrcv1'};
Bod = 20;
Bodlegend = 18;
root = 'D:\my-work-space\11-imcomplete-mv-deepIMC\Figures\conv';
%% read acc of LWNdimNMF
for idata = 1:length(dataname_list)
    if strcmp(dataname_list(idata),'wisconsin')
        percent = 0.5;
    else
        percent = 0.3;
    end
    nc = get_nClass(dataname_list(idata));
    nS = 4;
    path = [root,'\obj_res\obj_',dataname_list{idata},'_',...
        num2str(percent),'_[200,100,',num2str(nS*nc),'].mat'];
    load(path);
    marktype = choose_marktype_1(idata);
%     h(idata) = plot(1:300,obj(2:301),marktype,'linewidth',2);hold on;
    h(idata) = plot(1:300,log(obj(2:301)),marktype,'linewidth',2);hold on;
end
% title('LWNdimNMF','FontSize',Bod);
xlabel('#Iterations','FontSize',Bod);
% ylabel('Objective Value','FontSize',Bod);
ylabel('log(Objective Value)','FontSize',Bod);

% legend([h(1),h(2),h(3)],'Wisconsin','yaleA','MSRC-v1','FontSize',Bodlegend);
legend(h(1),'Wisconsin','FontSize',Bodlegend);
ah=axes('position',get(gca,'position'),'visible','off');
legend(ah,h(2:3),'yaleA','MSRC-v1','FontSize',Bodlegend);


function marktype = choose_marktype_1(idata)

    if idata==1
        marktype = '-r';%d
    elseif idata==2
        marktype = '-.g';%*
    elseif idata==3
        marktype = '-.c';%^
    end
end

function [nc] = get_nClass(dataset)

if strcmp(dataset,'wisconsin')
    nc = 5;
elseif strcmp(dataset,'yalea')
    nc = 15;
elseif strcmp(dataset,'msrcv1')
    nc = 7;
else
    error('wrong dataset name!!!!');
end
    
end

