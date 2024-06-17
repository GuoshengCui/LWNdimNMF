close all;
clear;
clc;
% 'msrcv1'----0.3
% 'handwritten'----0.5
figure('color','w');
dataname = 'msrcv1';
root = 'D:\my-work-space\11-imcomplete-mv-deepIMC\Figures\alpha_beta_k\k\';
%% read acc of IMCNAS-1
load([root,dataname,...
    '\0.3_k=[1,12]_alpha=-1_beta=1_lys_[200,100,28].mat']);
acc_stack = [];
nmi_stack = [];
pur_stack = [];
k_list = 1:12;
for inK = 1:length(k_list)
    acc_m(inK) = LWNdimNMF{inK}(1,1);
    nmi_m(inK) = LWNdimNMF{inK}(1,2);
    pur_m(inK) = LWNdimNMF{inK}(1,3);
    acc_s(inK) = LWNdimNMF{inK}(2,1);
    nmi_s(inK) = LWNdimNMF{inK}(2,2);
    pur_s(inK) = LWNdimNMF{inK}(2,3);
end
acc_stack = [acc_stack,acc_m];
nmi_stack = [nmi_stack,nmi_m];
pur_stack = [pur_stack,pur_m];
Bod = 1.5;
BodFond = 20;
yyaxis left; % ¼¤»î×ó±ßµÄÖá
h1 = errorbar(k_list,acc_m,acc_s,'-.s','LineWidth',Bod);
% h1 = plot(axis_X,acc_m,'b-s');
if strcmp(dataname,'msrcv1')
    ylim([0,max(acc_stack)+5]);
end
ylabel('AC (%)','FontSize',BodFond);
xlabel('\it k','FontSize',BodFond);
hold on;

yyaxis right; % ¼¤»îÓÒ±ßµÄÖá
h2 = errorbar(k_list,nmi_m,nmi_s,'-.s','LineWidth',Bod);
if strcmp(dataname,'msrcv1')
%     ylim([0,max(nmi_stack)+5]);
    ylim([0,max(acc_stack)+5]);
end
xlim([1,length(k_list)]);
ylabel('NMI (%)','FontSize',BodFond);
xlabel('\it k','FontSize',BodFond);
legend([h1,h2],'LWNdimNMF (AC)','LWNdimNMF (NMI)',...
    'Location','northeast','FontSize',BodFond);
grid on;












