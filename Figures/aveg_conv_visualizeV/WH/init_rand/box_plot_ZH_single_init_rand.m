
close all;
clear;
clc;



view = 1;
Bod = 14;
Bodxlabel = 16;
Bod_title = 12;
root = ['F:\我的工作空间\11-imcomplete-mv-deepIMC\Figures\',...
    'aveg_conv_visualizeV\WH\init_rand\'];
%% LWNdimNMF
%{
load([root,'0.5_wisconsin_3layers_LWNdimNMF_ZH.mat']);
% load([root,'0.7_wisconsin_3layers_LWNdimNMF_ZH.mat']);
Z1 = Z;H1 = H;
%%%%%%%%%%%%%%%%%%%%%%%% Z %%%%%%%%%%%%%%%%%%%%%%%%
figure(666);
x1 = Z1{view,1}(:);
x2 = Z1{view,2}(:);
x3 = Z1{view,3}(:);
x = [x1; x2; x3];

g1 = repmat({'1st layer'},length(Z1{view,1}(:)),1);
g2 = repmat({'2nd layer'},length(Z1{view,2}(:)),1);
g3 = repmat({'3rd layer'},length(Z1{view,3}(:)),1);
g = [g1; g2; g3];
boxplot(x,g);
set(gca,'FontSize',Bod);
title(['\mu_1=',num2str(round(mean(Z1{view,1}(:)),4)),', ',...
    '\mu_2=',num2str(round(mean(Z1{view,2}(:)),4)),', ',...
    '\mu_3=',num2str(round(mean(Z1{view,3}(:)),4))],'FontSize',Bod_title);
% xlabel('(a) LWNdimNMF Z_1^m, Z_2^m and Z_3^m','FontSize',...
%     Bodxlabel,'FontName','Times New Roman');

%%%%%%%%%%%%%%%%%%%%%%%% H %%%%%%%%%%%%%%%%%%%%%%%%
figure(777);
x1 = H1{view,1}(:);
x2 = H1{view,2}(:);
x3 = H1{view,3}(:);
x = [x1; x2; x3];

g1 = repmat({'1st layer'},length(H1{view,1}(:)),1);
g2 = repmat({'2nd layer'},length(H1{view,2}(:)),1);
g3 = repmat({'3rd layer'},length(H1{view,3}(:)),1);
g = [g1; g2; g3];
boxplot(x,g);
set(gca,'FontSize',Bod);
title(['\mu_1=',num2str(round(mean(H1{view,1}(:)),4)),', ',...
    '\mu_2=',num2str(round(mean(H1{view,2}(:)),4)),', ',...
    '\mu_3=',num2str(round(mean(H1{view,3}(:)),4))],'FontSize',Bod_title);
% xlabel('(a) LWNdimNMF H_1^m, H_2^m and H_3^m','FontSize',...
%     Bodxlabel,'FontName','Times New Roman');
%}
%% LWNdimNMF_ON
%{
load([root,'0.5_wisconsin_3layers_LWNdimNMF_NN_ZH.mat']);
% % % % load([root,'0.7_wisconsin_3layers_LWNdimNMF_NN_ZH.mat']);
Z3 = Z;H1 = H;
%%%%%%%%%%%%%%%%%%%%%%%% Z %%%%%%%%%%%%%%%%%%%%%%%%
figure(666);
x1 = Z3{view,1}(:);
x2 = Z3{view,2}(:);
x3 = Z3{view,3}(:);
x = [x1; x2; x3];

g1 = repmat({'1st layer'},length(Z3{view,1}(:)),1);
g2 = repmat({'2nd layer'},length(Z3{view,2}(:)),1);
g3 = repmat({'3rd layer'},length(Z3{view,3}(:)),1);
g = [g1; g2; g3];
boxplot(x,g);
set(gca,'FontSize',Bod);
title(['\mu_1=',num2str(round(mean(Z3{view,1}(:)),8)),', ',...
    '\mu_2=',num2str(round(mean(Z3{view,2}(:)),4)),', ',...
    '\mu_3=',num2str(round(mean(Z3{view,3}(:)),4))],'FontSize',Bod_title);
% xlabel('(c) LWNdimNMF\_ON Z_1^m, Z_2^m and Z_3^m','FontSize',Bodxlabel,'FontName','Times New Roman');

%%%%%%%%%%%%%%%%%%%%%%%% H %%%%%%%%%%%%%%%%%%%%%%%%
figure(777);
x1 = H1{view,1}(:);
x2 = H1{view,2}(:);
x3 = H1{view,3}(:);
x = [x1; x2; x3];

g1 = repmat({'1st layer'},length(H1{view,1}(:)),1);
g2 = repmat({'2nd layer'},length(H1{view,2}(:)),1);
g3 = repmat({'3rd layer'},length(H1{view,3}(:)),1);
g = [g1; g2; g3];
boxplot(x,g);
set(gca,'FontSize',Bod);
title(['\mu_1=',num2str(round(mean(H1{view,1}(:)),4)),', ',...
    '\mu_2=',num2str(round(mean(H1{view,2}(:)),4)),', ',...
    '\mu_3=',num2str(round(mean(H1{view,3}(:)),4))],'FontSize',Bod_title);
% xlabel('(a) LWNdimNMF\_ON H_1^m, H_2^m and H_3^m','FontSize',...
%     Bodxlabel,'FontName','Times New Roman');
%}
%% LWNdimNMF_HA
% {
load([root,'0.5_wisconsin_3layers_LWNdimNMF_HA_ZH.mat']);
load([root,'G.mat']);
Z2 = Z;H1 = H;H1{1,3} = Hc*G{1}';H1{2,3} = Hc*G{2}';
%%%%%%%%%%%%%%%%%%%%%%%% Z %%%%%%%%%%%%%%%%%%%%%%%%
figure(666);
x1 = Z2{view,1}(:);
x2 = Z2{view,2}(:);
x3 = Z2{view,3}(:);
x = [x1; x2; x3];

g1 = repmat({'1st layer'},length(Z2{view,1}(:)),1);
g2 = repmat({'2nd layer'},length(Z2{view,2}(:)),1);
g3 = repmat({'3rd layer'},length(Z2{view,3}(:)),1);
g = [g1; g2; g3];
boxplot(x,g);
set(gca,'FontSize',Bod);
title(['\mu_1=',num2str(round(mean(Z2{view,1}(:)),8)),', ',...
    '\mu_2=',num2str(round(mean(Z2{view,2}(:)),4)),', ',...
    '\mu_3=',num2str(round(mean(Z2{view,3}(:)),4))],'FontSize',Bod_title);
% xlabel('(a) LWNdimNMF\_HA Z_1^m, Z_2^m and Z_3^m','FontSize',...
%     Bodxlabel,'FontName','Times New Roman');

%%%%%%%%%%%%%%%%%%%%%%%% H %%%%%%%%%%%%%%%%%%%%%%%%
figure(777);
x1 = H1{view,1}(:);
x2 = H1{view,2}(:);
x3 = H1{view,3}(:);
x = [x1; x2; x3];

g1 = repmat({'1st layer'},length(H1{view,1}(:)),1);
g2 = repmat({'2nd layer'},length(H1{view,2}(:)),1);
g3 = repmat({'3rd layer'},length(H1{view,3}(:)),1);
g = [g1; g2; g3];
boxplot(x,g);
set(gca,'FontSize',Bod);
title(['\mu_1=',num2str(round(mean(H1{view,1}(:)),4)),', ',...
    '\mu_2=',num2str(round(mean(H1{view,2}(:)),4)),', ',...
    '\mu_3=',num2str(round(mean(H1{view,3}(:)),4))],'FontSize',Bod_title);
% xlabel('(a) LWNdimNMF\_HA H_1^m, H_2^m and H_c','FontSize',...
%     Bodxlabel,'FontName','Times New Roman');
%}
