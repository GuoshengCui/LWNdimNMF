
close all;
clear;
clc;


load('0.3_yalea_Z_3layers_LWNdimNMF.mat');
Z1 = Z;
load('0.3_yalea_Z_3layers_LWNdimNMF_HA.mat');
Z2 = Z;
load('0.3_yalea_Z_3layers_LWNdimNMF_NN.mat');
Z3 = Z;

view = 3;
Bod = 14;
Bodxlabel = 16;
Bod_title = 12;
%% LWNdimNMF
subplot(131);
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
title(['\mu_1=',num2str(mean(Z1{view,1}(:))),', ',...
    '\mu_2=',num2str(mean(Z1{view,2}(:))),', ',...
    '\mu_3=',num2str(mean(Z1{view,3}(:)))],'FontSize',Bod_title);
xlabel('(a) LWNdimNMF','FontSize',Bodxlabel,'FontName','Times New Roman');

%% LWNdimNMF_HA
subplot(132);
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
title(['\mu_1=',num2str(mean(Z2{view,1}(:))),', ',...
    '\mu_2=',num2str(mean(Z2{view,2}(:))),', ',...
    '\mu_3=',num2str(mean(Z2{view,3}(:)))],'FontSize',Bod_title);
xlabel('(b) LWNdimNMF\_HA','FontSize',Bodxlabel,'FontName','Times New Roman');

%% LWNdimNMF_ON
subplot(133);
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
title(['\mu_1=',num2str(mean(Z3{view,1}(:))),', ',...
    '\mu_2=',num2str(mean(Z3{view,2}(:))),', ',...
    '\mu_3=',num2str(mean(Z3{view,3}(:)))],'FontSize',Bod_title);
xlabel('(c) LWNdimNMF\_ON','FontSize',Bodxlabel,'FontName','Times New Roman');

