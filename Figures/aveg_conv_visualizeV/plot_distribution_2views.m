close all;
clear;
clc;


load('0.5_wisconsin_Z_3layers_LWNdimNMF.mat');
Z1 = Z;
load('0.5_wisconsin_Z_3layers_LWNdimNMF_HA.mat');
Z2 = Z;
load('0.5_wisconsin_Z_3layers_LWNdimNMF_NN.mat');
Z3 = Z;
% iview = 2;

% % % % y=Z1{1,1}(:);
% % % % 
% % % % ymin=min(y);
% % % % 
% % % % ymax=max(y);
% % % % x = linspace(ymin,ymax,20);

% % % % % ����
% % % % ydata = random('Normal',0,1,1,1024);
% % % % % ׼��һЩ����
% % % % bins = 100; %��100���������ͳ��
% % % % maxdat = max(ydata);%���ֵ
% % % % mindat = min(ydata);%��Сֵ
% % % % bin_space = (maxdat - mindat) / bins;%ÿ��bin���
% % % % xtick = mindat : bin_space : maxdat - bin_space;
% % % % % ��pdf
% % % % distribution = hist(ydata,bins);%ʹ��ֱ��ͼ�õ��������ڸ����������
% % % % pdf = bins * distribution / ((sum(distribution )) * (maxdat - mindat));%����pdf
% % % % % ��ͼ
% % % % figure;
% % % % plot(xtick,pdf);
% % % % % ��֤��sum(pdf) * bin_spaceӦ�ýӽ�1

% % x1 =normrnd(5,1,100,1);
% x1 = Z1{1,1}(:);
% % x2 =normrnd(6,1,100,1);
% x2 = Z1{1,2}(:);
% 
% x = [x1 x2];
% 
% boxplot(x,1,'g+',1,0);

% x1 = Z1{1,1}(:);
% subplot(131);
% boxplot(x1,1,'g+',1,0);
% set(gca,'xticklabel',{'1 layer'});
% x1 = Z1{1,2}(:);
% subplot(132);
% boxplot(x1,1,'g+',1,0);
% set(gca,'xticklabel',{'2 layer'});
% x1 = Z1{1,3}(:);
% subplot(133);
% boxplot(x1,1,'g+',1,0);
% set(gca,'xticklabel',{'3 layer'});

x1 = Z2{1,1}(:);
subplot(131);
boxplot(x1,1,'g+',1,0);
% boxplot(x1,'PlotStyle','compact');
set(gca,'xticklabel',{'1 layer'});
x1 = Z2{1,2}(:);
subplot(132);
boxplot(x1,1,'g+',1,0);
% boxplot(x1,'PlotStyle','compact');
set(gca,'xticklabel',{'2 layer'});
x1 = Z2{1,3}(:);
subplot(133);
boxplot(x1,1,'g+',1,0);
% boxplot(x1,'PlotStyle','compact');
set(gca,'xticklabel',{'3 layer'});
% 
% x1 = Z3{1,1}(:);
% subplot(131);
% boxplot(x1,1,'g+',1,0);
% x1 = Z3{1,2}(:);
% subplot(132);
% boxplot(x1,1,'g+',1,0);
% x1 = Z3{1,3}(:);
% subplot(133);
% boxplot(x1,1,'g+',1,0);

% x1 = Z1{1,1}(:);
% subplot(131);
% histfit(x1);
% x1 = Z1{1,2}(:);
% subplot(132);
% histfit(x1);
% x1 = Z1{1,3}(:);
% subplot(133);
% histfit(x1);

% x1 = Z2{1,1}(:);
% subplot(131);
% histfit(x1);
% x1 = Z2{1,2}(:);
% subplot(132);
% histfit(x1);
% x1 = Z2{1,3}(:);
% subplot(133);
% histfit(x1);

% x1 = Z3{1,1}(:);
% subplot(131);
% histfit(x1);
% x1 = Z3{1,2}(:);
% subplot(132);
% histfit(x1);
% x1 = Z3{1,3}(:);
% subplot(133);
% histfit(x1);
