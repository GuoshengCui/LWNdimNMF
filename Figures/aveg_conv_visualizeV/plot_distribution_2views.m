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

% % % % % 数据
% % % % ydata = random('Normal',0,1,1,1024);
% % % % % 准备一些变量
% % % % bins = 100; %分100个区间进行统计
% % % % maxdat = max(ydata);%最大值
% % % % mindat = min(ydata);%最小值
% % % % bin_space = (maxdat - mindat) / bins;%每个bin宽度
% % % % xtick = mindat : bin_space : maxdat - bin_space;
% % % % % 求pdf
% % % % distribution = hist(ydata,bins);%使用直方图得到数据落在各区间的总数
% % % % pdf = bins * distribution / ((sum(distribution )) * (maxdat - mindat));%计算pdf
% % % % % 画图
% % % % figure;
% % % % plot(xtick,pdf);
% % % % % 验证：sum(pdf) * bin_space应该接近1

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
