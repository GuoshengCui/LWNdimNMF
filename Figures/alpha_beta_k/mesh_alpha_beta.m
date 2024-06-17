close all;
clear;
clc;
% 'msrcv1'----0.3
% 'handwritten'----0.5
figure('color','w');
dataname = 'msrcv1';
% dataname = 'handwritten';
metric = 1;% display accuracy
root = 'D:\my-work-space\11-imcomplete-mv-deepIMC\Figures\alpha_beta_k\';
%% 
filepath = [root,dataname,'\0.3_k=3_[200,100,28].mat'];
% filepath = [root,dataname,'\0.5_k=8_[100,10].mat'];
load(filepath);
x = -3:3;
y = -3:3;
[X,Y] = meshgrid(x,y);
% R = sqrt(X.^2+Y.^2)+eps;
% Z = sin(R)./R;
for ix = 1:7
    for iy = 1:7
        Z(ix,iy) = LWNdimNMF{ix,iy}(1,1);% AC
%         Z(ix,iy) = LWNdimNMF{ix,iy}(1,2);% NMI
%         Z(ix,iy) = LWNdimNMF{ix,iy}(1,3);% Purity
    end
end
Bod = 20;
set(gca,'xticklabel',-3:3,'yticklabel',-3:3);
% xlim([-3,3]);
% ylim([-3,3]);
% mesh(X,Y,Z);
surf(X,Y,Z);
ylabel('log(\alpha)','FontSize',Bod);
xlabel('log(\beta)','FontSize',Bod);
zlabel('AC (%)','FontSize',Bod);
shading faceted;
% shading interp;
% shading flat;