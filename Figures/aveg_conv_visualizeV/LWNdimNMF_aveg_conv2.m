function [W,H,Ht,Hc,obj_final] = LWNdimNMF_aveg_conv2(X,gnd,G,options)
% (alpha,beta,p,max_iter,nn)
nClas = length(unique(gnd));
n_view = length(X);
nSmp_all = length(gnd);
% paras.nSmp_all = nSmp_all;
fea = X;
num_nmf = 10;

paras.alpha = options.alpha;% can not be set 0. fix this para as 1 !!! tune others
paras.beta = options.beta;% laplacian
paras.gamma = 1e-3;%
% ###################### paras of ICMVNMF_MvDGNMF ################
% paras.beta = 1e-3;
paras.lambda_ls = [0.05,0.05];% "recommend ... lambda1 and lambda2 as 10 and 1".
% paras.layers = [100,nClas];% layers(1):{20, 50, 100};layers(2)=nClass
ly = [];
for ily = 1:options.n_layer
    ly = [ly options.layers(ily)];
end
paras.layers = ly;
% if options.n_layer==3
% paras.layers = [options.layers(1),options.layers(2),options.layers(3)];
% elseif options.n_layer==2
% paras.layers = [options.layers(1),options.layers(2)];
% elseif options.n_layer==1
% paras.layers = [options.layers(1)];
% end
paras.n_layer = options.n_layer;
% paras.n_layer = 2;
% % % % paras.lambda_ls = [0.01,0.5,0.5];% "recommend ... lambda1 and lambda2 as 10 and 1".
% % % % paras.layers = [50,2*nClas,nClas];% layers(1):{20, 50, 100};layers(2)=nClass
% % % % paras.n_layer = 3;
% ################################################################
paras.p = 2;
paras.max_iter = options.max_iter;% 15,25 is good 
paras.dim = nClas;
paras.nClas = nClas;
paras.G = G;
paras.nSmp_all = nSmp_all;
paras.nClas = length(unique(gnd));
%% construct affinity matrix
%{
% n_neighbors = options.k;% 5,6 is good for 0.1; (2,1)1,2 is good for 0.5
n_neighbors = 3;
for i = 1:n_view
    sigma = 1;% 1 ecg:2 
    Si = construct_W(fea{i},n_neighbors,sigma);
    paras.Sg{i} = Si; 
    paras.Dg{i} = [];
end
%}
%% construct affinity matrix
% {
n_neighbors = options.k;
for i = 1:n_view
%     [nSmp,mFea] = size(fea{i});
    [Hinc,We,De,Dv] = Hpergraph2(fea{i}',n_neighbors);
    Sihyp = Hinc*We*De^(-1)*Hinc';
%     Si = Hinc*We/De*Hinc';
    paras.Shyp{i} = Sihyp;% Si
    paras.Dhyp{i} = Dv;% Dv
% % % %     diagDv = 1./sqrt(diag(Dv));
% % % %     Dv12 = diag(diagDv);
% % % %     Si = Dv12*Hinc*We*De^(-1)*Hinc'*Dv12;
% % % %     paras.S{i} = Si; 
% % % %     paras.D{i} = eye(size(Dv,1));
%     S = HK / De  * We * HK';
%     L = Dv-Si;
end
%}

% rand('seed',666);
% rand('seed',7);
% rand('seed',555);
for inmf = 1:10
    tic 
%     [Z,H,Hend,Hc,obj] = LWNdimNMF(fea,paras);W = [];Ht=[];
    [Z,H,Hc,obj] = LWNdimNMF_HAlign_out_Hc(fea,paras);W=[];Ht=[];
%     [Z,H,Hend,Hc,obj] = LWNdimNMF_NNorm(fea,paras);W = [];Ht=[];
%     [Z,H,Hend,Hc,obj] = DeepNMF(fea,paras);W = [];Ht=[];obj(3) = zeros(size(obj(2)));
%     [Z,H,Hend,Hc,obj] = DeepNMF_2layers(fea,paras);W = [];Ht=[];obj(3) = zeros(size(obj(2)));
%     [Z,H,Hend,Hc,obj] = DeepNMF_3layers(fea,paras);W = [];Ht=[];obj(3) = zeros(size(obj(2)));
toc

    obj_list(inmf) = obj(1);
    obj_rec1_list(inmf) = obj(2);
    obj_rec2_list(inmf) = obj(3);
end
obj_final = [obj_list;obj_rec1_list;obj_rec2_list];

end

function W = construct_W(fea,num_knn,sigma)

      opts = [];
      opts.NeighborMode = 'KNN';
      opts.k = num_knn;
      opts.WeightMode = 'HeatKernel';% Binary Cosine HeatKernel
      opts.t = sigma;
      W = constructW(fea,opts);
%       W = (W+W')/2;
end

function [Hinc,W,De,Dv] = Hpergraph2(V,k)
% the sizes of U and V are the same.
% k : k NN.
% sample matrix.
[~,n]=size(V);
% Dis = zeros(n,n);
% count = 0;
% for i = 1:n
%     for j = 1:n
%         if j>i
%             count = count + 1;
%             %             Dis(i,j) = double(exp(-norm(V(:,i)-V(:,j))));
%             Dis(i,j) = norm(V(:,i)-V(:,j));
%         end
%     end
% end
% sigma = double(sum(Dis(:))/count);
D = EuDist2(V');
sigma = mean(mean(D));
% sigma=1e2;
HK = zeros(n,n);
for i=1:n
    for j=1:n
        HK(i,j)=exp(-norm(V(:,i)-V(:,j))^2/(sigma^2));
    end
end
% HK = (HK+HK')/2;
Hinc = zeros(n,n);
for i=1:n
    [~,ind]=sort(HK(:,i),'descend'); % increasing order.
    col = ind(1:k+1);
%     Hinc(col,i) = 1; 
    Hinc(col,i) = HK(col,i);% this is used. 
%     Hinc(col,i) = HK(col,i)/sum(HK(col,i));
end
W = diag(sum(Hinc,1));
% W = eye(n);
% get D_{e}
De = diag(sum(Hinc,1));
% get D_{v}
Dv = diag(sum(Hinc*W,2));

end
