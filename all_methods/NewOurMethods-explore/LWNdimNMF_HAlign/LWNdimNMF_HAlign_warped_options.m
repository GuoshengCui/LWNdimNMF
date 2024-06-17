function [W,H,Hc] = LWNdimNMF_HAlign_warped_options(X,gnd,G,options)
% (alpha,beta,p,max_iter,nn)
nClas = length(unique(gnd));
n_view = length(X);
nSmp_all = length(gnd);
% paras.nSmp_all = nSmp_all;
fea = X;
num_nmf = 1;

paras.alpha = options.alpha;
% ################################################################
ly = [];
for ily = 1:options.n_layer
    ly = [ly options.layers(ily)];
end
paras.layers = ly;
paras.n_layer = options.n_layer;
% ################################################################
paras.p = 2;
paras.max_iter = options.max_iter; 
paras.dim = nClas;
paras.nClas = nClas;
paras.G = G;
paras.nSmp_all = nSmp_all;
paras.nClas = length(unique(gnd));
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

AC = [];NMI = [];AR = [];
Fscore = [];Precision = [];Recall = [];
for inmf = 1:num_nmf 
    tic 
    [Z,H,Hc,lwndimnmf_obj] = LWNdimNMF_HAlign_out_Hc(fea,paras);
    W = Z;
    toc
end


end

function [Hinc,W,De,Dv]=Hpergraph2(V,k)
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
    Hinc(col,i) = HK(col,i); 
%     Hinc(col,i) = HK(col,i)/sum(HK(col,i));
end
W = diag(sum(Hinc,1));
% W = eye(n);
% get D_{e}
De = diag(sum(Hinc,1));
% get D_{v}
Dv = diag(sum(Hinc*W,2));

end
