function [Z,H,Hend,Hc,obj] = DeepNMF(X,options)
% modified from "ICMVNMF_MvDGNMF" and "ICMVNMF"
% layer wise normalized deep incomplete multiview NMF (LWNdimNMF)
% sum_v{ sum_l |Xv-Z1Z2Z3...ZLHL|_F^2
% + alpha |HL-Hc|_F^2
% + sum_l lambda_l*tr(Hv_(l)*Lv*Hv_(l)')}
% paper: 2020-Neurocomputing-"Deep graph regularized non-negative
% matrix factorization for multi-view clustering"
% beta: "we suggest a range as {0.1, 0.5, 1, 5, 10}".
% the weights of two layers:
% "recommend the parameters lambda1 and lambda2 as 10 and 1".
% first layer hidden component k1 can be selected from the range
% {20, 50, 100};components of the last layer, k2, are related to the
% categories of corresponding dataset.
n_view = length(X);
for i = 1:n_view
    X{i} = X{i}';
    [mFea{i},nSmp{i}] = size(X{i});
    % normalize columns of X{i} to unit vector as paper recomended.
    %     X{i} = X{i}*spdiags(sqrt(sum(X{i}.*X{i},1))',0,nSmp{i},nSmp{i});
end
n_layer = options.n_layer;% 2 in paper.
% dim = options.nClas;
alpha = options.alpha;
beta = options.beta;%
G = options.G;
% lambda_ls = options.lambda_ls;% lambda_ls = [10,1] as paper recommended.
layers = options.layers;% layers(1):{20, 50, 100};layers(2)=nClass

%% load graph laplacian of unlabeled samples
usehypergraph = 1;
if alpha > 0
    for i = 1:n_view
        if usehypergraph
            S{i} = sparse(options.Shyp{i});
            D{i} = options.Dhyp{i};
            L{i} = D{i} - S{i};
        else
            S{i} = sparse(options.Sg{i});
            D{i} = spdiags(full(sum(S{i},2)),0,nSmp{i},nSmp{i});
            L{i} = D{i} - S{i};
        end
    end
else
    % % % %     for i = 1:n_view
    % % % %         L{i} = zeros(nSmp,nSmp);
    % % % %     end
end
%% initialize W{i} H{i} Hc
init_nmf = 1;
if ~init_nmf
    Z = cell(n_view,n_layer);
    H = cell(n_view);
    Hc = 0;
    for i = 1:n_view
        for ilayer = 1:n_layer
            if ilayer==1
                Z{i,ilayer} = rand(mFea{i},layers(ilayer));
            else
                Z{i,ilayer} = rand(layers(ilayer-1),layers(ilayer));
            end
            H{i} = rand(layers(n_layer),nSmp{i});
        end
        Hc = Hc + 1/n_view*H{i}*G{i};
    end
else
    tic
    Z = cell(n_view,n_layer);
    H = cell(n_view);
    Hc = 0;
    opt_nmf.maxIter = 100;
    opt_nmf.NMFmaxIter = opt_nmf.maxIter;
    opt_nmf.nRepeat = 1;
    opt_nmf.NMFnRepeat = opt_nmf.nRepeat;
    opt_nmf.minIter = 100;
    opt_nmf.error = 1e-8;
    opt_nmf.meanFitRatio = 0.1;
    for i = 1:n_view
        for ilayer = 1:n_layer
            if ilayer==1
%                 [F,P,~] = SemiNMF(X{i},layers(ilayer),300);
                [F,P,~,~,~,~] = NMF(X{i},layers(ilayer),opt_nmf,[],[]);
                Z{i,ilayer} = abs(F);
                Htemp = P';
            else
%                 [F,P,~] = SemiNMF(Htemp,layers(ilayer),300);
                [F,P,~,~,~,~] = NMF(Htemp,layers(ilayer),opt_nmf,[],[]);
                Z{i,ilayer} = abs(F);
                clear Htemp
                Htemp = P';
            end
        end
        H{i} = Htemp;
        Hc = Hc + 1/n_view*H{i}*G{i};
    end
    toc
end
%%
%{
for v = 1:n_view
    Norm = 2;
    NormV = 0;
    for ilayer = 1:n_layer
        [Z{v,ilayer},H{v,ilayer}] = ...
            NormalizeUV(Z{v,ilayer},H{v,ilayer},NormV,Norm);
    end
end
%}
%% calculate obj at step 0
% lwndimnmf_obj = 0;
lwndimnmf_obj = [];
obj_m = [];
[new_obj,new_obj_main] = CalculateObj(X,Z,H,Hc,alpha,beta,L,n_layer,G);
lwndimnmf_obj = [lwndimnmf_obj,new_obj];
obj_m = [obj_m,new_obj_main];
%% start optimization
max_iter = options.max_iter;
iter = 0;
while  iter<=max_iter
    iter = iter + 1;
    for v = 1:n_view
        
        %################# updata Zvl and HvL ###################%
        %--------------------- update Zvl ----------------------%
        for ilayer = 1:n_layer
            %%%%%%%%%% phi %%%%%%%%%%
            if ilayer == n_layer
                phi = H{v};
            else
                phi = 1;
                for ii = ilayer+1:n_layer
                    phi = phi*Z{v,ii};
                end
                phi = phi*H{v};
            end
            %%%%%%%%%% psi %%%%%%%%%%
            if ilayer == 1
                psi = eye(mFea{v});
                
            elseif ilayer > 1
                psi = 1;
                for ii = 1:ilayer-1
                    psi = psi*Z{v,ii};
                end
            end
            
            if ilayer == 1
                fenzi = psi'*X{v}*phi';
                fenmu = psi'*psi*Z{v,ilayer}*(phi*phi');
                
                Z{v,ilayer} = Z{v,ilayer}.*(fenzi./max(fenmu,1e-10));
            elseif ilayer > 1
                fenzi = psi'*X{v}*phi';
                fenmu = psi'*psi*Z{v,ilayer}*(phi*phi');
                if alpha > 0
                    Zp = Z{v,ilayer}*phi;
                    seedS = Zp*S{v}*phi';
                    seedD = Zp*D{v}*phi';
%                     seedS = Z{v,ilayer}*phi*S{v}*phi';
%                     seedD = Z{v,ilayer}*phi*D{v}*phi';
                    if ilayer == 1
                        T = 0;
                    elseif ilayer > 1
                        T = 0;
                        for ii = 1:ilayer-1
% % % %                             if ii == 1 && ilayer-1 == 1
% % % %                                 M = 0;
% % % %                                 I = eye(size(Z{v,ii+1},1));
% % % %                             elseif ii == 1 && ilayer-1 ~= 1
% % % %                                 M = 0;
% % % %                                 I = eye(size(Z{v,ii+1},1));
                            if ii == 1
                                M = 0;
                                I = eye(size(Z{v,ii+1},1));
                            else
                                M = Z{v,ii};
                                I = eye(size(Z{v,ii},2));
                            end
                            T = I + M'*T*M;
                        end
                    end
                    seedS = alpha*T*seedS;
                    seedD = alpha*T*seedD;
                    
                    fenzi = fenzi + seedS;
                    fenmu = fenmu + seedD;
                end
                
                Z{v,ilayer} = Z{v,ilayer}.*(fenzi./max(fenmu,1e-10));
            end
        end
        clear seedS seedD T psi phi 
        %--------------------- update HvL ----------------------%
        
        psi = 1;
        for ii = 1:n_layer
            psi = psi*Z{v,ii};
        end
        fenzi = psi'*X{v};
        fenmu = psi'*psi*H{v};
        phi = eye(nSmp{v});
        if alpha > 0
%             seedS = Z{v,ilayer}*phi*S{v}*phi';
%             seedD = Z{v,ilayer}*phi*D{v}*phi';
            seedS = H{v}*S{v};
            seedD = H{v}*D{v};
 
%             ilayer = n_layer;
            if n_layer == 1
                T = eye(size(seedS,1));
            elseif n_layer > 1
                T = 0;
                for ii = 1:n_layer
% % % %                     if ii == 1 && n_layer-1 == 1
% % % %                         M = 0;
% % % %                         I = eye(size(Z{v,ii+1},1));
% % % %                     elseif ii == 1 && n_layer-1 ~= 1
% % % %                         M = 0;
% % % %                         I = eye(size(Z{v,ii+1},1));
                    if ii == 1
                        M = 0;
                        I = eye(size(Z{v,ii+1},1));
                    else
                        M = Z{v,ii};
                        I = eye(size(Z{v,ii},2));
                    end
                    T = I + M'*T*M;
                end
            end
            seedS = alpha*T*seedS;
            seedD = alpha*T*seedD;
 

            fenzi = fenzi + seedS;
            fenmu = fenmu + seedD;
        end
        if beta > 0
            HcGv = Hc*G{v}';
            fenzi = fenzi + beta*HcGv;
            fenmu = fenmu + beta*H{v};
        end
        
        H{v} = H{v}.*(fenzi./max(fenmu,1e-10));
        
        clear seedS seedD T 
    end
    %--------------------- update Hc ----------------------%
    HG = 0;GG = 0;
    for i = 1:n_view
        HG = HG + H{i}*G{i};
        GG = GG + G{i}'*G{i};
    end
    Hc = Hc.*(HG./max(Hc*GG,1e-10));

    [newobj,newobj_main] = CalculateObj(X,Z,H,Hc,alpha,beta,L,n_layer,G);
    lwndimnmf_obj = [lwndimnmf_obj newobj]; %#ok<AGROW>
    obj_m = [obj_m newobj_main];
    % %         disp(num2str(iter))
end
obj = [lwndimnmf_obj(end);obj_m(end)];
% obj = [lwndimnmf_obj;obj_m];

for v = 1:n_view
    Norm = 1;% original 1.
    NormV = 0;
    W = 1;
    for ilayer = 1:n_layer
        W = W*Z{v,ilayer};
    end
    [Wout,H{v}] = NormalizeUV(W, H{v}, NormV, Norm);
% % % %     [Wout,H{v}] = NormalizeUV(Z{v,end}, H{v}, NormV, Norm);
end
    
for v = 1:n_view
    Hend{v} = H{v};
end
    
end

%==========================================================================
function [obj,obj_main] = CalculateObj(X,Z,H,Hc,alpha,beta,L,n_layer,G)
n_view = length(X);
obj_lap = 0;
obj_main = 0;
obj_consis = 0;
for v = 1:n_view
    ZZZZ = 1;
    for ilayer = 1:n_layer
        ZZZZ = ZZZZ*Z{v,ilayer};
        if alpha > 0
            Zt = 1;
            if ilayer+1 <= n_layer
                for ll = ilayer+1:n_layer
                    Zt = Zt*Z{v,ll};
                end
            end
            Ht = Zt*H{v};
                
            obj_lap = obj_lap + ...
                alpha*sum(sum(Ht.*(Ht*L{v})));
        end
    end
    dX = ZZZZ*H{v}-X{v};
    sumDX = sum(sum(dX.^2));
    obj_main = obj_main + sumDX;
%     obj_NMF = obj_NMF + sumDX;
    if beta > 0
        dHc = H{v}-Hc*G{v}';
        obj_consis = obj_consis + beta*sum(sum(dHc.^2));
    end
        
end

obj = obj_main + obj_lap + obj_consis;
end


function [U, V] = NormalizeUV(U, V, NormV, Norm)
    K = size(U,2);
    if Norm == 2
        if NormV
            norms = max(1e-15,sqrt(sum(V.^2,1)));
            V = spdiags(norms.^-1,0,K,K)*V;
            U = U*spdiags(norms,0,K,K);
        else
            norms = max(1e-15,sqrt(sum(U.^2,1)))';
            U = U*spdiags(norms.^-1,0,K,K);
            V = spdiags(norms,0,K,K)*V;
        end
    else
        if NormV
            norms = max(1e-15,sum(abs(V),1));
            V = spdiags(norms.^-1,0,K,K)*V;
            U = U*spdiags(norms,0,K,K);
        else
            norms = max(1e-15,sum(abs(U),1))';
            U = U*spdiags(norms.^-1,0,K,K);
            V = spdiags(norms,0,K,K)*V;
        end
    end
end
