function [Z,H,Hc,obj] = LWNdimNMF_HAlign_out_Hc(X,options)
% modified from "LWNdimNMF", Hard Alignment is applied.
% layer wise normalized deep incomplete multiview NMF (LWNdimNMF)
% sum_v{ sum_l |Hv_(l-1)-Zv_(l)Hv_(l)|_F^2
% + sum_l!=nl alpha*tr(Hv_(l)*Lv*Hv_(l)') 
% + alpha*tr(Hc*Gv'*Lv*Gv*Hc') }
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
%     if n_layer > 1
        H = cell(n_view,n_layer-1);
%     else
%         H = cell(n_view,n_layer-1);
%     end
    for i = 1:n_view
        for ilayer = 1:n_layer
            if ilayer==1 && ilayer~=n_layer
                Z{i,ilayer} = rand(mFea{i},layers(ilayer));
                H{i,ilayer} = rand(layers(ilayer),nSmp{i});
            elseif ilayer~=1 && ilayer~=n_layer
                Z{i,ilayer} = rand(layers(ilayer-1),layers(ilayer));
                H{i,ilayer} = rand(layers(ilayer),nSmp{i});
            elseif ilayer==1 && ilayer==n_layer
                Z{i,ilayer} = rand(mFea{i},layers(ilayer));
                Hc = rand(layers(end),size(G{1},2));
            elseif ilayer~=1 && ilayer==n_layer
                Z{i,ilayer} = rand(layers(ilayer-1),layers(ilayer));
                Hc = rand(layers(end),size(G{1},2));
            else
                error('wrong initialization!!!!');
            end
        end
        Hc = rand(layers(end),size(G{1},2));
    end
else
    tic;
    Z = cell(n_view,n_layer);
    H = cell(n_view,n_layer-1);
    Ht = cell(n_view);
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
                if ilayer ~= n_layer
                    H{i,ilayer} = P';
                end
            else
%                 [F,P,~] = SemiNMF(H{i,ilayer-1},layers(ilayer),300);
                [F,P,~,~,~,~] = NMF(H{i,ilayer-1},layers(ilayer),opt_nmf,[],[]);
                Z{i,ilayer} = abs(F);
                if ilayer ~= n_layer
                    H{i,ilayer} = P';
                end
            end
            if ilayer == n_layer
                Ht{i} = P';
%                     Hc = Hc + 1/n_view*P'/G{i};
            end
            clear P
        end
    end
    Hc = get_Ht(G,Ht,n_view);
%     Hc = Hc + 1/n_view*H{i,end}*G{i};
    toc;
end
%% 
%{
for v = 1:n_view
    Norm = 1;
    NormV = 0;
    [Z{v,ilayer},H{v,ilayer}] = ...
        NormalizeUV(Z{v,ilayer},H{v,ilayer},NormV,Norm);
end
%}
%{
for v = 1:n_view
    Norm = 1;
    NormV = 0;
    for ilayer = 1:n_layer
        [Z{v,ilayer},H{v,ilayer}] = ...
            NormalizeUV(Z{v,ilayer},H{v,ilayer},NormV,Norm);
    end
end
%}
%% calculate obj at step 0
lwndimnmf_obj = [];
obj_rec1 = [];
obj_rec2 = [];
[new_obj,new_obj_main] = CalculateObj(X,Z,H,Hc,alpha,L,n_layer,G);
lwndimnmf_obj = [lwndimnmf_obj,new_obj];
obj_rec1 = [obj_rec1,new_obj_main(1)];
obj_rec2 = [obj_rec2,new_obj_main(2)];
%% start optimization
max_iter = options.max_iter;
iter = 0;
while  iter<=max_iter 
    iter = iter + 1;
    for v = 1:n_view
        for ilayer = 1:n_layer
        %################# updata Z1 and H1 ###################%
        %--------------------- update Z1 ----------------------%
            if ilayer==1 && ilayer~=n_layer
                XH = X{v}*H{v,ilayer}';
                WHH = Z{v,ilayer}*H{v,ilayer}*H{v,ilayer}';
            elseif ilayer>1 && ilayer~=n_layer
                XH = H{v,ilayer-1}*H{v,ilayer}';
                WHH = Z{v,ilayer}*H{v,ilayer}*H{v,ilayer}';
            elseif ilayer==n_layer && ilayer~=1
                XH = H{v,ilayer-1}*G{v}*Hc';
                WHH = Z{v,ilayer}*Hc*G{v}'*G{v}*Hc';
            else % ilayer==n_layer && ilayer==1
                XH = X{v}*G{v}*Hc';
                WHH = Z{v,ilayer}*Hc*G{v}'*G{v}*Hc';
            end
            
            Z{v,ilayer} = Z{v,ilayer}.*(XH./max(WHH,1e-10)); % 3mk
        %--------------------- update H1 ----------------------%
            if ilayer==1 && ilayer~=n_layer
                WX = Z{v,ilayer}'*X{v}; % mnk or pk (p<<mn)
                WWH = Z{v,ilayer}'*Z{v,ilayer}*H{v,ilayer}; % mk^2

                if ilayer+1~=n_layer
                    HcGv = Z{v,ilayer+1}*H{v,ilayer+1};
                    WX = WX + HcGv;
                    WWH = WWH + H{v,ilayer};

                    if alpha > 0 
                        WX = WX + alpha*H{v,ilayer}*S{v};
                        WWH = WWH + alpha*H{v,ilayer}*D{v};
                    end
                elseif ilayer+1==n_layer
                    HcGv = Z{v,ilayer+1}*Hc*G{v}';
                    WX = WX + HcGv;
                    WWH = WWH + H{v,ilayer};

                    if alpha > 0 
                        WX = WX + alpha*H{v,ilayer}*S{v};
                        WWH = WWH + alpha*H{v,ilayer}*D{v};
                    end
                end
                clear HcGv
            elseif ilayer>1 && ilayer~=n_layer
                WX = Z{v,ilayer}'*H{v,ilayer-1}; % mnk or pk (p<<mn)
                WWH = Z{v,ilayer}'*Z{v,ilayer}*H{v,ilayer}; % mk^2
                if ilayer+1~=n_layer
                    HcGv = Z{v,ilayer+1}*H{v,ilayer+1};
                    WX = WX + HcGv;
                    WWH = WWH + H{v,ilayer};

                    if alpha > 0 
                        WX = WX + alpha*H{v,ilayer}*S{v};
                        WWH = WWH + alpha*H{v,ilayer}*D{v};
                    end
                elseif ilayer+1==n_layer
                    HcGv = Z{v,ilayer+1}*Hc*G{v}';
                    WX = WX + HcGv;
                    WWH = WWH + H{v,ilayer};

                    if alpha > 0 
                        WX = WX + alpha*H{v,ilayer}*S{v};
                        WWH = WWH + alpha*H{v,ilayer}*D{v};
                    end
                end
                clear HcGv
            end
            
            if ilayer~=n_layer
                H{v,ilayer} = H{v,ilayer}.*(WX./max(WWH,1e-10));
            end
        end
    end
    %% update Hc
    if  n_layer ~= 1
        WX = 0;WWH = 0;
        for ivvv = 1:n_view
            WX = WX + Z{ivvv,n_layer}'*H{ivvv,n_layer-1}*G{ivvv};
            WWH = WWH + Z{ivvv,n_layer}'*Z{ivvv,n_layer}*Hc*G{ivvv}'*G{ivvv};

            if alpha > 0 
                WX = WX + alpha*Hc*G{ivvv}'*S{ivvv}*G{ivvv};
                WWH = WWH + alpha*Hc*G{ivvv}'*D{ivvv}*G{ivvv};
            end
        end
    elseif n_layer == 1
        WX = 0;WWH = 0;
        for ivvv = 1:n_view
            WX = WX + Z{ivvv,n_layer}'*X{ivvv}*G{ivvv};
            WWH = WWH + Z{ivvv,n_layer}'*Z{ivvv,n_layer}*...
                Hc*G{ivvv}'*G{ivvv};

            if alpha > 0 
                WX = WX + alpha*Hc*G{ivvv}'*S{ivvv}*G{ivvv};
                WWH = WWH + alpha*Hc*G{ivvv}'*D{ivvv}*G{ivvv};
            end
        end
    end
    Hc = Hc.*(WX./max(WWH,1e-10));
    
    [newobj,newobj_main] = CalculateObj(X,Z,H,Hc,alpha,L,n_layer,G);
    lwndimnmf_obj = [lwndimnmf_obj newobj]; %#ok<AGROW>
    obj_rec1 = [obj_rec1 newobj_main(1)];
    obj_rec2 = [obj_rec2 newobj_main(2)];
% % % %         disp(num2str(iter))
end
obj = [lwndimnmf_obj(end);obj_rec1(end);obj_rec2(end)];
% obj = [lwndimnmf_obj;obj_rec1;obj_rec2];


end
%==========================================================================
function [obj,obj_main] = CalculateObj_(X,Z,H,Hc,alpha,L,n_layer,G)
    n_view = length(X);
    obj_NMF = 0;
    obj_NMF_abs = 0;
    for v = 1:n_view
        ZZZZ = 1;
        for ilayer = 1:n_layer
            ZZZZ = ZZZZ*Z{v,ilayer};
        end
        dX = ZZZZ*Hc*G{v}'-X{v};
        sumDX = sum(sum(dX.^2,1));
        obj_NMF_abs = obj_NMF_abs + sumDX;
    end
    obj_main = obj_NMF_abs;
    obj = obj_NMF_abs;
    
end

function [obj,obj_main] = CalculateObj(X,Z,H,Hc,alpha,L,n_layer,G)
    n_view = length(X);
    obj_NMF = 0;
    obj_rec1 = 0;
    for v = 1:n_view
        for ilayer = 1:n_layer
            if ilayer == 1 && ilayer~=n_layer
                dX = Z{v,ilayer}*H{v,ilayer}-X{v};
                sumDX = sum(sum(dX.^2,1));
                obj_rec1 = obj_rec1 + sumDX;
                obj_NMF = obj_NMF + sumDX;
                if alpha > 0
                    obj_NMF = obj_NMF + ...
                        alpha*sum(sum(H{v,ilayer}.*(H{v,ilayer}*L{v})));
                end
            elseif ilayer > 1 && ilayer~=n_layer
                dX = Z{v,ilayer}*H{v,ilayer}-H{v,ilayer-1};
                sumDX = sum(sum(dX.^2,1));
                obj_rec1 = obj_rec1 + sumDX;
                obj_NMF = obj_NMF + sumDX;
                if alpha > 0
                    obj_NMF = obj_NMF + ...
                        alpha*sum(sum(H{v,ilayer}.*(H{v,ilayer}*L{v})));
                end
            elseif ilayer > 1 && ilayer==n_layer
                dX = Z{v,ilayer}*Hc*G{v}'-H{v,ilayer-1};
                sumDX = sum(sum(dX.^2,1));
                obj_rec1 = obj_rec1 + sumDX;
                obj_NMF = obj_NMF + sumDX;
                if alpha > 0
                    obj_NMF = obj_NMF + ...
                        alpha*sum(sum(Hc*G{v}'.*(Hc*G{v}'*L{v})));
                end
            elseif ilayer == 1 && ilayer==n_layer
                dX = Z{v,ilayer}*Hc*G{v}'-X{v};
                sumDX = sum(sum(dX.^2,1));
                obj_rec1 = obj_rec1 + sumDX;
                obj_NMF = obj_NMF + sumDX;
                if alpha > 0
                    obj_NMF = obj_NMF + ...
                        alpha*sum(sum(Hc*G{v}'.*(Hc*G{v}'*L{v})));
                end
            end
        end
    end
    %%%%%%%%%%%%%%% calculate obj_rec2 %%%%%%%%%%%%%%%
    obj_rec2 = 0;
    for v = 1:n_view
        ZZZZ = 1;
        for ilayer = 1:n_layer
            ZZZZ = ZZZZ*Z{v,ilayer};
        end
        dX = ZZZZ*Hc*G{v}'-X{v};
        sumDX = sum(sum(dX.^2,1));
        obj_rec2 = obj_rec2 + sumDX;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    obj_main = [obj_rec1,obj_rec2];
    
    obj = obj_NMF;
end

function [obj,obj_main] = CalculateObj__(X,Z,H,Hc,alpha,L,n_layer,G)
    n_view = length(X);
    obj_NMF = 0;
    obj_main = 0;
    for v = 1:n_view
        for ilayer = 1:n_layer
            if ilayer == 1 && ilayer~=n_layer
                dX = Z{v,ilayer}*H{v,ilayer}-X{v};
                sumDX = sum(sum(dX.^2,1));
                obj_main = obj_main + sumDX;
                obj_NMF = obj_NMF + sumDX;
                if alpha > 0
                    obj_NMF = obj_NMF + ...
                        alpha*sum(sum(H{v,ilayer}.*(H{v,ilayer}*L{v})));
                end
            elseif ilayer > 1 && ilayer~=n_layer
                dX = Z{v,ilayer}*H{v,ilayer}-H{v,ilayer-1};
                sumDX = sum(sum(dX.^2,1));
                obj_main = obj_main + sumDX;
                obj_NMF = obj_NMF + sumDX;
                if alpha > 0
                    obj_NMF = obj_NMF + ...
                        alpha*sum(sum(H{v,ilayer}.*(H{v,ilayer}*L{v})));
                end
            elseif ilayer > 1 && ilayer==n_layer
                dX = Z{v,ilayer}*Hc*G{v}'-H{v,ilayer-1};
                sumDX = sum(sum(dX.^2,1));
                obj_main = obj_main + sumDX;
                obj_NMF = obj_NMF + sumDX;
                if alpha > 0
                    obj_NMF = obj_NMF + ...
                        alpha*sum(sum(Hc*G{v}'.*(Hc*G{v}'*L{v})));
                end
            elseif ilayer == 1 && ilayer==n_layer
                dX = Z{v,ilayer}*Hc*G{v}'-X{v};
                sumDX = sum(sum(dX.^2,1));
                obj_main = obj_main + sumDX;
                obj_NMF = obj_NMF + sumDX;
                if alpha > 0
                    obj_NMF = obj_NMF + ...
                        alpha*sum(sum(Hc*G{v}'.*(Hc*G{v}'*L{v})));
                end
            end
        end
    end
    
    obj = obj_NMF;
end

function [Ht] = get_Ht(G,H,n_view)
    M = 0;
    for i = 1:n_view
        M = M + sum(G{i}',2);
    end
    Ht = 0;
    for i = 1:n_view
        Ht = Ht + H{i}/(G{i}')*diag(1./M);
    end
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

