% 阈值化处理子函数
function [CKSym, CSym] = BuildAdjacency(CMat,K1)

N = size(CMat,1);
CAbs = abs(CMat);
CSym = CAbs + CAbs';
if (K1 ~= 0)
    [~,Ind] = sort( CSym,1,'descend' ); % ascend
    CK = zeros(N,N);
    for i = 1:N
        for j = 1:K1
            CK( Ind(j,i),i ) = CSym( Ind(j,i),i );
        end
    end
    CKSym = CK + CK';
else
    CKSym = CSym;
end