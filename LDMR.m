function [Xn] = LDMR(y, D, trls, Xinit, Einit, alpha, beta, imgsize)

mu = 1;
eps_abs = 1e-3;
eps_rel = 1e-3;
Max_Iter = 100;
p = imgsize(1);
q = imgsize(2);
n = size(D,2);
Xn = Xinit;
En = Einit;
Zn = zeros(prod(imgsize),1);

distance = sum((D - y*ones(1, n)).^2)';
w = distance./max(distance);

classnum = length(unique(trls));
WA = [];
for i=1:length(trls)
    WA = [WA D(:,i)];
end
M1 = zeros(n, n);
M2 = zeros(n, n);
nc = 0;
for i=1:classnum
    pos = find(trls==i);
    M1(pos,pos) = D(:,pos)'*D(:,pos);
    M2(pos,pos) = D(:,pos)'*D(:,pos);
    nc  = nc + length(pos);
end
M = pinv(alpha/mu*M1+beta/mu*(diag(w.^2))+D'*D)*D';

for iter = 1:Max_Iter
    Zo = Zn;
    Xo = Xn;
    Eo = En;

    % update E
    m1 = reshape( D*Xo - y + Zo/mu, imgsize);
    [AU,SU,VU] = svd(m1,'econ');
    SU = diag(SU);
    SVP = length(find(SU>1/mu));
    if SVP >= 1
        SU = SU(1:SVP)-1/mu;
    else
        SVP = 1;
        SU = 0;
    end
    En = AU(:,1:SVP)*diag(SU)*VU(:,1:SVP)';
    En = En(:);

    % update X
    g = y + En - Zo/mu;
    Xn = M*g;

    % update Z
    Zn = Zo + mu*(D*Xn - En - y);

    % check the convergence condition
    eps_pri = sqrt(p*q)*eps_abs + eps_rel*max( max(norm(D*Xn,2),norm(En,2)), norm(y,2));
    eps_dual = sqrt(n)*eps_abs+eps_rel*norm(D'*Zn, 2);
    r = D*Xn - y - En;
    s = mu*D'*(En-Eo);
    convergence = (norm(r,2)<eps_pri) && (norm(s,2)<eps_dual);
    if convergence
        break;
    end
end
end


