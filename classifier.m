function [label, nm] = classifier(A, X, trls, imgsize)

classnum = numel(unique(trls));
nm = [];
for i=1:classnum
    pos = find(trls == i);
    Xi  = X(pos);
    Ai = A(:,pos);
    r = reshape(A*X - Ai*Xi, imgsize);
    nm = [nm sum(svd(r))];
end
index = find(nm==min(nm));
label = index(1);

end

