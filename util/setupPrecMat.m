%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PURPOSE:      Setup precision matrices
%               Q = G'*G
%               
% AUTHOR:       Per Siden
%               Division of Statistics and Machine Learning
%               Department of Computer and Information Science
%               Linkoping University      
%
% FIRST VER.:   2016-01-15
% REVISED:      2017-05-22
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
function [Q,G] = setupPrecMat(QType,N,sz,ndim)


if strcmp(QType,'eye') % Identity
    G = speye(N);
    Q = speye(N);

elseif strcmp(QType,'L') % Laplacian
    if ndim == 2
        Rr = spdiags([-1*ones(sz(1),1),2*ones(sz(1),1),-1*ones(sz(1),1)],[-1,0,1],sz(1),sz(1));
        Rc = spdiags([-1*ones(sz(2),1),2*ones(sz(2),1),-1*ones(sz(2),1)],[-1,0,1],sz(2),sz(2));
        L = kron(Rc,spdiags(ones(sz(1),1),0,sz(1),sz(1))) + ...
                 kron(spdiags(ones(sz(2),1),0,sz(2),sz(2)),Rr);
        Q = L;
        G = '';
    elseif ndim == 3
        Rx = spdiags([-1*ones(sz(1),1),2*ones(sz(1),1),-1*ones(sz(1),1)],[-1,0,1],sz(1),sz(1));
        Ry = spdiags([-1*ones(sz(2),1),2*ones(sz(2),1),-1*ones(sz(2),1)],[-1,0,1],sz(2),sz(2));
        Rz = spdiags([-1*ones(sz(3),1),2*ones(sz(3),1),-1*ones(sz(3),1)],[-1,0,1],sz(3),sz(3));
        Lxy = kron(Ry,spdiags(ones(sz(1),1),0,sz(1),sz(1))) + ...
              kron(spdiags(ones(sz(2),1),0,sz(2),sz(2)),Rx);
        L = kron(Rz,spdiags(ones(sz(1)*sz(2),1),0,sz(1)*sz(2),sz(1)*sz(2))) + ...
                 kron(spdiags(ones(sz(3),1),0,sz(3),sz(3)),Lxy);
        Q = L;
        G = '';
    end

elseif strcmp(QType,'L2') % squared Laplacian
    if ndim == 2
        Rr = spdiags([-1*ones(sz(1),1),2*ones(sz(1),1),-1*ones(sz(1),1)],[-1,0,1],sz(1),sz(1));
        Rc = spdiags([-1*ones(sz(2),1),2*ones(sz(2),1),-1*ones(sz(2),1)],[-1,0,1],sz(2),sz(2));
        L = kron(Rc,spdiags(ones(sz(1),1),0,sz(1),sz(1))) + ...
            kron(spdiags(ones(sz(2),1),0,sz(2),sz(2)),Rr);
        Q = L'*L;
        G = L;
    elseif ndim == 3
        Rx = spdiags([-1*ones(sz(1),1),2*ones(sz(1),1),-1*ones(sz(1),1)],[-1,0,1],sz(1),sz(1));
        Ry = spdiags([-1*ones(sz(2),1),2*ones(sz(2),1),-1*ones(sz(2),1)],[-1,0,1],sz(2),sz(2));
        Rz = spdiags([-1*ones(sz(3),1),2*ones(sz(3),1),-1*ones(sz(3),1)],[-1,0,1],sz(3),sz(3));
        Lxy = kron(Ry,spdiags(ones(sz(1),1),0,sz(1),sz(1))) + ...
              kron(spdiags(ones(sz(2),1),0,sz(2),sz(2)),Rx);
        L = kron(Rz,spdiags(ones(sz(1)*sz(2),1),0,sz(1)*sz(2),sz(1)*sz(2))) + ...
                 kron(spdiags(ones(sz(3),1),0,sz(3),sz(3)),Lxy);
        Q = L'*L;
        G = L;
    end

elseif strcmp(QType,'LI') % intrinsic Laplacian
    if ndim == 2
        Dr = spdiags([-1*ones(sz(1)-1,1),1*ones(sz(1)-1,1)],[0,1],sz(1)-1,sz(1));
        Dc =  spdiags([-1*ones(sz(2)-1,1),1*ones(sz(2)-1,1)],[0,1],sz(2)-1,sz(2));
        Drs = kron(speye(sz(2)),Dr);
        Dcs = kron(Dc,speye(sz(1)));
        mask = 1:N;
        mask_mat = zeros(sz); mask_mat(mask) = 1;
        hasColNeigh = mask_mat(1:(end-1),:) & ~diff(mask_mat);
        hasRowNeigh = mask_mat(:,1:(end-1)) & ~diff(mask_mat')';
        G = [Drs(hasColNeigh(:),mask);Dcs(hasRowNeigh(:),mask)];
        Q = G'*G;
    elseif ndim == 3
        Dx = spdiags([-1*ones(sz(1)-1,1),1*ones(sz(1)-1,1)],[0,1],sz(1)-1,sz(1));
        Dy = spdiags([-1*ones(sz(2)-1,1),1*ones(sz(2)-1,1)],[0,1],sz(2)-1,sz(2));
        Dz = spdiags([-1*ones(sz(3)-1,1),1*ones(sz(3)-1,1)],[0,1],sz(3)-1,sz(3));
        Dxs = kron(speye(sz(3)),kron(speye(sz(2)),Dx));
        Dys = kron(speye(sz(3)),kron(Dy,speye(sz(1))));
        Dzs = kron(Dz,kron(speye(sz(2)),speye(sz(1))));
        mask = 1:N;
        mask_mat = zeros(sz); mask_mat(mask) = 1;
        hasXNeigh = mask_mat(1:(end-1),:,:) & ~diff(mask_mat);
        hasYNeigh = mask_mat(:,1:(end-1),:) & ~diff(mask_mat,1,2);
        hasZNeigh = mask_mat(:,:,1:(end-1)) & ~diff(mask_mat,1,3);
        G = [Dxs(hasXNeigh(:),mask);Dys(hasYNeigh(:),mask);Dzs(hasZNeigh(:),mask)];
        Q = G'*G;
    end

elseif strcmp(QType,'L2I') % intrinsic squared Laplacian
    if ndim == 2
        Rr = spdiags([-1*ones(sz(1),1),2*ones(sz(1),1),-1*ones(sz(1),1)],[-1,0,1],sz(1),sz(1));
        Rc = spdiags([-1*ones(sz(2),1),2*ones(sz(2),1),-1*ones(sz(2),1)],[-1,0,1],sz(2),sz(2));
        L = kron(Rc,spdiags(ones(sz(1),1),0,sz(1),sz(1))) + ...
                 kron(spdiags(ones(sz(2),1),0,sz(2),sz(2)),Rr);
        L = L - diag(sum(L));
        Q = L'*L;
        G = L;
    elseif ndim == 3
        Rx = spdiags([-1*ones(sz(1),1),2*ones(sz(1),1),-1*ones(sz(1),1)],[-1,0,1],sz(1),sz(1));
        Ry = spdiags([-1*ones(sz(2),1),2*ones(sz(2),1),-1*ones(sz(2),1)],[-1,0,1],sz(2),sz(2));
        Rz = spdiags([-1*ones(sz(3),1),2*ones(sz(3),1),-1*ones(sz(3),1)],[-1,0,1],sz(3),sz(3));
        Lxy = kron(Ry,spdiags(ones(sz(1),1),0,sz(1),sz(1))) + ...
              kron(spdiags(ones(sz(2),1),0,sz(2),sz(2)),Rx);
        L = kron(Rz,spdiags(ones(sz(1)*sz(2),1),0,sz(1)*sz(2),sz(1)*sz(2))) + ...
                 kron(spdiags(ones(sz(3),1),0,sz(3),sz(3)),Lxy);
        L = L - diag(sum(L));
        Q = L'*L;
        G = L;
    end
else
    display('ERROR: Invalid Precision Matrix Type!');
end



