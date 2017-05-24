function [s2BlockRBMC] = blockRBMC(nBlocks,sz,Q,x)
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PURPOSE:      Block RBMC algorithm
%               
%               nBlocks - number of blocks in each dimension
%               sz - domain size
%               Q - precision matrix
%               x - posterior samples
%
% AUTHOR:       Per Siden
%               Division of Statistics and Machine Learning
%               Department of Computer and Information Science
%               Linkoping University      
%
% FIRST VER.:   2017-05-22
% REVISED:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ndim = length(sz);
Ns = size(x,2);
N = size(Q,1);
C = abs(Q) > 0;

% Set up block indices
if ndim == 2
    
    Nx = nBlocks;
    Ny = nBlocks;
    Ni = Nx * Ny;

    % Row/Column indices that make up interfaces
    Wx = ceil(linspace(0,sz(1)+1,Nx+2));
    Wy = ceil(linspace(0,sz(2)+1,Ny+2));
    [Wi_tmp,Wj_tmp] = meshgrid(1:Nx,1:Ny);
    Wi = Wi_tmp'; Wi = Wi(:); 
    Wj = Wj_tmp'; Wj = Wj(:);

    for k = Ni:-1:1
        I(k).dummy = 0;
    end

    % Subblock indices
    for k = 1:Ni

        % determine subblock corners, midpoints and lengths
        c = [Wx(Wi(k)) Wx(Wi(k)+2) Wy(Wj(k)) Wy(Wj(k)+2)];
        m = [Wx(Wi(k)+1) Wy(Wj(k)+1)];

        pe = floor(0.5 * (c + [m(1),m(1),m(2),m(2)]));
        if Wi(k) == 1; pe(1) = c(1);end
        if Wi(k) == Nx;pe(2) = c(2)-1;end
        if Wj(k) == 1; pe(3) = c(3);end
        if Wj(k) == Ny;pe(4) = c(4)-1;end

        % Small inner square
        [ii,jj] = meshgrid((pe(1)+1):pe(2),(pe(3)+1):pe(4));
        ij = [ii(:)';jj(:)'];
        I(k).Y = unique(sub2ind(sz,ij(1,:),ij(2,:))); 

        % Outer square border (V)
        ij = [c(1)*ones(1,c(4)-c(3)+1),c(2)*ones(1,c(4)-c(3)+1),c(1):c(2),c(1):c(2);...
              c(3):c(4),c(3):c(4),c(3)*ones(1,c(2)-c(1)+1),c(4)*ones(1,c(2)-c(1)+1)];
        ij = ij(:,ij(1,:)>0 & ij(1,:)<=sz(1) & ij(2,:)>0 & ij(2,:)<=sz(2));
        I(k).V = unique(sub2ind(sz,ij(1,:),ij(2,:))); 

        % cube filled (U)
        [ii,jj] = meshgrid(c(1):c(2),c(3):c(4));
        ij = [ii(:)';jj(:)'];
        ij = ij(:,ij(1,:)>0 & ij(1,:)<=sz(1) & ij(2,:)>0 & ij(2,:)<=sz(2));
        I(k).U = unique(sub2ind(sz,ij(1,:),ij(2,:)));
        I(k).N = length(I(k).U); 

        % inner part of cube (cV)
        I(k).cV = setdiff(I(k).U,I(k).V);
        I(k).YincV = find(ismember(I(k).cV,I(k).Y));

    end
    
elseif ndim == 3
    
    Nx = nBlocks;
    Ny = nBlocks;
    Nz = nBlocks;
    Ni = Nx * Ny * Nz;

    % Row/Column indices that make up interfaces
    Wx = ceil(linspace(0,sz(1)+1,Nx+2));
    Wy = ceil(linspace(0,sz(2)+1,Ny+2));
    Wz = ceil(linspace(0,sz(3)+1,Nz+2));
    [Wi_tmp,Wj_tmp,Wk_tmp] = meshgrid(1:Nx,1:Ny,1:Nz);
    Wi = reshape(permute(Wi_tmp,[2,1,3]),Ni,1); 
    Wj = reshape(permute(Wj_tmp,[2,1,3]),Ni,1); 
    Wk = reshape(permute(Wk_tmp,[2,1,3]),Ni,1);

    for k = Ni:-1:1
    I(k).dummy = 0;
    end

    % Subblock indices
    for k = 1:Ni

        % determine subblock corners, midpoints and lengths
        c = [Wx(Wi(k)) Wx(Wi(k)+2) Wy(Wj(k)) Wy(Wj(k)+2) Wz(Wk(k)) Wz(Wk(k)+2)];
        m = [Wx(Wi(k)+1) Wy(Wj(k)+1) Wz(Wk(k)+1)];
        l = [c(2) - c(1) + 1,c(4) - c(3) + 1,c(6) - c(5) + 1];

        pe = floor(0.5 * (c + [m(1),m(1),m(2),m(2),m(3),m(3)]));
        if Wi(k) == 1; pe(1) = c(1);end
        if Wi(k) == Nx;pe(2) = c(2)-1;end
        if Wj(k) == 1; pe(3) = c(3);end
        if Wj(k) == Ny;pe(4) = c(4)-1;end
        if Wk(k) == 1; pe(5) = c(5);end
        if Wk(k) == Nz;pe(6) = c(6)-1;end

        % Small inner cube 3d
        [ijk1,ijk2,ijk3] = meshgrid((pe(1)+1):pe(2),(pe(3)+1):pe(4),(pe(5)+1):pe(6));
        ijk = [ijk1(:) ijk2(:) ijk3(:)]';
        I(k).Y = unique(sub2ind(sz,ijk(1,:),ijk(2,:),ijk(3,:))); 

        % Outer cube border (V)
        [ij1,ij2] = meshgrid(c(1):c(2),c(3):c(4));
        [ik1,ik2] = meshgrid(c(1):c(2),(c(5)+1):(c(6)-1));
        [jk1,jk2] = meshgrid((c(3)+1):(c(4)-1),(c(5)+1):(c(6)-1));
        ijk = [ij1(:) ij2(:) c(5)*ones(l(1)*l(2),1);...
               ij1(:) ij2(:) c(6)*ones(l(1)*l(2),1);...
               ik1(:) c(3)*ones(l(1)*(l(3)-2),1) ik2(:);...
               ik1(:) c(4)*ones(l(1)*(l(3)-2),1) ik2(:) ;...                
               c(1)*ones((l(2)-2)*(l(3)-2),1) jk1(:) jk2(:);...
               c(2)*ones((l(2)-2)*(l(3)-2),1) jk1(:) jk2(:)]';
        ijk = ijk(:,ijk(1,:)>0 & ijk(1,:)<=sz(1) & ijk(2,:)>0 & ...
                    ijk(2,:)<=sz(2) & ijk(3,:)>0 & ijk(3,:)<=sz(3));
        I(k).V = unique(sub2ind(sz,ijk(1,:),ijk(2,:),ijk(3,:)));  

        % cube filled (U)
        [ii,jj,kk] = meshgrid(c(1):c(2),c(3):c(4),c(5):c(6));
        ijk = [ii(:)';jj(:)';kk(:)'];
        ijk = ijk(:,ijk(1,:)>0 & ijk(1,:)<=sz(1) & ijk(2,:)>0 & ...
                    ijk(2,:)<=sz(2) & ijk(3,:)>0 & ijk(3,:)<=sz(3));
        I(k).U = unique(sub2ind(sz,ijk(1,:),ijk(2,:),ijk(3,:)));  
        I(k).N = length(I(k).U); 

        % inner part of cube (cV)
        I(k).cV = setdiff(I(k).U,I(k).V);
        I(k).YincV = find(ismember(I(k).cV,I(k).Y));

    end

end

% All inner cube nodes
Y = unique([I.Y]);
for k = 1:Ni
    I(k).YinY = find(ismember(Y,I(k).Y));
end

% Estimation algorithm
s2BlockRBMC = zeros(N,1);
for k = 1:Ni

    I(k).C = C(I(k).cV,I(k).cV);
    prio = ones(1,length(I(k).cV));
    prio(I(k).YincV) = 2;
    I(k).B = camd(I(k).C,1,prio);
    I(k).iB = zeros(1,length(I(k).cV)); I(k).iB(I(k).B) = 1:length(I(k).cV);
    YincVB = I(k).iB(I(k).YincV);
    cVQ = Q(I(k).cV,I(k).cV);
    cVR = chol(cVQ(I(k).B,I(k).B));

    minind = min(YincVB);    
    [~,~,~,~,S_tmp] = symbfact(I(k).C(I(k).B,I(k).B));
    MCCovar2 = QinvCondBreakEarly(cVR,triu(S_tmp),[],minind);
    MCCovar2 = MCCovar2(YincVB,YincVB);
    kappa = cVR' \ (Q(I(k).cV(I(k).B),I(k).V)*x(I(k).V,:));
    kappa = cVR \ kappa; 

    MCCovar3 = diag(MCCovar2);
    for i = 1:Ns
        MCCovar3 = MCCovar3 + 1/Ns*kappa(YincVB,i).^2;
    end
    s2BlockRBMC(I(k).Y) = MCCovar3;
end