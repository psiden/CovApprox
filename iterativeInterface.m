function [s2IterInter] = iterativeInterface(nBlocks,sz,Q,x)
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PURPOSE:      Iterative interface algorithm
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

        % plus (W)
        ij = [m(1)*ones(1,c(4)-c(3)-1),(c(1)+1):(c(2)-1);...
              (c(3)+1):(c(4)-1),m(2)*ones(1,c(2)-c(1)-1)];
        I(k).W = unique(sub2ind(sz,ij(1,:),ij(2,:)));
        
        % Divide W into inner plus (Z) and the rest (cZ)
        pe = floor(0.5 * (c + [m(1),m(1),m(2),m(2)]));
        if Wi(k) == 1; pe(1) = c(1);end
        if Wi(k) == Nx;pe(2) = c(2)-1;end
        if Wj(k) == 1; pe(3) = c(3);end
        if Wj(k) == Ny;pe(4) = c(4)-1;end
        ij = [m(1)*ones(1,pe(4)-pe(3)),(pe(1)+1):pe(2);...
              (pe(3)+1):pe(4),m(2)*ones(1,pe(2)-pe(1))];
        I(k).Z = unique(sub2ind(sz,ij(1,:),ij(2,:)));
        I(k).ZinW = find(ismember(I(k).W,I(k).Z));
        I(k).cZ = setdiff(I(k).W,I(k).Z);
        I(k).cZinW = find(ismember(I(k).W,I(k).cZ));

        % square boundary (V)
        ij = [c(1)*ones(1,c(4)-c(3)+1),c(2)*ones(1,c(4)-c(3)+1),c(1):c(2),c(1):c(2);...
              c(3):c(4),c(3):c(4),c(3)*ones(1,c(2)-c(1)+1),c(4)*ones(1,c(2)-c(1)+1)];
        ij = ij(:,ij(1,:)>0 & ij(1,:)<=sz(1) & ij(2,:)>0 & ij(2,:)<=sz(2));
        I(k).V = unique(sub2ind(sz,ij(1,:),ij(2,:)));

        % square filled (U)
        [ii,jj] = meshgrid(c(1):c(2),c(3):c(4));
        ij = [ii(:)';jj(:)'];
        ij = ij(:,ij(1,:)>0 & ij(1,:)<=sz(1) & ij(2,:)>0 & ij(2,:)<=sz(2));
        I(k).U = unique(sub2ind(sz,ij(1,:),ij(2,:)));
        I(k).N = length(I(k).U);
        I(k).WinU = find(ismember(I(k).U,I(k).W));
        I(k).VinU = find(ismember(I(k).U,I(k).V));

        % inner of square (cV)
        I(k).cV = setdiff(I(k).U,I(k).V);
        I(k).WincV = find(ismember(I(k).cV,I(k).W));

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

        % plus 3-dimensional (W)
        [ij1,ij2] = meshgrid((c(1)+1):(c(2)-1),(c(3)+1):(c(4)-1));
        [ik1,ik2] = meshgrid((c(1)+1):(c(2)-1),(c(5)+1):(c(6)-1));
        [jk1,jk2] = meshgrid((c(3)+1):(c(4)-1),(c(5)+1):(c(6)-1));
        ijk = [ij1(:) ij2(:) m(3)*ones((l(1)-2)*(l(2)-2),1);...
               ik1(:) m(2)*ones((l(1)-2)*(l(3)-2),1) ik2(:);...
               m(1)*ones((l(2)-2)*(l(3)-2),1) jk1(:) jk2(:)]';
        I(k).W = unique(sub2ind(sz,ijk(1,:),ijk(2,:),ijk(3,:)));   
        
        % Divide W into inner plus 3d (Z) and the rest (Zi and cZi)
        pe = floor(0.5 * (c + [m(1),m(1),m(2),m(2),m(3),m(3)]));
        if Wi(k) == 1; pe(1) = c(1);end
        if Wi(k) == Nx;pe(2) = c(2)-1;end
        if Wj(k) == 1; pe(3) = c(3);end
        if Wj(k) == Ny;pe(4) = c(4)-1;end
        if Wk(k) == 1; pe(5) = c(5);end
        if Wk(k) == Nz;pe(6) = c(6)-1;end
        [ij1,ij2] = meshgrid((pe(1)+1):pe(2),(pe(3)+1):pe(4));
        [ik1,ik2] = meshgrid((pe(1)+1):pe(2),(pe(5)+1):pe(6));
        [jk1,jk2] = meshgrid((pe(3)+1):pe(4),(pe(5)+1):pe(6));
        ijk = [ij1(:) ij2(:) m(3)*ones((pe(2)-pe(1))*(pe(4)-pe(3)),1);...
               ik1(:) m(2)*ones((pe(2)-pe(1))*(pe(6)-pe(5)),1) ik2(:);...
               m(1)*ones((pe(4)-pe(3))*(pe(6)-pe(5)),1) jk1(:) jk2(:)]';
        I(k).Z = unique(sub2ind(sz,ijk(1,:),ijk(2,:),ijk(3,:))); 
        I(k).cZ = setdiff(I(k).W,I(k).Z);
        I(k).ZinW = find(ismember(I(k).W,I(k).Z));
        I(k).cZinW = find(ismember(I(k).W,I(k).cZ));
                
        % cube border (V)
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
        I(k).WinU = find(ismember(I(k).U,I(k).W));
        I(k).VinU = find(ismember(I(k).U,I(k).V));

        % inner part of cube (cV)
        I(k).cV = setdiff(I(k).U,I(k).V);
        I(k).WincV = find(ismember(I(k).cV,I(k).W));

    end
end
    
% All interface nodes
W = unique([I.W]);
for k = 1:Ni
    I(k).WinW = find(ismember(W,I(k).W));
    I(k).ZinBigW = find(ismember(W,I(k).Z));
    I(k).cZinBigW = find(ismember(W,I(k).cZ));
    I(k).VinW = find(ismember(W,I(k).V));
    
    [Wi_tmp,Wj_tmp] = meshgrid(I(k).WinW,I(k).WinW);
    I(k).Wi = Wi_tmp(:)';I(k).Wj = Wj_tmp(:)';
end
W = unique([I.W]);
Wi = [I.Wi];Wj = [I.Wj];
iQW = sparse(Wi,Wj,1e-16*ones(length(Wi),1),length(W),length(W));
clear Wi Wj;
I = rmfield(I,{'Wi','Wj'});

% Phase 1, Local reordering, MC estimation 
for k = 1:Ni
    
    I(k).C = C(I(k).cV,I(k).cV);
    prio = ones(1,length(I(k).cV));
    prio(I(k).WincV) = 2;
    I(k).B = camd(I(k).C,1,prio);
    I(k).iB = zeros(1,length(I(k).cV)); I(k).iB(I(k).B) = 1:length(I(k).cV);
    WincVB = I(k).iB(I(k).WincV);
    cVQ = Q(I(k).cV,I(k).cV);
    cVR = chol(cVQ(I(k).B,I(k).B));
    WR = cVR(WincVB,WincVB);
    zthresh = 0; %1e-10; 
    Mtmp = cVR' \ Q(I(k).cV(I(k).B),I(k).V);
    Mtmp = Mtmp(WincVB,:);
    Msparse = (abs(Mtmp) > zthresh);
    [Mi,Mj] = ind2sub(size(Mtmp),find(Msparse));
    I(k).M = sparse(Mi,Mj,Mtmp(Msparse),size(Mtmp,1),size(Mtmp,2));
    Mx = Mtmp*x(I(k).V,:);
    Covtmp = speye(length(I(k).W));
    for i = 1:Ns
        Covtmp = Covtmp + 1/Ns*Mx(:,i)*Mx(:,i)';
    end
    MCCovar = WR \ Covtmp;
    MCCovar = MCCovar / WR';
    
    iQW(I(k).ZinBigW,I(k).ZinBigW) = MCCovar(I(k).ZinW,I(k).ZinW);
    iQW(I(k).ZinBigW,I(k).cZinBigW) = MCCovar(I(k).ZinW,I(k).cZinW);
    iQW(I(k).cZinBigW,I(k).ZinBigW) = MCCovar(I(k).cZinW,I(k).ZinW);
end

% Phase 2
niter = 1;
for i = 1:niter
    for k = 1:Ni
        WincVB = I(k).iB(I(k).WincV);
        cVQ = Q(I(k).cV,I(k).cV);
        cVR = chol(cVQ(I(k).B,I(k).B));
        WR = cVR(WincVB,WincVB);
        MSMT = I(k).M*iQW(I(k).VinW,I(k).VinW)*I(k).M';
        Covtmp = speye(length(I(k).W)) + MSMT;
        Covar = WR \ Covtmp;
        Covar = Covar / WR';

        iQW(I(k).WinW,I(k).WinW) = Covar;
    end
end

% Phase 3, Compute only Q:s sparse elements in last iteration
iQvec = zeros(nnz(C),1);
C2 = double(C);
C2(C) = 1:nnz(C);
for k = 1:Ni
    ifC = C(I(k).U,I(k).U);
    prio = ones(1,I(k).N);
    prio(I(k).VinU) = 2;
    B = camd(ifC,1,prio);
    [~,VB] = sort(B((end-length(I(k).V)+1):end));
    iVB = zeros(1,length(I(k).V)); iVB(VB) = 1:length(I(k).V);
    VinWB = I(k).VinW(iVB);
    C2_tmp = C2(I(k).U,I(k).U);
    CB2_tmp = C2_tmp(B,B);
    iQind = full(CB2_tmp(CB2_tmp > 0));
    CB2 = ifC(B,B);
    [~,~,~,~,SB_tmp] = symbfact(CB2);
    R_tmp = chol(Q(I(k).U(B),I(k).U(B)));
    invU = QinvCond(R_tmp,triu(SB_tmp),iQW(VinWB,VinWB));        
    iQvec(iQind) = invU(CB2);
end
[iQi,iQj] = ind2sub([N,N],find(C));
iQ = sparse(iQi,iQj,iQvec,N,N);
s2IterInter = diag(iQ);

