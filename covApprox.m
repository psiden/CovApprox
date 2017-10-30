%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PURPOSE:      Example script to test the different methods in Sid?n et
%               al. (2017, Efficient Covariance Approximations for Large 
%               Sparse Precision Matrices).
%
% AUTHOR:       Per Siden
%               Division of Statistics and Machine Learning
%               Department of Computer and Information Science
%               Linkoping University      
%
% FIRST VER.:   2017-05-22
% REVISED:      2017-10-30
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Run small example

close all, clear all, clc
SuiteSparsePath = 'Enter path to SuiteSparse';
% rng(100);

addpath('util/');
addpath(strcat(SuiteSparsePath,'/SuiteSparse/CAMD/MATLAB'));

sz = [200,200];
% sz = [80,80,80];
compareWithExact = 1;
Ns = 20;
nBlocks = 10;
N = prod(sz);
lambda = .1*(1+rand(N,1));

% Posterior matrix Q = G'*G with first order random walk prior
[Q,G] = setupPrecMat('LI',N,sz,length(sz));
Q = Q + spdiags(lambda,0,N,N);
C = abs(Q) > 0;
P = amd(C); % AMD reordering
iP = zeros(1,N); iP(P) = 1:N;
QP = Q(P,P);

if compareWithExact
    disp('Computing exact selected inverse using the Takahashi equations.');
    tic
    RP = chol(QP);
    invQP = Qinv(RP);
    s2True = diag(invQP(iP,iP));
%     save('s2True.mat','s2True','-v7.3');
%     load('s2True.mat','s2True');if size(s2True,1) ~= N;error('loaded invQP has wrong size');end;
    disp(['Time:       ',num2str(toc),' s.']);
end
    
%% MC

disp('MC estimation');
tic 
x = sampleGMRF(G,QP,lambda,P,iP,Ns);
s2MC = 1/Ns * sum(x.^2,2);
s2CIMC = s2MC/Ns .* [chi2inv(.025,Ns),chi2inv(.975,Ns)];
disp(['Time:       ',num2str(toc),' s.']);
if compareWithExact
    maxerrMC = max(abs(s2MC ./ s2True - 1));
    RMSEMC = sqrt(mean((s2MC ./ s2True - 1).^2));
    disp(['Max error: ',num2str(100*maxerrMC),'%.   RMSE:',num2str(100*RMSEMC),'%.']);
end

%% Hutchinson

disp('Hutchinson estimation');
tic 
Vs = 2*(round(rand(N,Ns))-.5);
VsP = Vs(P,:);
MVs = zeros(N,Ns);
M1 = ichol(QP);
for i = 1:Ns
    [wP,flag,relres] = pcg(QP,VsP(:,i),1e-9,500,M1,M1',zeros(N,1));
    MVs(:,i) = wP(iP);
end

s2Hutch = dot(Vs,MVs,2) ./ dot(Vs,Vs,2);
disp(['Time:       ',num2str(toc),' s.']);
if compareWithExact
    maxerrHutch = max(abs(s2Hutch ./ s2True - 1));
    RMSEHutch = sqrt(mean((s2Hutch ./ s2True - 1).^2));
    disp(['Max error: ',num2str(100*maxerrHutch),'%.   RMSE:',num2str(100*RMSEHutch),'%.']);
end

%% Simple RBMC

disp('Simple RBMC estimation');
tic 
x = sampleGMRF(G,QP,lambda,P,iP,Ns);
diagQ = diag(Q);
s2SimpleRBMC = 1./diagQ + 1/Ns*sum(((Q-diag(diagQ))*x ./ diagQ).^2,2);
s2CISimpleRBMC = (1./diagQ) + 1/Ns*(s2SimpleRBMC-(1./diagQ)) .* ...
                   [chi2inv(.025,Ns),chi2inv(.975,Ns)];
disp(['Time:       ',num2str(toc),' s.']);
if compareWithExact
    maxerrSimpleRBMC = max(abs(s2SimpleRBMC ./ s2True - 1));
    RMSESimpleRBMC = sqrt(mean((s2SimpleRBMC ./ s2True - 1).^2));
    disp(['Max error: ',num2str(100*maxerrSimpleRBMC),'%.   RMSE:',num2str(100*RMSESimpleRBMC),'%.']);
end

%% Block RBMC

disp('Block RBMC estimation');
tic
x = sampleGMRF(G,QP,lambda,P,iP,Ns);
[s2BlockRBMC,s2CIBlockRBMC] = blockRBMC(nBlocks,sz,Q,x);

disp(['Time:       ',num2str(toc),' s.']);
if compareWithExact
    maxerrBlockRBMC = max(abs(s2BlockRBMC ./ s2True - 1));
    RMSEBlockRBMC = sqrt(mean((s2BlockRBMC ./ s2True - 1).^2));
    disp(['Max error: ',num2str(100*maxerrBlockRBMC),'%.   RMSE:',num2str(100*RMSEBlockRBMC),'%.']);
end

%% Iterative interface

disp('Iterative interface estimation');
tic
x = sampleGMRF(G,QP,lambda,P,iP,Ns);
s2IterInter = iterativeInterface(nBlocks,sz,Q,x);

disp(['Time:       ',num2str(toc),' s.']);
if compareWithExact
    maxerrIterInter = max(abs(s2IterInter ./ s2True - 1));
    RMSEIterInter = sqrt(mean((s2IterInter ./ s2True - 1).^2));
    disp(['Max error: ',num2str(100*maxerrIterInter),'%.   RMSE:',num2str(100*RMSEIterInter),'%.']);
end
