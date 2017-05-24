function x = sampleGMRF(G,QP,lambda,P,iP,Ns)
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PURPOSE:      Function for GMRF sampling using PCG
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
    N = size(QP,1);
    x = zeros(N,Ns);
    M1 = ichol(QP);
    NGQ = size(G,1);
    b = G' * randn(NGQ,Ns) + sqrt(lambda) .* randn(N,Ns);
    bP = b(P,:);
    for i = 1:Ns
        [wP,flag,relres] = pcg(QP,bP(:,i),1e-9,500,M1,M1',zeros(N,1));
        x(:,i) = wP(iP);
    end
end