# CovApprox
This document describes the CovApprox code that can be used to run the simple RBMC, block 
RBMC and iterative interface methods desribed in Sidén et al. (2017, Efficient Covariance 
Approximations for Large Sparse Precision Matrices, arXiv preprint).

The code computes variances for the same model as in Table 1 in the paper, but could easily 
be modified to work for other models as well.

To run the code, first make sure the C++-files have been mexed correctly on your system,
by running mexAll.m. The code also depends on a function in the SuiteSparse library
(http://faculty.cse.tamu.edu/davis/suitesparse.html), CAMD. Make sure these
functions are also correctly mexed before running. The example code is in the script called 
covApprox.m. Use at own risk.

Most of the code was written by Per Sidén. The code for the Qinv-algorithm (the Takahashi 
equations) was originally written by Johan Lindström (Qinv.cpp and Qinv.m). QinvConv.m, 
QinvCond.cpp, QinvCondBreakEarly.m and QinvCondBreakEarly.cpp are modifications of that
code, modified by Per Sidén. QinvCond differs from the Qinv-algorithm in that the covariance 
matrix of the last elements has been pre-estimated and is given as input. Other covariance 
elements are computed given this. QinvCondBreakEarly is a further modifications which only 
computes covariance elements down to element BI and then breaks.
