//#define TIMING
//To compile on Linux:
//mex QinvCond.cpp CFLAGS="\$CFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp"
//Author: Johan Lindstrom
//Modified by Per Siden 2017

#include "mex.h"
#include "matrix.h"
#include<cstdio>
#include<list>
#include<vector>
#include <string>
#ifdef _OPENMP
#include<omp.h>
#endif
#ifdef TIMING
#include<ctime>
#endif

typedef std::pair<size_t,double> Qpairtype;
typedef std::vector< Qpairtype > Qvectype;
typedef std::vector< Qvectype > Qtype;
typedef std::vector<size_t> Svectype;
typedef std::vector< Svectype > Stype;

/**Checks if a given array contains a square, full, real valued, double matrix.
 *@param tmp pointer to a matlab array.*/
bool mxIsRealSparseDouble(const mxArray* tmp){
  return mxIsSparse(tmp) && !mxIsComplex(tmp) &&
    mxIsDouble(tmp) && !mxIsEmpty(tmp);
}

/**Checks if a given array contains a real positive scalar.
 *@param tmp pointer to a matlab array.*/
bool mxIsRealPosScalar(const mxArray* tmp){
  return mxGetNumberOfElements(tmp)==1 && !mxIsComplex(tmp) &&
    mxIsDouble(tmp) && !mxIsEmpty(tmp);
}

/**Entry point for matlab interface (mex function).
 *A standard matlab entry point which generates a mex-file.
 */
//mexFunction entry point.
extern "C" void mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[]){
#ifdef TIMING
  clock_t start = clock();
  const double ticks_per_ms = static_cast<double>(CLOCKS_PER_SEC)/1000;
#endif
  //inital data check
  std::string sss = "sss";  
  if(nlhs>1)
    mexErrMsgTxt("Function returns ONE value.");
  if(nrhs<1 && nrhs>4)
    mexErrMsgTxt("Function requires 1-4 input(s).");
  if(!mxIsRealSparseDouble(prhs[0]))
    mexErrMsgTxt("First argument should be a sparse, double matrix.");
  if(!mxIsRealSparseDouble(prhs[1]))
    mexErrMsgTxt("Second argument should be a sparse, double matrix.");
//   if(!mxIsRealSparseDouble(prhs[2]))
//    mexErrMsgTxt("Third argument should be a sparse, double matrix.");
  int Nthreads=8;
  if(nrhs==4){
    if(!mxIsRealPosScalar(prhs[3]))
      mexErrMsgTxt("Fourth argument should be a real scalar.");
    double *ptrData = mxGetPr(prhs[3]);
    if(ptrData==NULL)
      mexErrMsgTxt("Unable to obtain pointers to fourth argument.");
    Nthreads = static_cast<int>( *ptrData );
    if(Nthreads<1)
      Nthreads=1;
  }
#ifdef _OPENMP
  //set nbr threads in openMP
  omp_set_dynamic(1);
  omp_set_num_threads(Nthreads);
#endif
  
  //Optain pointer to the elements in the R-matrix
  mwIndex *Rir = mxGetIr(prhs[0]);
  mwIndex *Rjc = mxGetJc(prhs[0]);
  double *Rpr = mxGetPr(prhs[0]);
  if(Rir==NULL || Rjc==NULL || Rpr==NULL)
    mexErrMsgTxt("Unable to obtain pointers to data in R-matrix.");
  size_t n =  mxGetN(prhs[0]);
  if(n != mxGetM(prhs[0]) )
    mexErrMsgTxt("R-matrix should be square");
  if(Rjc[n]==Rjc[n-1])
    mexErrMsgTxt("R-matrix has an empty last row.");
  
  //Optain pointer to the elements in the S-matrix
  mwIndex *Sir = mxGetIr(prhs[1]);
  mwIndex *Sjc = mxGetJc(prhs[1]);
  double *Spr = mxGetPr(prhs[1]);
  if(Sir==NULL || Sjc==NULL || Spr==NULL)
    mexErrMsgTxt("Unable to obtain pointers to data in S-matrix.");
  if(n != mxGetN(prhs[1]) || n != mxGetM(prhs[1]))
    mexErrMsgTxt("S-matrix should be of same size as R-matrix");
  if(Sjc[n]==Sjc[n-1])
    mexErrMsgTxt("S-matrix has an empty last row.");
    
  // Possibly load predefined last block of iQ, V
  size_t nLast;
  mwIndex *Vir,*Vjc;
  double *Vpr;
  if(!mxIsRealSparseDouble(prhs[2])){
    nLast = 0;
  } else {
    //Optain pointer to the elements in the V-matrix
    Vir = mxGetIr(prhs[2]);
    Vjc = mxGetJc(prhs[2]);
    Vpr = mxGetPr(prhs[2]);
    if(Vir==NULL || Vjc==NULL || Vpr==NULL)
      mexErrMsgTxt("Unable to obtain pointers to data in V-matrix.");
    nLast = mxGetN(prhs[2]);
    if(nLast != mxGetM(prhs[2]) )
      mexErrMsgTxt("V-matrix should be square");
  }
    
  size_t i,j,k;
  Qtype R(n);
  std::vector<double> D(n);
  //Extract the elements and store the sparse R-matrix
  //in a more convinient format.
  if(Rjc[n]-Rjc[n-1] == 1){
    //only one element in the last column, assume lower triangular matrix
    for(size_t c=0;c<n;++c){
      if(Rir[Rjc[c]] != c) //not a diagonal element
	mexErrMsgTxt("R-matrix is not lower triangular.");
      if(Sir[Sjc[c]] != c) //not a diagonal element
	mexErrMsgTxt("S-matrix is not lower triangular.");
      D[c] = Rpr[Rjc[c]];
      R[c].resize(Sjc[c+1]-Sjc[c]);
      k=Rjc[c];
      for(j=Sjc[c],i=0;j<Sjc[c+1];++j,++i){
        if(Sir[j]==Rir[k]){
          R[c][i] = Qpairtype(Sir[j], Rpr[k]);
          ++k;
        } else {
          R[c][i] = Qpairtype(Sir[j],0);
        }
      }
    }
  }else{
    //assume upper triangular matrix - first find number of element in each row
    std::vector<size_t> nRow(n), iRow(n), SnRow(n);
    for(size_t c=0;c<n;++c){
      for(j=Rjc[c];j<Rjc[c+1];++j)
        ++nRow[Rir[j]];
      if(Rir[Rjc[c+1]-1] != c) //not a diagonal element
        mexErrMsgTxt("R-matrix is not upper triangular.");
      D[c] = Rpr[Rjc[c+1]-1];
      for(j=Sjc[c];j<Sjc[c+1];++j)
        ++SnRow[Sir[j]];
      if(Sir[Sjc[c+1]-1] != c) //not a diagonal element
        mexErrMsgTxt("S-matrix is not upper triangular.");
    }
    for(size_t c=0;c<n;++c) {
      R[c].resize( SnRow[c] );
    }
    for(size_t c=0;c<n;++c){
      k=Rjc[c];
      for(j=Sjc[c];j<Sjc[c+1];++j)
        if(Sir[j]==Rir[k]){  
          R[Sir[j]][iRow[Sir[j]]++] = Qpairtype(c, Rpr[k]);
          ++k;
        } else {
          R[Sir[j]][iRow[Sir[j]]++] = Qpairtype(c,0);
        }
     }
  }//if(Rjc[n]-Rjc[n-1] == 1){ }else{ }
  
  Qvectype::iterator pos;
  size_t Nmax=0;
  //divide all elemnts in R by the diagonal-elements
  for(i=0; i<n; ++i){
    //find the maximal number of non-zero elements in any row of S
    if(Nmax < R[i].size())
      Nmax = R[i].size();
    //compute R[i,j]/D[i]
    for(pos=R[i].begin(); pos!=R[i].end(); ++pos)
      (pos->second) /=D[i];
    //and compute 1/d^2
    D[i] = 1/(D[i]*D[i]);
  }
  //count number of elements that is going to end up in iQ
  std::vector<size_t> nnz(n,1);
  for(i=0; i<n; ++i){
    //first find the indices of the non-zero elements
    for(pos=++(R[i].begin()); pos!=R[i].end(); ++pos){
      nnz[i]++;
      nnz[pos -> first]++;
    }
  }
 
  //vectors containing the location and values within one column
  std::vector<size_t> ii(Nmax);
  std::vector<double> s(Nmax);
  std::vector< Qvectype::iterator > iQpos(Nmax);
  std::vector< Qvectype::iterator > iQstart(n);
  //create a structure holding the inverse matrix
  Qtype iQ(n);
  for(i=0; i<n; ++i){
    iQ[i].resize(nnz[i]);
    iQstart[i] = iQ[i].end();
  }
  
#ifdef TIMING
  double compute_iQ_ind = 0;
  double compute_iQ_mult = 0;
  double compute_iQ_diag = 0;
  double compute_iQ_addEl = 0;
  double init_time = static_cast<double>(clock()-start)  / ticks_per_ms;
#endif

  //loop over the columns of the matrix
  i = n;
  while(i>0){
#ifdef TIMING
    start = clock();
#endif
    --i;
    //first find the indices of the non-zero elements
    for(pos=++(R[i].begin()), j=0; pos!=R[i].end(); ++pos, j++){
      ii[j] = pos->first; //index of elements
      s[j] = 0; //set values to zero
      iQpos[j] = iQstart[ii[j]]; //start of each iQ row
    }      
    
    double diag;
    if(i>=n-nLast){ //copy from V
      j = R[i].size()-1;
      k = Vjc[i+1-n+nLast]-1;
      while(j>0){
        --j;
        while((ii[j] < Vir[k]+n-nLast) && (k >= Vjc[i-n+nLast])) {  
          --k;
        }
        if(ii[j] == (Vir[k]+n-nLast)) {
          s[j] = -Vpr[k];
          --k;
        } else {
          s[j] = 0;
        } 
      }
      while((i < Vir[k]+n-nLast) && (k >= Vjc[i-n+nLast])) {
        --k;  
      }
      if(i == (Vir[k]+n-nLast)) {
        diag = Vpr[k];
      } else {
        mexErrMsgTxt("V-matrix has zeros on the diagonal");
      }
           
    } else {
#ifdef TIMING
      compute_iQ_ind += static_cast<double>(clock()-start) / ticks_per_ms;
      start = clock();
#endif
      //multiply the row of R with the rows of iQ
#pragma omp parallel for private(pos)
     for(int j2=0; j2<(R[i].size()-1); ++j2){
       Qvectype::iterator iQpos_tmp = iQpos[j2];
       Qvectype::iterator iQend = iQ[ii[j2]].end();
       for(pos=++(R[i].begin()); pos!=R[i].end(); ++pos){
         for(;iQpos_tmp != iQend && iQpos_tmp->first < pos->first; ++iQpos_tmp){}
         if(iQpos_tmp != iQend && iQpos_tmp->first == pos->first)
           s[j2] += (iQpos_tmp->second) * (pos->second);
          }
       }
#ifdef TIMING
      compute_iQ_mult += static_cast<double>(clock()-start)  / ticks_per_ms;
      start = clock();
#endif
    
      //the diagonal elements
      diag = D[i];
      for(pos=++(R[i].begin()), j=0; pos!=R[i].end(); ++pos, ++j)
        diag += s[j] * (pos->second);
        
#ifdef TIMING
      compute_iQ_diag += static_cast<double>(clock()-start)  / ticks_per_ms;
      start = clock();
#endif
    
    }
    
    //add the elements to iQ
    j = R[i].size()-1;
    while(j>0){
      --j;
      *(--iQstart[i]) = Qpairtype(ii[j], -s[j]);
      *(--iQstart[ ii[j] ]) = Qpairtype(i, -s[j]);
    }
    *(--iQstart[i]) = Qpairtype(i, diag);
#ifdef TIMING
    compute_iQ_addEl += static_cast<double>(clock()-start)  / ticks_per_ms;
#endif
  }//i = n; while(i>0){

#ifdef TIMING
  double compute_iQ = compute_iQ_ind + compute_iQ_mult +
    compute_iQ_diag + compute_iQ_addEl;
  start = clock();
#endif

  //count number of non-zero elements in iQ
  Nmax = 0;
  for(i=0; i<n; ++i)
    Nmax += iQ[i].size();
  
  //construct sparse matrix.
  plhs[0] = mxCreateSparse(n,n,Nmax,mxREAL);
  double* pr = mxGetPr(plhs[0]);
  mwIndex* ir = mxGetIr(plhs[0]);
  mwIndex* jc = mxGetJc(plhs[0]);

  jc[0] = 0;
  for(i=0, j=0; i<n; ++i){
    jc[i+1] = jc[i] + iQ[i].size();
    for(pos=iQ[i].begin(); pos!=iQ[i].end(); ++pos, ++j){
      ir[j] = pos->first;
      pr[j] = pos->second;
    }
  }

#ifdef TIMING
  double create_iQ = static_cast<double>(clock()-start)  / ticks_per_ms;
  mexPrintf("Timing info (in ms):\n");
  mexPrintf("     init: %7.1f\n",init_time);
  mexPrintf(" comp. iQ ind : %7.1f\n", compute_iQ_ind);
  mexPrintf(" comp. iQ mult: %7.1f\n", compute_iQ_mult);
  mexPrintf(" comp. iQ diag: %7.1f\n", compute_iQ_diag);
  mexPrintf(" comp. iQ add : %7.1f\n", compute_iQ_addEl);
  mexPrintf(" comp. iQ: %7.1f\n",compute_iQ);
  mexPrintf("create iQ: %7.1f\n",create_iQ);
  mexPrintf("    total: %7.1f\n", init_time + compute_iQ + create_iQ);
  //mexPrintf("\nNumber of threads: %i/%i\n",Nthreads,omp_get_max_threads());
  mexPrintf("\nNumber of threads: %i\n",Nthreads);
#endif
}//extern "C" void mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[])

/**\interface writeSparse
 *
   function writeSparse(fname,A)
 * WRITESPARSE Writes sparse matrix to a text file.
 *
 * writeSparse(filname,A)
 * filenam: Filename to write to.
 * A: Sparse matrix to be written.
   fid = fopen(fname,'w');
   [I,J,S] = find(A);
   I = I-1; 
   J=J-1;
   fwrite(fid,[size(A,1) size(A,2)],'uint32');
   for i=1:numel(I)
     fwrite(fid,[I(i) J(i)],'uint32');
     fwrite(fid,S(i),'float64');
   end
   fclose(fid);
 * Implemented in a MATLAB mex file using:
 *  writeSparse.cpp
 * Copyright 2007 Johan Lindstr?m
 */
