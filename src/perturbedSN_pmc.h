#ifndef GIBBS_H
#define GIBBS_H

#include "RcppArmadillo.h"
#include <omp.h>

using namespace Rcpp;
using namespace arma;

class PMC
{
private:
  mat Y;            // the data (n,J)
  mat psiX;         // the covariates for weights (n,R)
  uvec C;           // the group membership (n,1)
  int K;            // number of mixture components 
  int J;            // the number of groups
  int n;            // number of observations
  int p;            // observation dimension
  size_t R;         // number of covariate functions for weights
  int num_particles, num_iter, num_burnin, num_thin, num_display;   // number of iteration, burnin and thinning
  // int seed;         // initial random seed
  size_t treestr;
  
  /* --- hyperparameters --- */
  int length_chain;
  vec tau_a;
  double e0;
  vec b0;
  mat E0, invE0, B0, invB0;
  bool merge_step;
  double merge_par;
  double zeta;
  vec gam_mu;  // gamma R-variate-Normal mean 
  mat gam_Sig; // gamma R-variate-Normal Covariance 
  mat gam_Sig_inv; // inverse of gamma R-variate-Normal Covariance 
  
  /* --- initial values --- */
  uvec T;
  cube saveGam;  // save just the state for now

  /* --- storage --- */
  umat saveT;
  mat saveZ;
  cube saveXi, saveXi0, savePsi, saveAlpha, saveW;
  cube saveG, saveOmega, saveE;
  mat saveLog_py, savePerplexity;
  vec saveA0;
  umat saveNResampled;
  // field<vec> saveGam;

  /* --- functions --- */
  void main_loop(const Rcpp::List& prior, const Rcpp::List& initParticles, 
                 bool init);
  
  double sampleA0(double a0, const arma::umat& N, double a_par);
  
  // arma::mat sampleLogWs(const arma::umat& N, double a0);
  
  // arma::mat getWs();  
  
  void getLogWs(mat& logW);  
  // void getLogWsUT(mat& logW);  
  
  void sampleGam();
  // void sampleGamUT();
    
  Rcpp::List initialParticles();
  
  Rcpp::List sampleXi(const mat& Y_k, const uvec& C_k, uvec N_k, 
                      Rcpp::List particles);
  
  Rcpp::List sampleG(const mat& Y_k, const uvec& C_k, uvec N_k, 
                     Rcpp::List particles, Rcpp::List prior);
    
  Rcpp::List samplePsi(const mat& Y_k, const uvec& C_k, uvec N_k, 
                       Rcpp::List particles, Rcpp::List prior);
  
  Rcpp::List sampleZ(const mat& Y_k, const uvec& C_k, Rcpp::List particles);
  
  Rcpp::List sampleXi0(const mat& Y_k, uvec N_k, Rcpp::List particles);
  
  Rcpp::List sampleE(uvec N_k, Rcpp::List particles, Rcpp::List prior);
    
  arma::vec logPriorDens(Rcpp::List particles, Rcpp::List prior);

  arma::vec logPostDens(const mat& Y_k, const uvec& C_k, uvec N_k,
                        Rcpp::List particles, Rcpp::List prior);
    
  Rcpp::List iter(const uvec& T, int k, const umat& N, 
                  Rcpp::List particles, arma::mat log_dQ, Rcpp::List prior);

  arma::uvec sampleT(const arma::cube& xi, const arma::cube& Omega,
                     const arma::mat& alpha, const arma::mat& logW);
    
public:
  // constructor 
  PMC(arma::mat Y,
      arma::mat psiX,
      arma::uvec C,
      Rcpp::List prior,
      Rcpp::List pmc,
      Rcpp::List state,
      Rcpp::List initParticles, bool init);
  
  Rcpp::List get_chain();
};

#endif