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
  size_t K;            // number of mixture components 
  size_t J;            // the number of groups
  size_t n;            // number of observations
  size_t p;            // observation dimension
  size_t R;         // number of covariate functions for weights
  size_t num_particles, num_iter, num_burnin, num_thin, num_display;   // number of iteration, burnin and thinning
  // int seed;         // initial random seed
  size_t treestr;
  bool to_save_W;
  bool use_skew;
  
  /* --- hyperparameters --- */
  size_t length_chain;
  vec tau_a;
  double e0;
  mat Lamb, invLamb, cholLamb, cholInvLamb;
  bool is_Lamb_Zero;
  double ldLamb, sgnLamb;
  vec b0;
  // mat E0, invE0, B0, invB0;
  mat E0, invE0, cholE0, cholInvE0, B0, invB0, cholB0;
  double ldE0, sgnE0;
  bool merge_step;
  double merge_par;
  double zeta;
  vec gam_mu;  // gamma R-variate-Normal mean 
  mat gam_Sig; // gamma R-variate-Normal Covariance 
  mat gam_Sig_inv; // inverse of gamma R-variate-Normal Covariance 
  double m0;
  size_t min_nclu;
  
  /* --- initial values --- */
  uvec T;
  cube saveGam;  // save just the state for now

  /* --- storage --- */
  umat saveT;
  mat saveZ;
  cube saveXi, saveXi0, savePsi, saveAlpha, saveW;
  cube saveG, saveOmega, saveE;
  mat saveLog_py, savePerplexity;
  umat saveNResampled;

  /* --- functions --- */
  void main_loop(const Rcpp::List& initParticles, bool init);
  
  void getLogWs(mat& logW);  
  
  void sampleGam();
    
  Rcpp::List initialParticles();
  
  void sampleXi(const mat& Y_k, const uvec& C_k, const uvec& N_k, Rcpp::List& particles, mat& log_dQ, const size_t pp);
  
  void sampleG(const mat& Y_k, const uvec& C_k, const uvec& N_k, Rcpp::List& particles, mat& log_dQ, const size_t pp);
    
  void samplePsi(const mat& Y_k, const uvec& C_k, const uvec& N_k, Rcpp::List& particles, mat& log_dQ, const size_t pp);
  
  void sampleZ(const mat& Y_k, const uvec& C_k, Rcpp::List& particles, mat& log_dQ, const size_t pp);
  
  void sampleXi0(const mat& Y_k, const uvec& N_k, Rcpp::List& particles, mat& log_dQ, const size_t pp);
  
  void sampleE(const uvec& N_k, Rcpp::List& particles, mat& log_dQ, const size_t pp);
    
  arma::vec logPriorDens(Rcpp::List& particles);

  arma::vec logPostDens(const mat& Y_k, const uvec& C_k, const uvec& N_k, Rcpp::List& particles);
    
  Rcpp::List iter(const size_t k, const umat& N, Rcpp::List& all_particles);

  void sampleT(const arma::cube& xi, const arma::cube& Omega, const arma::mat& alpha, const arma::mat& logW);

  void clearNonSaves();
    
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

  void get_chain_v2(Rcpp::List& chain, bool to_clr_saves);
};

#endif