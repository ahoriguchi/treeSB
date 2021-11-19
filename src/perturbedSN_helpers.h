#ifndef HELPERS_H
#define HELPERS_H

#include "RcppArmadillo.h"

using namespace Rcpp;
using namespace arma;

arma::mat mvrnormArma(int n, arma::vec mu, arma::mat sigma);

arma::mat rWishartArma(arma::mat Sigma, int df);

double Eint( double xi, double om, double al );

double KL(  arma::vec xi0_1, 
            arma::vec xi0_2, 
            arma::mat Omega_1, 
            arma::mat Omega_2,
            arma::vec alpha_1,
            arma::vec alpha_2  );

arma::vec dmsnArma(arma::mat y, arma::rowvec xi, arma::mat omega, 
                   arma::vec alpha, bool logd = false);

int sampling(vec probs);

double ers_a_inf(double a);

double nrs_a_inf(double a);

double rtruncnormArma(double mu, double sigma, double a);

arma::vec dmvnrm_arma_precision(arma::mat x,  
                                arma::rowvec mean,  
                                arma::mat omega, 
                                bool logd = true);

double dIWishartArmaHelp(arma::mat& W, const double v, arma::mat& S, const bool logd = true, const double ldS=0, const double sgnS=0);

double dIWishartArma(arma::mat& W, const double v, arma::mat& S, const bool logd = true);

// double dIWishartArma(arma::mat W, double v, arma::mat S, bool logd = true);

arma::uvec randsamp(int n, int min, int max);

// void armadillo_set_seed_random();
// 
// void armadillo_set_seed(unsigned int val);

// void set_seed(unsigned int seed);

// int rndndn();


double rPG(double b=1.0, double c=0.0);

double rPG1z(double z);

arma::colvec rpg(arma::colvec shape, arma::colvec scale);

#endif