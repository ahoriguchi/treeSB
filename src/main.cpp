#include "RcppArmadillo.h"
#include "perturbedSN_pmc.h"
#include "perturbedSN_helpers.h"

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
Rcpp::List perturbedSNcpp(arma::mat Y,
							arma::mat psiX,
							arma::uvec C,
							Rcpp::List prior,
							Rcpp::List pmc,
							Rcpp::List state,
							Rcpp::List initParticles, bool init)
{
	Rcpp::RNGScope scope;  
	PMC H(Y, psiX, C, prior, pmc, state, initParticles, init);
	// List chain = H.get_chain();

	List chain;
	H.get_chain_v2(chain, true);

	// PMC* H(Y, psiX, C, prior, pmc, state, initParticles, init);
	// List chain = H->get_chain();
	// delete H;  // H is not needed again and is likely very large.
	
	// List data = Rcpp::List::create(  
	//   Rcpp::Named( "Y" ) = Y,
	//   Rcpp::Named( "psiX" ) = psiX,
	//   Rcpp::Named( "C" ) = C
	// ) ; 
	
	return Rcpp::List::create(  
		Rcpp::Named( "chain" ) = chain,
		// Rcpp::Named( "data" ) = data,
		Rcpp::Named( "prior" ) = prior,
		Rcpp::Named( "pmc" ) = pmc //,
		// Rcpp::Named( "control" ) = control
	) ;    
}