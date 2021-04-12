#include "RcppArmadillo.h"
#include "perturbedSN_pmc.h"
#include "perturbedSN_helpers.h"


using namespace Rcpp;
using namespace arma;
using namespace std;

const double log2pi = std::log(2.0 * M_PI);

PMC::PMC(arma::mat Y,
         arma::mat psiX,
         arma::uvec C,
         Rcpp::List prior,
         Rcpp::List pmc,
         Rcpp::List state, 
         Rcpp::List initParticles,
         bool init) : Y(Y), psiX(psiX), C(C)
{
  // Rcout << "start of PMC::PMC()" << endl;
  p = Y.n_cols;
  n = Y.n_rows;
  J = C.max() + 1;
  K = Rcpp::as<int>(prior["K"]);
  R = psiX.n_cols;
  
  num_particles = Rcpp::as<int>(pmc["npart"]);
  num_iter = Rcpp::as<int>(pmc["nskip"]) * Rcpp::as<int>(pmc["nsave"]);
  num_burnin = Rcpp::as<int>(pmc["nburn"]);
  num_thin = Rcpp::as<int>(pmc["nskip"]);
  num_display = Rcpp::as<int>(pmc["ndisplay"]);
  
  // seed = Rcpp::as<int>(pmc["seed"]);
  
  
  length_chain = num_iter/num_thin;
  
  saveT.set_size(length_chain, n);
  saveZ.set_size(length_chain, n);
  saveW.set_size(n, K, length_chain);  // saveW.set_size(J, K, length_chain);
  saveXi.set_size(J, p*K, length_chain);
  saveXi0.set_size(p, K, length_chain);
  saveE.set_size(p, K*p, length_chain);
  savePsi.set_size(p, K, length_chain);
  saveG.set_size(p, K*p, length_chain);
  // saveS.set_size(num_particles, K, length_chain);
  // saveS.set_size(length_chain, K);
  // saveS.fill(1);
  // saveVarphi.set_size(length_chain);
  saveOmega.set_size(p, K*p, length_chain);
  saveAlpha.set_size(p, K, length_chain);
  saveA0.set_size(length_chain);
  saveLog_py.set_size(length_chain, K);
  saveNResampled.set_size(length_chain, K);
  savePerplexity.set_size(length_chain, K);

  T = Rcpp::as<uvec>(state["t"]);

  tau_a = Rcpp::as<vec>(prior["tau_a"]);
  e0 = Rcpp::as<double>(prior["e0"]);
  E0 = Rcpp::as<mat>(prior["E0"]);
  invE0 = inv_sympd(E0);
  b0 = Rcpp::as<vec>(prior["b0"]);
  B0 = Rcpp::as<mat>(prior["B0"]);
  invB0 = inv_sympd(B0);
  merge_step = Rcpp::as<bool>(prior["merge_step"]);
  merge_par = Rcpp::as<double>(prior["merge_par"]);
  zeta = Rcpp::as<double>(prior["zeta"]);
  gam_mu = Rcpp::as<vec>(prior["gam_mu"]);
  gam_Sig = Rcpp::as<mat>(prior["gam_Sig"]);
  gam_Sig_inv = arma::inv(gam_Sig);
  
  // saveGam = mvrnormArma(K, gam_mu, gam_Sig).t();   
  
  // XX: This is inefficient because of needless copying. 
  saveGam.set_size(J, K-1, R);
  mat tmp = mvrnormArma(J*(K-1), gam_mu, gam_Sig);  // to perform only one matrix inverse
  // Rcout << "tmp: " << tmp.n_rows << " x " << tmp.n_cols << "matrix" << endl;
  for (size_t j=0; j<J; j++)
    for (size_t m=0; m<(K-1); m++)
      saveGam.tube(j, m) = tmp.row((K-1)*j + m);


  //   saveParticles = control$saveParticles
  //   if(saveParticles) {
  //     outFolder = control$outFolder
  //     if(!(outFolder %in% dir())) {
  //       dir.create(outFolder, recursive=T)
  //     }
  //     if(!('Iterations' %in% dir(outFolder))){
  //       dir.create(paste0(outFolder, '/Iterations'), recursive=T)
  //     }
  //   }
  //   verbose = control$verbose
  
  main_loop(prior, initParticles, init);
}

void PMC::main_loop(const Rcpp::List& prior, const Rcpp::List& initParticles, 
                    bool init) {
  // Rcout << "start of main_loop()" << endl;

  int km = 0;
  
  umat N(J, K);
  mat logW(n, K);
  // mat logW(J, K);
  // logW.fill(log(1.0/K));
  cube xi(J, p, K);
  mat xi0(p, K);
  mat psi(p, K);
  cube G(p,p,K);
  cube E(p,p,K);
  cube Omega(p,p,K);
  mat alpha(p,K);
  rowvec z(n);

  // double a = 1;
  double a0 = 1;
  // double a_old = a;
  double a0_old = a0;
  double a_par  = sqrt(K);
  // double a0_par  = sqrt(K);
  double a_count = 0; 
  int a_tot = 100; 

  
  mat log_dQ(num_particles, 8);
  log_dQ.fill(0);
  vec log_py(K);
  vec perplexity(K);
  uvec nResampled(K);

  Rcpp::List all_particles;
  if (init) {
    Rcout << "initializing all particles..." << endl;
    // all_particles = Rcpp::as<Rcpp::List>(initialParticles( T ) ); 
    all_particles = Rcpp::as<Rcpp::List>(initialParticles()); 
    Rcout << "Done" << endl;
  } else {
    Rcout << "using fixed initial particles" << endl;
    all_particles = Rcpp::as<Rcpp::List>(initParticles);
  }
  
  for (int it=0; it<(num_iter+num_burnin); it++) {
    if ((it+1)%num_display == 0)
      Rcout << "Iteration: " << it + 1 << " of " << num_iter + num_burnin << endl;
    
    N.fill(0);
    for (size_t i=0; i<n; i++) { N(C(i), T(i))++; }
    
    if (merge_step && it>0) {
      for (size_t k=0; k < K-1 ; k++) {
        if (sum(N.col(k)) > 0) {
          for (size_t kk=k+1; kk < K ; kk++) {
            if (sum(N.col(kk)) > 0) {
              double kl_div = KL( xi0.col(k),
                                  xi0.col(kk),
                                  Omega.slice(k),
                                  Omega.slice(kk),
                                  alpha.col(k),
                                  alpha.col(kk) );
              if( kl_div < R::qchisq(merge_par, (double)p, 1, 0) ) {
                N.col(k) = N.col(k) + N.col(kk);
                N.col(kk) = zeros<uvec>(J);
                T(find(T==kk)).fill(k);
                Rcout << "Merged clusters (iteration " << it+1 << ")" << endl;
                if( (it+1 > num_burnin) && ((it+1) % num_thin == 0)) {
                  Rcout << "Merged clusters after burn-in period (iteration " << it+1 << "). Consider longer burn-in." << endl;
                }
              }
            }
          }
        }
      }
    }
    
    // Rcout << "start of a0_old" << endl;
    a0_old = a0;
    a0 = sampleA0( a0, N, a_par );
    
    if (a0 != a0_old)
      a_count++;
    if ( (it <= num_burnin) && ((it+1) % a_tot == 0) ) {
      if (a_count < 30)
        a_par *= 1.1;
      if (a_count > 50)
        a_par *= 0.9;
      a_count = 0;
    } 

    // Rcout << "psiX (nrow=" << psiX.n_rows << ", ncol=" << psiX.n_cols << ")" << endl;
    // Rcout << "** before sampleGam()" << endl;
    sampleGam();  // logW = sampleLogWsCov(N, a0);
    getLogWs(logW);
    // Rcout << "** after sampleGam()" << endl;
    
    for (size_t k=0; k<K; k++) {
      Rcpp::List iterSummary = iter( T, k, N, all_particles[k], log_dQ, //varphi, 
                                     prior);

      all_particles[k] = Rcpp::as<Rcpp::List>(iterSummary["particles"]);
      
      Rcpp::List temp = all_particles[k];
      
      xi.slice(k) = mean(Rcpp::as<cube>(temp["xi"]), 2);
      xi0.col(k) = mean(Rcpp::as<mat>(temp["xi0"]),0).t();
      psi.col(k) = mean(Rcpp::as<mat>(temp["psi"]),0).t();
      G.slice(k) = reshape(mean(Rcpp::as<mat>(temp["G"]),0), p ,p);
      E.slice(k) = reshape(mean(Rcpp::as<mat>(temp["E"]),0), p ,p);
      z.cols(find(T==k)) = mean(Rcpp::as<mat>(temp["z"]),0);

      Omega.slice(k) = G.slice(k) + psi.col(k) * psi.col(k).t();
      mat inv_Omega = inv_sympd(Omega.slice(k));
      vec numerator = arma::sqrt(diagmat(Omega.slice(k).diag())) * inv_Omega * psi.col(k);
      double denominator = as_scalar(arma::sqrt(1 - psi.col(k).t() * inv_Omega * psi.col(k)) );
      alpha.col(k) = numerator/denominator;

      log_py(k) = Rcpp::as<double>(iterSummary["log_py"]);
      perplexity(k) = Rcpp::as<double>(iterSummary["perplexity"]);
      nResampled(k) = Rcpp::as<int>(iterSummary["nResampled"]);
    }
    
    // Rcout << "** before sampleT()" << endl;
    T = sampleT(xi, Omega, alpha, logW);
    // Rcout << "** after sampleT()" << endl;
    
    if( (it+1 > num_burnin) && ((it+1) % num_thin == 0))
    {
      saveT.row(km) = T.t();
      saveZ.row(km) = z;
      saveW.slice(km) = exp(logW);  // saveW.slice(km) = exp(logW);
      saveXi.slice(km) = reshape( mat(xi.memptr(), xi.n_elem, 1, false), J, K*p);
      saveXi0.slice(km) = xi0;
      savePsi.slice(km) = psi;
      saveG.slice(km) = reshape( mat(G.memptr(), G.n_elem, 1, false), p, K*p);
      saveE.slice(km) = reshape( mat(E.memptr(), E.n_elem, 1, false), p, K*p);
      saveLog_py.row(km) = log_py.t();
      saveA0(km) = a0;
      savePerplexity.row(km) = perplexity.t();
      saveNResampled.row(km) = nResampled.t();
      saveOmega.slice(km) = reshape( mat(Omega.memptr(), Omega.n_elem, 1, false), p, K*p);
      saveAlpha.slice(km) = alpha;
      km++;
    }
  } // end it loop
  // Rcout << "end of main_loop()" << endl;
}

//--------------------------------------------------------


/* slow version (slow by necessity?) */
// Polya-gamma data augmentation per 2021 Rigon Durante
// saveGam is (J, K-1, R)
void PMC::sampleGam() {
  
  for (size_t m=0; m<(K-1); m++) {

    size_t levelm = floor(log2(m+1) * (1 + std::numeric_limits<double>::epsilon()));
    size_t Km = K / pow(2, levelm);  // number of leafs under m
    size_t lfl = (m+1) * Km - K;   // Subtract K to shift from K:(2K-1) to 0:(K-1)

    for(size_t j=0; j<J; j++) {
      
      uvec Cj_and_k_under_m = arma::find(lfl<=T && T<(lfl+Km) && C==j);
      uvec Tij = T(Cj_and_k_under_m);
      mat psiXij = psiX.rows(Cj_and_k_under_m);  
      
      // 1. Update PG data
      vec gamtmp = saveGam.tube(j, m);  // Compiler wants me to store this before multiplying.
      vec pgdat = rpg(ones<vec>(Tij.size()), psiXij * gamtmp);
      
      // 2. Update gamma
      vec kappa(Tij.size());
      kappa.fill(-0.5);
      kappa(arma::find( Tij<(lfl+Km/2) )).fill(0.5);
      mat gam_Sigjm = arma::inv(psiXij.t() * diagmat(pgdat) * psiXij + gam_Sig_inv);
      vec gam_mujm = gam_Sigjm * (psiXij.t() * kappa + gam_Sig_inv * gam_mu);
      saveGam.tube(j, m) = mvrnormArma(1, gam_mujm, gam_Sigjm).t();
      
    }
  }
}


// Returns log weights, which is a deterministic function of psiX and gam, 
// so there's no sampling involved in this function. 
// All of the sampling happens in sampleGam(). 
// saveGam is (J, K-1, R)
// logW is (n, K)
void PMC::getLogWs(mat& logW) {
  // Rcout << "start of PMC::getLogWs()" << endl;
  if (K==1) { 
    logW.col(0).ones();
    return;
  }  
  logW.zeros();
  
  // Compute eta. 
  mat eta(n, K-1);
  for (size_t j=0; j<J; j++) {
    uvec C_j = arma::find(C==j);
    mat jGam = saveGam.row(j);  // (K-1, R)
    eta.rows(C_j) = psiX.rows(C_j) * jGam.t();  // (n_j, K-1)
  }

  // Compute log weights. 
  // Could probably do this more efficiently. Sth like:
  // vec signs(K-1);  // elements are -1, 0, or 1
  // set signs according to k;
  // logW.col(k) -= sum(log(1 + exp(signs*eta.col(l))));
  for (size_t k=0; k<K; k++) {
    size_t l=k+K;  // root=1 index scheme
    while (l > 1) {
      int s = pow(-1, l%2 + 1);  // sign: want odd l to be 1; even l to be -1
      l /= 2;
      logW.col(k) -= log(1 + exp( s*eta.col(l-1) ));
    }
  }

}


// Compute logs. Reuse logW to sidestep create/copy of a second (n,K) matrix. 
// for (int k=(K-2); k>=0; k--) 
//   logW.col(k) = logW.col(k) - sum(log(1 + exp(logW.cols(0, k))), 1); 

// Reuse logW to sidestep creation (and copying) of a second (n,K) matrix. 
// Use k-1 instead of k to avoid size_t wraparound: 0-> 18446744073709551615.
// vec v = ones<vec>(n);
// for (size_t k=(K-1); k>0; k--)  
//   for (size_t l=0; l<=(k-1); l++) 
//     logW.col(k-1) = logW.col(k-1) - log(1 + exp(logW.col(l)));  
// 
// 
// vec logsum = sum(log(1 + exp(logW.cols(0, k))), 1);

// arma::mat PMC::sampleLogWs(const arma::umat& N, double a0) {
//   
//   mat logW(J,K);
//   for(int j=0; j<J; j++) {
//     // logW.row(j) = rDirichlet( N.row(j).t() +  a0 * ones<vec>(K) / K ).t();
//     logW.row(j) = rDirichlet( zeta * conv_to<vec>::from(N.row(j).t()) +  
//       a0 * ones<vec>(K) / K ).t();
//   }
//   
//   return logW;
// }

// arma::mat PMC::sampleLogWsCov(const arma::umat& N, double a0) {
//   Rcout << "start of PMC::sampleLogWsCov()" << endl;
//     
//     // Polya-gamma data augmentation per 2021 Rigon Durante
//   for (size_t k=0; k<(K-1); k++) {
//     
//     uvec ij = arma::find(T>(k-1));
//     uvec Tij = T(ij);
//     mat psiXij = psiX.rows(ij);  // mat psiXij(n, R, fill::zeros);
//     vec kappa(ij.size(), fill::zeros);
//     kappa(arma::find(Tij==k)).ones();
//     for (auto& ka : kappa)
//       ka -= 0.5;
//     
//     for(size_t j=0; j<J; j++) {
//       
//       // 1. Update PG data. 
//       vec gamtmp = saveGam.tube(j, k);  
//       vec pgdat = psiXij * gamtmp; // Compiler error if I don't create gamtmp.
//       for (auto& c : pgdat)   // Can re-use memory this way. 
//         c = rPG(1, c);  // XX: sample from PG(1, b)
//       
//       // 2. Update gamma. 
//       // Rcout << "** diag_pg" << endl;
//       mat gam_Sigjk = arma::inv(psiXij.t() * diagmat(pgdat) * psiXij + gam_Sig_inv);
//       vec gam_mujk = gam_Sigjk * (psiXij.t() * kappa + gam_Sig_inv * gam_mu);
//       
//       
//       Rcout << "** before saveGam" << endl;
//       // saveGam.tube(j, k) = gamjk;
//       // vec gamk = mvrnormArma(1, gam_mujk, gam_Sigjk).t();
//       // saveGam.col(k) = mvrnormArma(1, gam_mujk, gam_Sigjk).t();
//       saveGam.tube(j, k) = mvrnormArma(1, gam_mujk, gam_Sigjk).t();
//       Rcout << "** after saveGam" << endl;
//       
//       // 3. Set weight omegajk as function of gamk. 
//       // double etajk = psiX * gamk; // XX: not correct
//       // logW(j,k) = gamjk;
//       
//     }
//     
//   }
//   
//   return PMC::sampleLogWs(N, a0);
// }

// arma::mat PMC::getWs() {
//   // Returns weights, which is a deterministic function of psiX and gam, 
//   // so there's no sampling involved in this function. 
//   // All of the sampling happens in gam. 
//   Rcout << "start of PMC::getWs()" << endl;
//   
//   // Rcout << "start of eta" << endl;
//   mat W(n, K);
//   for (size_t j=0; j<J; j++) {
//     uvec jrows = arma::find(C==j);
//     mat jGam = saveGam.row(j);
//     W.rows(jrows) = psiX.rows(jrows) * jGam.t();  // eta
//   }
//   
//   // What's the most computationally-stable way to compute logistic function?
//   // Rcout << "start of logit" << endl;
//   // W = exp(W / (ones<mat>(n, K) + W));  // logit^{-1}(eta)
//   W = ones<mat>(n, K) / (ones<mat>(n, K) + exp(-W));  // logit^{-1}(eta)
//   
//   // Break sticks. Sidesteps creation (and copying) of a second (n,K) matrix. 
//   // Rcout << "start of stick-breaking" << endl;
//   vec v = ones<vec>(n);
//   for (size_t k=(K-1); k>0; k--)  
//     for (size_t l=0; l<k; l++) 
//       W.col(k) = W.col(k) % (v - W.col(l));  // Can do notation shorthand? 
//   
//   return(W);
// }



    
Rcpp::List PMC::initialParticles() {
      // Rcpp::List PMC::initialParticles( arma::uvec T ) {
  // Given the arguments, initialPoints creates a population 
  // of particles from simple proposals: sample z iid N(0,1).
  // If N_k>0:
  // psi_k as mean of the normal part of the full conditional (with xi0k=mean(Y_k)),
  // xi_jk, xi0k as mean of full conditional [xi0k|psi_k, Sk=0, ...],
  // G as mean of IW part of full conditional (taking Lambda=0pxp, Sk)
  // If N_k==0:
  // psi as mean of the normal part of the full conditional 
  // (with all data and xi0k=mean(Y)),
  // xi0k as mean of full conditional with full data, latter psi
  // half of xi_j as above xi0k, half as xi0k + noise
  // G as mean of IW part of full conditional (taking Lambda=0pxp, S)
  //
  // E_k from E_k|S_k=0,...  which is the prior for E_k
  // S_k and varphi for now from their priors
  // *** consider sampling S_k from prior and then the rest accordingly (instead of all from S_k=0)
  Rcout << "start of PMC::initialParticles()" << endl;
  
  Rcpp::List all_particles(K);
  
  for (int k=0; k<K; k++) {
    uvec T_k = arma::find(T==k);
    int n_k = T_k.n_elem;

    mat z_k(num_particles, n_k, fill::zeros);
    mat psi_k(num_particles,p, fill::zeros);
    cube xi_k(J, p, num_particles);
    mat xi_0k(num_particles, p);
    mat G_k(num_particles, pow(p,2));
    mat E_k(num_particles, pow(p,2));

    if( n_k > p + 2 ) {  // XX: lots of duplicate code. condense? 
    
      // This is clumsy, but apparently required for reproducibility,
      // when initializing with randn<mat>(num_particles, n_k) results 
      // are not consistent across platforms. 
      for (size_t row=0; row<num_particles; row++) 
        for (size_t col=0; col<n_k; col++) 
          z_k(row,col) = randn();
 
      mat Y_k = Y.rows(T_k);
      rowvec mean_y = mean(Y_k, 0);
      mat absz = abs(z_k);
      vec mean_absz = mean(absz, 1);
      
      mat zmat = absz.each_col() - mean_absz;
      mat ymat = Y_k.each_row() - mean_y;
      vec devz = sum(pow(zmat,2), 1);
      for (size_t iN=0; iN<num_particles; iN++) 
        for (size_t ip=0; ip<p; ip++) 
          psi_k(iN, ip) = as_scalar(accu(zmat.row(iN).t() % ymat.col(ip)) / devz(iN));  
      
      xi_0k = -1.0*(psi_k % repmat(mean_absz, 1, p));   
      xi_0k.each_row() += mean_y;
      
      uvec C_k = C(T_k);
      for (int j=0; j<J; j++) {
        uvec j_indices = find(C_k==j);
        int n_jk = j_indices.n_elem;
        if (n_jk>0) {
          mat z_jk = z_k.cols(j_indices);
          mat Y_jk = Y_k.rows(j_indices);
          rowvec mean_yjk = mean(Y_jk, 0);
          mat absz_jk = abs(z_jk);
          vec mean_absz_jk = mean(absz_jk, 1);
          
          mat xi_jk = -1.0*(psi_k % repmat(mean_absz_jk, 1, p));  
          xi_jk.each_row() += mean_yjk;
          xi_k.subcube(j,0,0,j,p-1,num_particles-1) = xi_jk.t();
        } else {
          xi_k.subcube(j,0,0,j,p-1,num_particles-1) = xi_0k.t();
        }
      }
      
      for (size_t iN=0; iN<num_particles; iN++) {
        mat e = ( Y_k.each_row() - xi_0k.row(iN) );
        e -= ( repmat(psi_k.row(iN), n_k, 1) % repmat(abs(z_k.row(iN).t()), 1, p) );
        mat Gsum(p, p, fill::zeros);
        for (size_t i=0; i<n_k; i++) { Gsum += e.row(i).t() * e.row(i); }  // XX: can do some more efficient matrix multiplication, I think
        G_k.row(iN) = vectorise(Gsum / n_k).t();
      }
    } 
    else { // if N_k==0, do same with all data, but return empty z
      int n_all = Y.n_rows;
      
      // This is clumsy, but apparently required for reproducibility,
      // when initializing with randn<mat>(num_particles, n_k) results 
      // are not consistent across platforms.
      mat temp_z_k( num_particles, n_all );
      for (size_t row=0; row<num_particles; row++) 
        for (size_t col=0; col<n_k; col++) 
          temp_z_k(row,col) = randn();
      
      rowvec mean_y = mean(Y, 0);
      mat absz = abs(temp_z_k);
      vec mean_absz = mean(absz, 1);
      
      mat zmat = absz.each_col() - mean_absz;
      mat ymat = Y.each_row() - mean_y;
      vec devz = sum(pow(zmat,2), 1);
      
      for (size_t iN=0; iN<num_particles; iN++) 
        for (size_t ip=0; ip<p; ip++) 
          psi_k(iN, ip) = as_scalar(accu(zmat.row(iN).t() % ymat.col(ip)) / devz(iN));  
      
      xi_0k = -1.0*(psi_k % repmat(mean_absz, 1, p));
      xi_0k.each_row() += mean_y;
      
      for (size_t j=0; j<J; j++) 
          xi_k.subcube(j,0,0,j,p-1,num_particles-1) = trans( xi_0k + mvrnormArma(num_particles, mean(xi_0k, 0).t(), E0/(e0 - p - 1.0)) );
      
      for (size_t iN=0; iN<num_particles; iN++) {
        mat e = ( Y.each_row() - xi_0k.row(iN) );
        e -= ( repmat(psi_k.row(iN), n_all, 1) % repmat(abs(temp_z_k.row(iN).t()), 1, p) );  
        mat Gsum(p, p, fill::zeros);
        for (size_t i=0; i<n_all; i++ ) { Gsum += e.row(i).t() * e.row(i); }
        G_k.row(iN) = vectorise(Gsum / n_all).t();
      }
      z_k.set_size( num_particles, 0 );
    } // end N_k==0
    
    for (size_t iN=0; iN<num_particles; iN++) {
      mat E = inv_sympd(rWishartArma(inv_sympd(E0), e0 ));  // XX: is this correct? there's no dependence on iN
      E_k.row(iN) = vectorise(E).t();
    }
    
    Rcpp::List particles = Rcpp::List::create(
      Rcpp::Named( "z" ) = z_k,
      Rcpp::Named( "psi" ) = psi_k,
      Rcpp::Named( "xi" ) = xi_k,
      Rcpp::Named( "xi0" ) = xi_0k,
      Rcpp::Named( "G" ) = G_k,
      Rcpp::Named( "E" ) = E_k//,
      // Rcpp::Named( "S" ) = S_k
      );
    
    all_particles(k) = particles;
  } // end k for loop
  return all_particles;
}


Rcpp::List PMC::sampleXi(const mat& Y_k, const uvec& C_k, uvec N_k, 
                         Rcpp::List particles) {
  // Given the arguments, this function returns a population of MC draws 
  // for the values of the variable Xi, in the p-variate skew-N model.
  int nk = Y_k.n_rows;

  // uvec S = Rcpp::as<umat>(particles["S"]);
  mat z = Rcpp::as<mat>(particles["z"]);
  mat psi = Rcpp::as<mat>(particles["psi"]);
  mat G = Rcpp::as<mat>(particles["G"]);
  mat E = Rcpp::as<mat>(particles["E"]);
  mat xi0 = Rcpp::as<mat>(particles["xi0"]);
  
  cube xi(J, p, num_particles);
  xi.fill(0);
  mat log_dxi(num_particles, J);
  
  if(nk == 0) {
    // Rcout << "empty xi" << endl;
    for ( int iN=0; iN<num_particles; iN++ ) {
        mat tempE = reshape(E.row(iN), p, p);
        double sgn, ld;
        log_det( ld, sgn, tempE );
        vec eigval; mat eigvec;
        eig_sym(eigval , eigvec,  tempE) ;
        mat invSqrtL_invU = diagmat(1.0/sqrt(eigval)) * inv(eigvec);
        for(int j=0; j<J; j++) {
          // xi.slice(iN).row(j) = trans( mvnrnd(xi0.row(iN).t(), tempE) );
          xi.slice(iN).row(j) = mvrnormArma(1, xi0.row(iN).t(), tempE);
            
          // Rcout << "empty xi "<< xi.slice(iN).row(j) << endl;
          vec normedXi = invSqrtL_invU * trans(xi.slice(iN).row(j) - xi0.row(iN));
          log_dxi(iN, j) = as_scalar(-log2pi * (p/2.0) -0.5*normedXi.t()*normedXi -0.5*sgn*ld );
          // log_dxi(iN, j) = as_scalar(dmvnrm_arma_precision(
          //                  xi.slice(iN).row(j), xi0.row(iN),
          //                  inv_sympd(tempE) ) );
        } // end j loop
    } // end iN loop
    // Rcout << "end empty xi" << endl;
  } else { // if nk>0:
    for ( int iN=0; iN<num_particles; iN++ ) {
        xi.slice(iN) = repmat( xi0.row(iN), J, 1 );
        log_dxi.row(iN).fill(0);
        mat invE = inv_sympd(reshape(E.row(iN),p,p));
        mat invG = inv_sympd(reshape(G.row(iN),p,p));
        for(int j=0; j<J; j++) {
          int njk = N_k(j);
          if ( njk ==0 ) {
            // xi.slice(iN).row(j) = xi0.row(iN);
            xi.slice(iN).row(j) = trans( mvnrnd(xi0.row(iN).t(), reshape(E.row(iN),p,p)) );
            // log_dxi(iN, j) = 0;
            log_dxi(iN, j) = as_scalar(dmvnrm_arma_precision(
                                              xi.slice(iN).row(j),
                                              xi0.row(iN),
                                              reshape(E.row(iN),p,p)) );
          } else { // if njk>0
            uvec jk_idx = find(C_k==j);
            mat Yjk = Y_k.rows(jk_idx);
            mat zin = z.row(iN);
            rowvec zjk = zin.cols(jk_idx);
            // mat v = inv_sympd(invE + njk*invG);
            mat v = inv_sympd(invE + zeta*njk*invG);
            rowvec mean_y = mean(Yjk, 0);
            // vec m = invE * xi0.row(iN).t() + njk*invG * trans((mean_y - (psi.row(iN) * mean(abs(zjk)))));
            vec m = invE * xi0.row(iN).t() + zeta*njk*invG * trans((mean_y - (psi.row(iN) * mean(abs(zjk)))));
            xi.slice(iN).row(j) = trans(mvnrnd(v * m, v));
            log_dxi(iN, j) = as_scalar(dmvnrm_arma_precision(
                                  xi.slice(iN).row(j), trans(v * m),
                                  invE + zeta*njk*invG ) );
            // log_dxi(iN, j) = as_scalar(dmvnrm_arma_precision(
            //   xi.slice(iN).row(j), trans(v * m),
            //   invE + njk*invG ) );
          }
        } // end j loop
      // } // end S(iN)==1 if
    } // end iN (particles) loop
  } // end nk>0 loop
  
  vec vec_log_dxi = sum(log_dxi, 1);
  
  return Rcpp::List::create(
    Rcpp::Named( "xi" ) = xi,
    Rcpp::Named( "log_dq" ) = vec_log_dxi
  );
}

Rcpp::List PMC::sampleG(const  mat& Y_k, const uvec& C_k, uvec N_k,
                        Rcpp::List particles, Rcpp::List prior ) {
  
  int nk = Y_k.n_rows;
  
  // uvec S = Rcpp::as<umat>(particles["S"]);
  cube xi = Rcpp::as<cube>(particles["xi"]);
  mat xi0 = Rcpp::as<mat>(particles["xi0"]);
  mat absz = abs(Rcpp::as<mat>(particles["z"]));
  mat psi = Rcpp::as<mat>(particles["psi"]);
  double m = Rcpp::as<double>(prior["m0"]);
  mat Lambda = Rcpp::as<mat>(prior["Lambda"]);
  
  mat G(num_particles, pow(p,2));
  vec log_dG(num_particles);
  int iN;
  mat ek(nk,p);
  mat zp(nk,p);
  mat Lambda_k(p,p);
  mat g(p,p);
  
  if (nk==0) {
    // Rcout << "empty G" << endl;
    for ( iN=0; iN<num_particles; iN++) {
      g = inv_sympd(rWishartArma(inv_sympd(Lambda), m ));
      G.row(iN) = vectorise(g).t();
      log_dG(iN) = dIWishartArma(g, m, Lambda);
    }
    // Rcout << "end empty G" << endl;
  } else { // if nk>0
    for ( iN=0; iN<num_particles; iN++ ) {
      mat Sk(p,p);
      Sk.fill(0);
        for (int j=0; j<J; j++) {
          int njk = N_k(j);
          if ( njk > 0 ) {
            uvec jk_idx = find(C_k==j);
            mat ejk(njk, p);
            mat zpj(njk, p);
            rowvec xi_j = xi.subcube(j, 0, iN, j, p-1, iN);
            mat Y_jk = Y_k.rows(jk_idx);
            ejk =  Y_jk.each_row() - xi_j;
            vec zrow = trans( absz.row(iN) );
            zpj = repmat( zrow(jk_idx) , 1 ,p);
            zpj.each_row() %= psi.row(iN);
            ejk -= zpj;
            Sk += ejk.t() * ejk;
          }
        // }
      }
      if ( accu(abs(Lambda))==0 )
        Lambda_k = zeta * Sk;
        // Lambda_k = Sk;
      else
        Lambda_k = inv_sympd(Lambda) + zeta * Sk;
        // Lambda_k = inv_sympd(Lambda) + Sk;
      
      g = inv_sympd(rWishartArma(inv_sympd(Lambda_k), ceil(zeta * nk) + m ));
      G.row(iN) = vectorise(g).t();
      log_dG(iN) = dIWishartArma(g, nk+m, Lambda_k);
      // log_dG(iN) = dIWishartArma(g, zeta*nk+m, Lambda_k);
    }
  }
  
  return Rcpp::List::create(  
    Rcpp::Named( "G" ) = G,
    Rcpp::Named( "log_dq" ) = log_dG  );
}


Rcpp::List PMC::samplePsi(const  mat& Y_k, const uvec& C_k, uvec N_k, 
                          Rcpp::List particles, Rcpp::List prior) {
  
  int nk = Y_k.n_rows;
  int p = Y_k.n_cols;
  
  // uvec S = Rcpp::as<umat>(particles["S"]);
  cube xi = Rcpp::as<cube>(particles["xi"]);
  mat G = Rcpp::as<mat>(particles["G"]);
  mat xi0 = Rcpp::as<mat>(particles["xi0"]);
  mat absz = abs(Rcpp::as<mat>(particles["z"]));
  vec sums_z2 = sum(pow(absz,2), 1);
  
  mat psi(num_particles, p);
  vec log_dpsi(num_particles);
  
  if(nk == 0) {
    // Rcout << "empty psi" << endl;
    // add sampling from prior on n-ball
    for ( int iN=0; iN<num_particles; iN++ ) {
      vec x(p);
      for (int pp=0; pp<p; pp++) {
        x(pp) = randn();
      }
      
      double r = norm(x);
      double u = randu();
      // sample from Sigma prior (take psi*psi element here to be zero)
      double m = Rcpp::as<double>(prior["m0"]);
      mat Lambda = Rcpp::as<mat>(prior["Lambda"]);
      mat g = inv_sympd(rWishartArma(inv_sympd(Lambda), m ));
      rowvec h = sqrt(g.diag().t());
      mat invD = inv(diagmat(h));
      double detOmega =  det(invD * g * invD);
    
      vec delta = pow(u, 1/p) * (1 / r) * pow(detOmega, p/2) * x;
      psi.row(iN) = trans( diagmat(h) * delta );
      double logSphere = (p/2.0) * log(M_PI) - lgamma(p/2.0+1.0);
      log_dpsi(iN) = 1 / (logSphere * log(detOmega));
    }
    // Rcout << "end empty psi" << endl;
  } else { // if nk>0:
    for ( int iN=0; iN<num_particles; iN++ ) {
      rowvec mpsi(p);
      mpsi.fill(0);
        for (int j=0; j<J; j++) {
          int njk = N_k(j);
          if ( njk > 0 ) {
            uvec jk_idx = find(C_k==j);
            rowvec xi_j = xi.subcube(j, 0, iN, j, p-1, iN);
            mat Y_jk = Y_k.rows(jk_idx);
            mat Yjk_dmean = Y_jk.each_row() - xi_j;
            vec zrow = trans( absz.row(iN) );
            Yjk_dmean.each_col() %= zrow(jk_idx);
            mpsi += sum(Yjk_dmean, 0);
          } 
        } // end j loop
        mpsi /= sums_z2(iN);
      // }
      mat vpsi = reshape(G.row(iN) / (zeta*sums_z2(iN)), p, p);
      // mat vpsi = reshape(G.row(iN) / (sums_z2(iN)), p, p);
      psi.row(iN) = (arma::mvnrnd(mpsi.t(), vpsi)).t();
      log_dpsi(iN) = as_scalar(
        dmvnrm_arma_precision(  psi.row(iN),
                                mpsi,
                                inv_sympd(vpsi) ));
    } // end iN (particles) loop
  } // end nk>0 if
  
  return Rcpp::List::create(  
    Rcpp::Named( "psi" ) = psi,
    Rcpp::Named( "log_dq" ) = log_dpsi
  );
  
}

Rcpp::List PMC::sampleZ(const mat& Y_k, const uvec& C_k, Rcpp::List particles ) {
  
  int nk = Y_k.n_rows;
  vec log_dZ(num_particles);
  mat z(num_particles, nk);
  
  if ( nk==0 ) {
    // Rcout << "empty z" << endl;
    log_dZ.fill(0); 
    // Rcout << "end empty z" << endl;
  } else {
    cube xi = Rcpp::as<cube>(particles["xi"]);
    mat xi0 = Rcpp::as<mat>(particles["xi0"]);
    mat psi = Rcpp::as<mat>(particles["psi"]);
    mat G = Rcpp::as<mat>(particles["G"]);
    int i;
    double m;
    mat xi_j;
    double sgn, absz;
    vec tp(nk); vec ldZ(nk); vec u(nk);
    
    for (int ip=0; ip<num_particles; ip++) {
      mat invG = inv_sympd(reshape(G.row(ip), p, p));
      // double v = 1.0/(1.0 + as_scalar(psi.row(ip) * invG * psi.row(ip).t()));
      double v = 1.0/(1.0 + zeta*as_scalar(psi.row(ip) * invG * psi.row(ip).t()));
      double sv = sqrt(v);
      u = randu(nk);
      // Rcout << "randu(nk)" << u(0) << endl;
      for ( i=0; i<nk; i++ ) {
          xi_j = xi.subcube(C_k(i), 0, ip, C_k(i), p-1, ip);
          m = as_scalar(v * zeta * (psi.row(ip) * invG * (Y_k.row(i) - xi_j).t()));
        if (u(i) < 0.5) {
          sgn = -1.0;
        } else {
          sgn = 1.0;
        }
        absz = rtruncnormArma(m, sv, 0);
        z(ip, i) = sgn * absz;
        tp(i) = R::pnorm(0, -m, sv, true, true);
        ldZ(i) = R::dnorm(absz, m, sv, true);
      }
      log_dZ(ip) = accu( ldZ - tp - log(2) );
    }
  }
  
  return Rcpp::List::create(
    Rcpp::Named( "z" ) = z,
    Rcpp::Named( "log_dq" ) = log_dZ
  );
}


Rcpp::List PMC::sampleXi0(const mat& Y_k, uvec N_k, Rcpp::List particles) {
  int nk = Y_k.n_rows;

  mat psi = Rcpp::as<mat>(particles["psi"]);
  mat G = Rcpp::as<mat>(particles["G"]);
  mat E = Rcpp::as<mat>(particles["E"]);
  cube xi = Rcpp::as<cube>(particles["xi"]);
  mat absz = abs(Rcpp::as<mat>(particles["z"]));

  mat xi0(num_particles, p);
  xi0.fill(0);
  vec log_dxi0(num_particles);
  
  if(nk == 0) {
    xi0 = mvrnormArma(num_particles, b0, B0);
    log_dxi0 =  dmvnrm_arma_precision(xi0,
                                      b0.t(),
                                      invB0 );
  } else { // if nk>0:
    for ( int iN=0; iN<num_particles; iN++ ) {
      vec mxi0(p);
      mat vxi0(p,p);
      mat inv_vxi0(p,p);
      mat invG = inv_sympd(reshape(G.row(iN), p, p));
        uvec jk_idx = find(N_k>0);
        mat invE = inv_sympd(reshape(E.row(iN), p, p));
        inv_vxi0 = invB0 + zeta*jk_idx.n_elem*invE;
        vxi0 = inv_sympd(inv_vxi0);
        mat xi_slice = xi.slice(iN);
        vec mean_xi = trans( mean(xi_slice.rows(jk_idx), 0) );
        mxi0 = vxi0 * (invB0*b0 + zeta * jk_idx.n_elem * invE * mean_xi);
      // } // end S(iN)==1 if
      xi0.row(iN) = mvrnormArma(1, mxi0, vxi0);
      log_dxi0(iN) =  as_scalar(dmvnrm_arma_precision(
                                  xi0.row(iN), 
                                  mxi0.t(),
                                  inv_vxi0 ) );
    } // end iN (particles) loop
  } // end nk>0 loop
  
  return Rcpp::List::create(
    Rcpp::Named( "xi0" ) = xi0,
    Rcpp::Named( "log_dq" ) = log_dxi0
  );
}

Rcpp::List PMC::sampleE(uvec N_k, Rcpp::List particles, Rcpp::List prior) {
  
  // Rcout << "e0=" << e0 << endl;
  int nk = accu(N_k);
  
  // uvec S = Rcpp::as<umat>(particles["S"]);
  cube xi = Rcpp::as<cube>(particles["xi"]);
  mat xi0 = Rcpp::as<mat>(particles["xi0"]);
  // double e0 = Rcpp::as<double>(prior["e0"]);
  // mat E0 = Rcpp::as<mat>(prior["E0"]);
  
  mat E(num_particles, pow(p,2));
  vec log_dE(num_particles);
  int iN;
  
  if (nk==0) {
    // Rcout << "empty E" << endl;
    for ( iN=0; iN<num_particles; iN++) {
      // mat e = inv_sympd(rWishartArma( inv_sympd(E0), e0 ));
      mat e = inv_sympd(rWishartArma( invE0, e0 ));
      E.row(iN) = vectorise(e).t();
      log_dE(iN) = dIWishartArma(e, e0, E0);
    }
    // Rcout << "end empty E" << endl;
  } else { // if nk>0
    for ( iN=0; iN<num_particles; iN++ ) {
      mat Sk(p,p);
      Sk.fill(0);
        uvec jk_idx = find(N_k>0);
        double ek = e0 + jk_idx.n_elem;
        mat xi_slice = xi.slice(iN).rows(jk_idx);
        mat xi_dmean = xi_slice.each_row() - xi0.row(iN);
        mat mE = ( invE0 + xi_dmean.t() * xi_dmean );
        mat e = inv( rWishartArma( mE, ek ) );
        E.row(iN) = vectorise(e).t();
        log_dE(iN) = dIWishartArma(e, ek, inv_sympd(mE));
      // }
    }
  }
    
  return Rcpp::List::create(  
    Rcpp::Named( "E" ) = E,
    Rcpp::Named( "log_dq" ) = log_dE  );
}

double PMC::sampleA0(double a0, const arma::umat& N, double a_par)
{
  double output = a0;    
  double log_ratio = 0;
  double temp = rgammaBayes(  pow( a0, 2 ) * a_par, 
                              a0 * a_par );
  // Rcout << "rgammabayes" << temp << endl;
  
  log_ratio += R::dgamma(a0, pow(temp,2)* a_par, 1/temp/a_par, 1);
  log_ratio -= R::dgamma(temp, pow(a0,2)* a_par, 1/a0/a_par, 1);
  log_ratio += R::dgamma(temp, tau_a(0), 1/tau_a(1), 1);
  log_ratio -= R::dgamma(a0, tau_a(0), 1/tau_a(1), 1);
  
  for(int j = 0; j < J; j++)  {
    log_ratio += marginalLikeDirichlet( N.row(j).t(), (temp/K)*ones<vec>(K)  );
    log_ratio -= marginalLikeDirichlet( N.row(j).t(), (a0/K)*ones<vec>(K)  );
  }
  if( exp(log_ratio) > randu() )
    output = temp;
  
  return output;
}

// ----------------------------------------------------------------------------------

arma::vec PMC::logPriorDens( Rcpp::List particles,
                             Rcpp::List prior ) {

  // uvec S = Rcpp::as<uvec>(particles["S"]);
  cube xi = Rcpp::as<cube>(particles["xi"]);
  mat xi0 = Rcpp::as<mat>(particles["xi0"]);
  mat psi = Rcpp::as<mat>(particles["psi"]);
  mat G = Rcpp::as<mat>(particles["G"]);
  mat E = Rcpp::as<mat>(particles["E"]);
  
  // a_varphi = Rcpp::as<double>(prior["a_varphi"]);
  // b_varphi = Rcpp::as<double>(prior["b_varphi"]);
  double m = Rcpp::as<double>(prior["m0"]);
  mat Lambda = Rcpp::as<mat>(prior["Lambda"]);
  
  mat h(num_particles, p);
  vec logDetOmega(num_particles);
  vec logDetSigma(num_particles);
  vec logPriorG(num_particles);
  vec logPriorE(num_particles);
  // vec logPriorS = log(1 - varphi);
  // logPriorS( find(S==1) ) = log( varphi( find(S==1) ) );
  // vec logPriorVarphi(num_particles);
  
  int iN;
  mat Sigma(p,p);
  mat invD(p,p);
  double logpxi = 0;

// #pragma omp parallel for private(ip, Sigma, invD)
  for ( iN=0; iN<num_particles; iN++ ) {
    Sigma = reshape(G.row(iN), p, p) + psi.row(iN).t() * psi.row(iN);
    // if(!all(eigen(Sigma.iN)$values>0)) out[iN] = 1
    logDetSigma(iN) = log(det(Sigma));
    h.row(iN) = sqrt(Sigma.diag().t());
    mat invD = inv(diagmat(h.row(iN)));
    logDetOmega(iN) =  log(det(invD * Sigma * invD));
    
    mat tempE = reshape(E.row(iN), p, p);
    // temppriore(iN) = log(det(tempE));
    // if (S(iN)==1) {
      for (int j=0; j<J; j++) {
        logpxi += as_scalar( dmvnrm_arma_precision(
                           xi.slice(iN).row(j), xi0.row(iN),
                           inv_sympd(tempE) ));
      }
    // } else {
    //   logpxi = 0;
    // }
    
    logPriorE(iN) = dIWishartArma(tempE, e0, E0);
    // logPriorVarphi(iN) = dBeta( varphi(iN), a_varphi, b_varphi );
    if (m==0) {
      logPriorG(iN) = -((p+1.0)/2.0) * logDetSigma(iN);
    } else {
      logPriorG(iN) = dIWishartArma( Sigma, m, Lambda);
    }
  }
  // pos = 1 - out

  double logSphere = (p/2.0) * log(M_PI) - lgamma(p/2.0+1.0);

  vec logPriorDens = logpxi - logSphere - 0.5 * logDetOmega +
                        logPriorG + logPriorE -
                        sum(log(h), 1);

  // logPriorG + logPriorE + logPriorS + logPriorVarphi -
  
  //  logPriorDens[which(particlesStar$pos == 0)] = rep(-Inf, sum(particlesStar$pos == 0))
  return(logPriorDens);
}

arma::vec PMC::logPostDens( const mat& Y_k, const uvec& C_k, uvec N_k,
                            Rcpp::List particles,
                            Rcpp::List prior ) {

  int nk = Y_k.n_rows;
  vec loglikelihood(num_particles);
  loglikelihood.fill(0);

  if (nk>0) {
      // uvec S = Rcpp::as<umat>(particles["S"]);
      cube xi = Rcpp::as<cube>(particles["xi"]);
      mat xi0 = Rcpp::as<mat>(particles["xi0"]);
      mat psi = Rcpp::as<mat>(particles["psi"]);
      mat G = Rcpp::as<mat>(particles["G"]);
      mat absz = abs(Rcpp::as<mat>(particles["z"]));
    
      vec sums_z2(num_particles);
      int iN;
      // mat e(nk,p);
      mat zp;
      mat g(p,p);
      mat ee(num_particles, pow(p,2));
      vec detG(num_particles);
      vec PsiVec(num_particles);
    
    // #pragma omp parallel for private(ip, e, zp, Lambda, g)
      for (iN=0; iN<num_particles; iN++) {
          for (int j=0; j<J; j++) {
            int njk = N_k(j);
            if ( njk > 0 ) {
              uvec jk_idx = find(C_k==j);
              mat absz_jk = absz.cols(jk_idx);
              sums_z2 = arma::sum(absz_jk % absz_jk, 1);
              mat Y_jk = Y_k.rows(jk_idx);
              rowvec xi_j = xi.subcube(j, 0, iN, j, p-1, iN);
              mat e = Y_jk.each_row() - xi_j;
              zp = repmat(absz_jk.row(iN).t(), 1, p);
              zp.each_row() %= psi.row(iN);
              e -= zp;
              ee = e.t() * e;
              g = reshape(G.row(iN), p, p);
              detG(iN) = det(g);
              PsiVec(iN) = accu(inv(g) % ee);
              loglikelihood(iN) += as_scalar((- njk/2.0) * log(detG(iN)) - 0.5 * PsiVec(iN) + (- 0.5) * sums_z2.row(iN));
            }
          }
        // }
      }
      // loglikelihood
      double loglik_normalizing_const = - nk * (p+1.0)/2.0 * log(2.0*M_PI);
      loglikelihood += loglik_normalizing_const;
  } // end nk>0 
  
  
  vec log_prior = logPriorDens( particles, //varphi, 
                                prior );
  // Numerator of the unnormalized importance weights
  vec log_pi = log_prior + zeta * loglikelihood;
  return(log_pi);
}

Rcpp::List PMC::iter(const uvec& T,
                     int k,
                     const umat& N,
                     Rcpp::List particles,
                     mat log_dQ,
                     // vec varphi,
                     Rcpp::List prior ) {
  
  uvec T_k = arma::find(T==k);
  mat Y_k = Y.rows(T_k);
  uvec C_k = C(T_k);
  
  int nk = Y_k.n_rows;

  // Proposal step (with permuted sweeps):

  List drawList;
  // Rcout << "start z" << endl;
  drawList = sampleZ(Y_k, C_k, particles);
  particles["z"] = Rcpp::as<mat>(drawList["z"]);
  log_dQ.col(1) = Rcpp::as<vec>(drawList["log_dq"]);
  // Rcout << "end z" << endl;

  uvec parPerm = randsamp(5,3,7);
  // Rcout << "randsamp" << parPerm << endl;

  for(int ipar=0; ipar<5; ipar++) {
    switch ( parPerm(ipar) ) {
    case 3:
      // Rcout << "start xi" << endl;
      drawList = sampleXi( Y_k, C_k, N.col(k), particles);
      particles["xi"] = Rcpp::as<cube>(drawList["xi"]);
      log_dQ.col(3) = Rcpp::as<vec>(drawList["log_dq"]);
      // Rcout << "end xi" << endl;
      break;
    case 4:
      // Rcout << "start G" << endl;
      drawList = sampleG(Y_k, C_k, N.col(k), particles, prior);
      particles["G"] = Rcpp::as<mat>(drawList["G"]);
      log_dQ.col(4) = Rcpp::as<vec>(drawList["log_dq"]);
      // Rcout << "end G" << endl;
      break;
    case 5:
      // Rcout << "start psi" << endl;
      drawList = samplePsi( Y_k, C_k, N.col(k), particles, prior );
      particles["psi"] = Rcpp::as<mat>(drawList["psi"]);
      log_dQ.col(5) = Rcpp::as<vec>(drawList["log_dq"]);
      // Rcout << "end psi" << endl;
      break;
    case 6:
      // Rcout << "start xi0" << endl;
      drawList = sampleXi0( Y_k, N.col(k), particles);
      particles["xi0"] = Rcpp::as<mat>(drawList["xi0"]);
      log_dQ.col(6) = Rcpp::as<vec>(drawList["log_dq"]);
      // Rcout << "end xi0" << endl;
      break;
    case 7:
      // Rcout << "start E" << endl;
      drawList = sampleE( N.col(k), particles, prior);
      particles["E"] = Rcpp::as<mat>(drawList["E"]);
      log_dQ.col(7) = Rcpp::as<vec>(drawList["log_dq"]);
      // Rcout << "end E" << endl;
      break;
    }
  }
  
  vec iw;
  double log_py, perplexity;
  vec log_pitilde;
  if (nk>0) {
    log_pitilde = logPostDens(Y_k, C_k, N.col(k), particles, //varphi, 
                              prior);
  } else {
    log_pitilde = logPriorDens( particles, //varphi, 
                                prior );
  }
  vec log_q = sum(log_dQ, 1);
  // Rcout << "q " << log_q.t() << endl;
  vec log_iw = log_pitilde - log_q;
  // Rcout << "log iw raw " << log_iw.t() << endl;
  double cnst = max(log_iw);	// Constant needed for the computation of the marginal likelihood
  log_py = cnst + log(accu(exp(log_iw-cnst))) - log(num_particles);
  vec iw_bar = exp(log_iw);
  vec log_iw_b = log_iw - log(num_particles) - log_py;
  vec iw_b = exp(log_iw_b);  // unnormalized weights
  // Rcout << "log iw unnormalized " << iw_b.t() << endl;
  // Rcout << "iw unnormalized " << exp(iw_b.t()) << endl;
  iw = iw_b/accu(iw_b); // normalized weights
  // Rcout << "iw final " << iw.t() << endl;
  perplexity = exp(-accu(iw % log(iw))) / num_particles;

  
  // Rcout << "end copmute weights" << endl;
  
  
  // if (nk==0) Rcout << "start empty sample with replacement " << endl;
  
  uvec resamp(num_particles);
  for (int iN=0; iN<num_particles; iN++) {
    resamp(iN) = sampling(iw);
  }
  
  uvec uresamp = unique(resamp);
  int nResampled = uresamp.n_elem;

  // // Resampling step
  // Rcout << "start resample" << endl;
  cube MMM = Rcpp::as<cube>(particles["xi"]);
  cube MM = MMM;
  for (int i=0; i<num_particles; i++) {
    MM.slice(i) = MMM.slice(resamp(i));
  }
  particles["xi"] = MM;
  mat M;
  M = Rcpp::as<mat>(particles["G"]).rows(resamp);
  particles["G"] = M;
  // Rcout << M << endl;
  M = Rcpp::as<mat>(particles["psi"]).rows(resamp);
  particles["psi"] = M;
  // if (nk==0) Rcout << "start resamp xi0" << endl;
  M = Rcpp::as<mat>(particles["xi0"]).rows(resamp);
  particles["xi0"] = M;
  // if (nk==0) Rcout << "end resamp xi0" << endl;
  M = Rcpp::as<mat>(particles["E"]).rows(resamp);
  particles["E"] = M;

  // Rcout << "end resampling" << endl;
  
  return Rcpp::List::create(
    Rcpp::Named( "particles" ) = particles,
    Rcpp::Named( "log_py" ) = log_py,
    Rcpp::Named( "nResampled" ) = nResampled,
    Rcpp::Named( "perplexity" ) = perplexity  );
}

arma::uvec PMC::sampleT(const arma::cube& xi,
                        const arma::cube& Omega,
                        const arma::mat&  alpha,
                        const arma::mat&  logW)
{
  // Rcout << "start of sampleT()" << endl;
  mat PT(n, K);
  uvec k_idx(1);  // XX: Stupid variable used for indexing PT. 
  for (size_t j=0; j<J; j++) {
    uvec C_j = arma::find(C==j);
    for (size_t k=0; k<K; k++) {  // exp(logW(j,k))
      k_idx(0) = k;
      PT(C_j, k_idx) = dmsnArma(Y.rows(C_j),  // % wvec.col(k);
                                xi.slice(k).row(j),
                                Omega.slice(k), 
                                alpha.col(k)); 
    }
  }
  PT %= exp(logW);
  
  uvec T(n);
  NumericVector U = runif(n);
  for (size_t i=0; i<n; i++) {
    vec prob = PT.row(i).t();
    vec probsum = cumsum(prob);
    double x = U(i) * sum(prob);
    
    size_t k=0;
    while (k<(K-1) && probsum(k)<x) { k++; }
    T(i) = k;
  }

  return(T);
}

// for (k=0; k<K; k++) {
//   Rcout << "start of k-for loop" << endl;
//   uvec k_idx(1);  // replace with PT.submat(C_j, span(k))?
//   k_idx(0) = k;
//   for (j=0; j<J; j++) {
//     Rcout << "start of j-for loop" << endl;
//     C_j = arma::find(C==j);
//     // vec tmpGam = saveGam.tube(j, k);  // conv_to<colvec>::from()
//     // Rcout << "tmpGam.size(): " << tmpGam.size() << endl;
//     // mat tp = psiX.rows(C_j);
//     // Rcout << "tp: " << tp.n_rows << " x " << tp.n_cols << " matrix" << endl;
//     // vec wvec = psiX.rows(C_j) * tmpGam;  // weights
//     // Rcout << "wvec.size(): " << wvec.size() << endl;
//     PT.submat(C_j, k_idx) = dmsnArma(Y.rows(C_j),  // exp(logW(j,k))
//                                      xi.slice(k).row(j),
//                                      Omega.slice(k), // scalar multiplication
//                                      alpha.col(k)) * 0.5;
//   }
// }


//--------------------------------------------------------

Rcpp::List PMC::get_chain()
{
  Rcout << "start of get_chain()" << endl;
  return Rcpp::List::create(
    Rcpp::Named( "t" ) = saveT,
    Rcpp::Named( "z" ) = saveZ,
    Rcpp::Named( "W" ) = saveW,
    Rcpp::Named( "xi" ) = saveXi,
    Rcpp::Named( "xi0" ) = saveXi0,
    Rcpp::Named( "psi" ) = savePsi,
    Rcpp::Named( "G" ) = saveG,
    Rcpp::Named( "E" ) = saveE,
    Rcpp::Named( "a0" ) = saveA0,
    Rcpp::Named( "log_py" ) = saveLog_py,
    Rcpp::Named( "perplexity" ) = savePerplexity,
    Rcpp::Named( "nResampled" ) = saveNResampled,
    Rcpp::Named( "Omega" ) = saveOmega,
    Rcpp::Named( "alpha" ) = saveAlpha,
    Rcpp::Named( "gamma" ) = saveGam
  );
}


