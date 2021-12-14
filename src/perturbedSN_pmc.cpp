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
    K = Rcpp::as<size_t>(prior["K"]);
    R = psiX.n_cols;
    treestr = Rcpp::as<size_t>(prior["treestr"]);
    to_save_W = false;
    use_skew = false;
    
    num_particles = Rcpp::as<size_t>(pmc["npart"]);
    num_iter = Rcpp::as<size_t>(pmc["nskip"]) * Rcpp::as<size_t>(pmc["nsave"]);
    num_burnin = Rcpp::as<size_t>(pmc["nburn"]);
    num_thin = Rcpp::as<size_t>(pmc["nskip"]);
    num_display = Rcpp::as<size_t>(pmc["ndisplay"]);
    
    // seed = Rcpp::as<int>(pmc["seed"]);
    
    
    length_chain = num_iter/num_thin;
    
    saveT.set_size(length_chain, n);
    saveZ.set_size(length_chain, n);
    if (to_save_W) {
        saveW.set_size(n, K, length_chain);  // saveW.set_size(J, K, length_chain);
    }
    saveXi.set_size(J, p*K, length_chain);
    saveXi0.set_size(p, K, length_chain);
    saveE.set_size(p, K*p, length_chain);
    savePsi.set_size(p, K, length_chain);
    saveG.set_size(p, K*p, length_chain);
    saveOmega.set_size(p, K*p, length_chain);
    saveAlpha.set_size(p, K, length_chain);
    saveLog_py.set_size(length_chain, K);
    saveNResampled.set_size(length_chain, K);
    savePerplexity.set_size(length_chain, K);

    T = Rcpp::as<uvec>(state["t"]);

    tau_a = Rcpp::as<vec>(prior["tau_a"]);
    Lamb = Rcpp::as<mat>(prior["Lambda"]);
    is_Lamb_Zero = (accu(abs(Lamb)) == 0);
    chol(cholLamb, Lamb);
    log_det(ldLamb, sgnLamb, Lamb);
    invLamb = arma::inv_sympd(Lamb);

    chol(cholInvLamb, invLamb);
    e0 = Rcpp::as<double>(prior["e0"]);
    E0 = Rcpp::as<mat>(prior["E0"]);
    chol(cholE0, E0);
    log_det(ldE0, sgnE0, E0);
    invE0 = arma::inv_sympd(E0);
    chol(cholInvE0, invE0);
    // invE0 = inv_sympd(E0);
    b0 = Rcpp::as<vec>(prior["b0"]);
    B0 = Rcpp::as<mat>(prior["B0"]);
    chol(cholB0, B0);
    invB0 = inv_sympd(B0);
    merge_step = Rcpp::as<bool>(prior["merge_step"]);
    merge_par = Rcpp::as<double>(prior["merge_par"]);
    zeta = Rcpp::as<double>(prior["zeta"]);
    gam_mu = Rcpp::as<vec>(prior["gam_mu"]);
    gam_Sig = Rcpp::as<mat>(prior["gam_Sig"]);
    gam_Sig_inv = arma::inv(gam_Sig);
    m0 = Rcpp::as<double>(prior["m0"]);
    min_nclu = Rcpp::as<size_t>(prior["min_nclu"]);;
    
    main_loop(initParticles, init);
}

void PMC::main_loop(const Rcpp::List& initParticles, bool init) {
    // Rcout << "start of main_loop()" << endl;

    size_t km = 0;
    
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
    alpha.fill(0);
    rowvec z(n);

    // XX: This is inefficient because of needless copying. 
    size_t Kstar = K-1;  // K-1 for balanced tree; K for unbalanced tree
    saveGam.set_size(J, Kstar, R);
    mat tmp = mvrnormArma(J*Kstar, gam_mu, gam_Sig);  // to perform only one matrix inverse
    for (size_t j=0; j<J; j++)
        for (size_t m=0; m<Kstar; m++)
            saveGam.tube(j, m) = tmp.row(Kstar*j + m);

    // mat log_dQ(num_particles, 8);
    // log_dQ.fill(0);
    vec log_py(K);
    vec perplexity(K);
    uvec nResampled(K);

    Rcpp::List all_particles;
    if (init) {
        Rcout << "initializing all particles..." << endl;
        all_particles = Rcpp::as<Rcpp::List>(initialParticles()); 
        Rcout << "Done" << endl;
    } else {
        Rcout << "using fixed initial particles" << endl;
        all_particles = Rcpp::as<Rcpp::List>(initParticles);
    }




    
    for (size_t it=0; it<(num_iter+num_burnin); it++) {

        N.fill(0);
        for (size_t i=0; i<n; i++)  {
            N(C(i), T(i))++; 
        }

        if ((it+1)%num_display == 0) {
            Rcout << "Iteration: " << it + 1 << " of " << num_iter + num_burnin << "  --  ";
            Rcout << "Cluster sizes (desc order, zeros not shown): " << endl;
            Rcout << "*** ";
            vec Ntmp(K, fill::zeros);
            for (size_t k=0; k<K; k++) {
                for (size_t j=0; j<J; j++) {
                    Ntmp(k) += N(j,k);
                }
            }
            Ntmp = sort(Ntmp, "descend");
            size_t kk=0;
            do {
                Rcout << Ntmp(kk) << " ";
                kk++;
            } while (Ntmp(kk)>0);
            Rcout << endl;
        }
        

        bool to_save_it = (it+1 > num_burnin) && ((it+1) % num_thin == 0);
        
        if (merge_step && it>0) {
            // Post-hoc merge
            for (size_t k=0; k < K-1; k++) {
                if (sum(N.col(k)) > 0) {
                    for (size_t l=k+1; l<K; l++) {
                        if (sum(N.col(l)) > 0) {
                            mat Omega_2 = Omega.slice(l);
                            add_diag_until_sympd(Omega_2, "KL(): Omega_2 is not sympd()");
                            double kl_div = KL( xi0.col(k), xi0.col(l), Omega.slice(k), Omega_2, alpha.col(k), alpha.col(l) );
                            if ( kl_div < R::qchisq(merge_par, (double)p, 1, 0) ) {
                                N.col(k) = N.col(k) + N.col(l);
                                N.col(l) = zeros<uvec>(J);
                                T(find(T==l)).fill(k);
                                Rcout << "Merged clusters (iteration " << it+1 << ")" << endl;
                                if (to_save_it)  Rcout << "Merged clusters after burn-in period (iteration " << it+1 << "). Consider longer burn-in." << endl;
                            }
                        }
                    }
                }
            }
        }

        // Post-hoc split
        // if (min_nclu > 1) {
        //     size_t nclu = 0;
        //     size_t k_largest = 0;
        //     size_t n_largest = 0;
        //     uvec Nsum(K); Nsum.fill(1);
        //     for (size_t k=0; k<K && nclu<min_nclu; k++) {
        //         size_t size_k = sum(N.col(k));
        //         if (size_k > 0) {
        //             nclu++;
        //             if (size_k > n_largest) {
        //                 k_largest = k;
        //                 n_largest = size_k;
        //             }
        //         } else {
        //             Nsum(k) = 0;
        //         }
        //     }

        //     if (nclu < min_nclu) {
        //         uvec indsT = find(T==k_largest);
        //         uvec indsN = find(Nsum==0);
        //         size_t cap = std::min(length(indsN), 3*min_nclu);
        //         for (size_t i=0; i<n_largest; i++) {
        //             size_t ith_empty_clu = indsN(i %% cap);
        //             T(indsT(i)) = ith_empty_clu;  // Assign label "ith_empty_clu" to obs i
        //         }
        //     }
        // }

        
        sampleGam();  
        getLogWs(logW);
        
        for (size_t k=0; k<K; k++) {

            Rcpp::List iterSummary = iter(k, N, all_particles);
            // all_particles[k] = Rcpp::as<Rcpp::List>(iterSummary["particles"]);
            Rcpp::List temp = all_particles[k];
            // Rcpp::List temp = all_particles["particles"];
            
            xi.slice(k) = mean(Rcpp::as<cube>(temp["xi"]), 2);
            xi0.col(k) = mean(Rcpp::as<mat>(temp["xi0"]),0).t();
            psi.col(k) = mean(Rcpp::as<mat>(temp["psi"]),0).t();
            G.slice(k) = reshape(mean(Rcpp::as<mat>(temp["G"]),0), p ,p);
            E.slice(k) = reshape(mean(Rcpp::as<mat>(temp["E"]),0), p ,p);
            z.cols(find(T==k)) = mean(Rcpp::as<mat>(temp["z"]),0);

            Omega.slice(k) = G.slice(k) + psi.col(k) * psi.col(k).t();
            vec inv_Omega_psi = solve(Omega.slice(k), psi.col(k));
            // mat inv_Omega = inv(Omega.slice(k));
            vec numerator = arma::sqrt(diagmat(Omega.slice(k).diag())) * inv_Omega_psi;
            double denominator = as_scalar(arma::sqrt(1 - psi.col(k).t() * inv_Omega_psi) );
            if (use_skew)  alpha.col(k) = numerator/denominator;

            log_py(k) = Rcpp::as<double>(iterSummary["log_py"]);
            perplexity(k) = Rcpp::as<double>(iterSummary["perplexity"]);
            nResampled(k) = Rcpp::as<size_t>(iterSummary["nResampled"]);
        }
        
        // Rcout << "** before sampleT()" << endl;
        sampleT(xi, Omega, alpha, logW);
        // Rcout << "** after sampleT()" << endl;
        
        if (to_save_it)
        {
            saveT.row(km) = T.t();
            saveZ.row(km) = z;
            if (to_save_W) {
                saveW.slice(km) = exp(logW);  // saveW.slice(km) = exp(logW);
            }
            saveXi.slice(km) = reshape( mat(xi.memptr(), xi.n_elem, 1, false), J, K*p);
            saveXi0.slice(km) = xi0;
            savePsi.slice(km) = psi;
            saveG.slice(km) = reshape( mat(G.memptr(), G.n_elem, 1, false), p, K*p);
            saveE.slice(km) = reshape( mat(E.memptr(), E.n_elem, 1, false), p, K*p);
            saveLog_py.row(km) = log_py.t();
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


// /* slow version (slow by necessity?) */
// // Polya-gamma data augmentation per 2021 Rigon Durante
// // I can definitely combine a lot of code, 
// // but idk how to store things like lfl<=T && T<(lfl+Km).
// // saveGam is (J, K-1, R)
// void PMC::sampleGam() {

//     double bump = 1 + std::numeric_limits<double>::epsilon(); // for numerical approximation errors

//     size_t levelm, Km, lfl; 
//     uvec Tij, adesc, ldesc;  // all/left descendants

//     for (size_t m=0; m<(K-1); m++) {

//         if (treestr == 1) {  // BT
//             levelm = log2(m+1) * bump;
//             Km = K / pow(2, levelm);  // number of leafs under m
//             lfl = (m+1) * Km - K;   // Subtract K to shift from K:(2K-1) to 0:(K-1)
//             // vec gam_mu_adapt = gam_mu * (1 + log2(m+1));
//         }

//         for (size_t j=0; j<J; j++) {

//             if (treestr == 1) {
//                 adesc = arma::find(lfl<=T && T<(lfl+Km) && C==j);
//                 Tij = T(adesc);
//                 ldesc = arma::find( Tij<(lfl+Km/2) );
//             } else if (treestr == 0) { // LT
//                 adesc = arma::find(T>=m && C==j);
//                 Tij = T(adesc);
//                 ldesc = arma::find( Tij==m );
//             }
            
//             // 1. Update PG data
//             mat psiXij = psiX.rows(adesc);  
//             vec gamvec = saveGam.tube(j, m);  // Compiler wants me to store this before multiplying.
//             vec pgdat = rpg(ones<vec>(adesc.size()), psiXij * gamvec);
            
//             // 2. Update gamma
//             vec kappa(adesc.size());
//             kappa.fill(-0.5);
//             kappa(ldesc).fill(0.5);
//             mat gam_Sigjm = arma::inv(psiXij.t() * diagmat(pgdat) * psiXij + gam_Sig_inv);
//             vec gam_mujm = gam_Sigjm * (psiXij.t() * kappa + gam_Sig_inv * gam_mu);
//             saveGam.tube(j, m) = mvrnormArma(1, gam_mujm, gam_Sigjm).t();
            
//         }
//     }
    
// }


/* slow version (slow by necessity?) */
// Polya-gamma data augmentation per 2021 Rigon Durante
// I can definitely combine a lot of code, 
// but idk how to store things like lfl<=T && T<(lfl+Km).
// saveGam is (J, K-1, R)
void PMC::sampleGam() {

    if (treestr == 1) {  // BT

        double bump = 1 + std::numeric_limits<double>::epsilon(); // for numerical approximation errors

        for (size_t m=0; m<(K-1); m++) {

            size_t levelm = log2(m+1) * bump;
            size_t Km = K / pow(2, levelm);  // number of leafs under m
            size_t lfl = (m+1) * Km - K;   // Subtract K to shift from K:(2K-1) to 0:(K-1)

            for (size_t j=0; j<J; j++) {
                
                uvec Cj_and_k_under_m = arma::find(lfl<=T && T<(lfl+Km) && C==j);
                uvec Tij = T(Cj_and_k_under_m);
                mat psiXij = psiX.rows(Cj_and_k_under_m);  
                
                // 1. Update PG data
                vec gamvec = saveGam.tube(j, m);  // Compiler wants me to store this before multiplying.
                vec pgdat = rpg1scale(psiXij * gamvec);  // vec pgdat = rpg(ones<vec>(Tij.size()), psiXij * gamvec);
                
                // 2. Update gamma
                vec kappa(Tij.size());
                kappa.fill(-0.5);
                kappa(arma::find( Tij<(lfl+Km/2) )).fill(0.5);
                mat gam_Sigjm = arma::inv(psiXij.t() * diagmat(pgdat) * psiXij + gam_Sig_inv);
                vec gam_mujm = gam_Sigjm * (psiXij.t() * kappa + gam_Sig_inv * gam_mu);
                saveGam.tube(j, m) = mvrnormArma(1, gam_mujm, gam_Sigjm).t();
                
            }
        }

    } else if (treestr == 0) { // LT

        for (size_t m=0; m<(K-1); m++) {
            for (size_t j=0; j<J; j++) {
                
                uvec ij = arma::find(T>=m && C==j);
                uvec Tij = T(ij);
                mat psiXij = psiX.rows(ij);  
                
                // 1. Update PG data
                vec gamvec = saveGam.tube(j, m);  // Compiler wants me to store this before multiplying.
                vec pgdat = rpg1scale(psiXij * gamvec);  // vec pgdat = rpg(ones<vec>(Tij.size()), psiXij * gamvec);
                
                // 2. Update gamma
                vec kappa(Tij.size());
                kappa.fill(-0.5);
                kappa(arma::find( Tij==m )).fill(0.5);
                mat gam_Sigjm = arma::inv(psiXij.t() * diagmat(pgdat) * psiXij + gam_Sig_inv);
                vec gam_mujm = gam_Sigjm * (psiXij.t() * kappa + gam_Sig_inv * gam_mu);
                saveGam.tube(j, m) = mvrnormArma(1, gam_mujm, gam_Sigjm).t();
                
            }
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
    logW.zeros();
    if (K==1)  return; 
    
    // Compute eta. 
    mat eta(n, K-1);
    for (size_t j=0; j<J; j++) {
        uvec C_j = arma::find(C==j);
        mat jGam = saveGam.row(j);  // (K-1, R) 
        if (R==1)  eta.rows(C_j) = psiX.rows(C_j) * jGam;  // (n_j, K-1)
        else  eta.rows(C_j) = psiX.rows(C_j) * jGam.t();  // (n_j, K-1)
    }

    // Compute log weights. 
    if (treestr == 1) {  // BT

        // Could probably do this more efficiently. Sth like:
        // vec signs(K-1);  // elements are -1, 0, or 1
        // set signs according to k;
        // logW.col(k) -= sum(log(1 + exp(signs*eta.col(l))));
        for (size_t k=0; k<K; k++) {
            size_t l=k+K;  // root=1 index scheme
            while (l > 1) {
                // int s = pow(-1, l%2 + 1);  // sign: want odd l to be 1, even l to be -1
                int s = (l%2 == 0 ? -1 : 1);
                l /= 2;
                logW.col(k) -= log(1 + exp( s*eta.col(l-1) ));
            }
        }

    } else if (treestr == 0) {  // LT

        logW.tail_cols(K-1) = cumsum(-log(1 + exp(eta)), 1);
        logW.head_cols(K-1) -= log(1 + exp(-eta));

    }

}






        
Rcpp::List PMC::initialParticles() {
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
    mat E0_scaled = E0/(e0 - p - 1.0);
    
    for (size_t k=0; k<K; k++) {
        uvec T_k = arma::find(T==k);
        size_t n_k = T_k.n_elem;

        mat z_k(num_particles, n_k, fill::zeros);
        mat psi_k(num_particles,p, fill::zeros);
        cube xi_k(J, p, num_particles);
        mat xi_0k(num_particles, p);
        mat G_k(num_particles, pow(p,2));
        mat E_k(num_particles, pow(p,2));

        if (n_k > p+2) {  // XX: lots of duplicate code. condense? 
        
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
            vec devz = sum(pow(zmat,2), 1);
            zmat.each_col() /= devz;
            mat ymat = Y_k.each_row() - mean_y;
            psi_k = zmat * ymat;
            // for (size_t iN=0; iN<num_particles; iN++) 
            //     for (size_t ip=0; ip<p; ip++) 
            //         psi_k(iN, ip) = as_scalar(zmat.row(iN) * ymat.col(ip));  
            // psi_k(iN, ip) = as_scalar(accu(zmat.row(iN).t() % ymat.col(ip)) / devz(iN));  
            
            xi_0k = -1.0*(psi_k % repmat(mean_absz, 1, p));   
            xi_0k.each_row() += mean_y;
            
            uvec C_k = C(T_k);
            for (size_t j=0; j<J; j++) {
                uvec j_indices = find(C_k==j);
                size_t n_jk = j_indices.n_elem;
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
                for (size_t i=0; i<n_k; i++) { Gsum += e.row(i).t() * e.row(i) / n_k; }  // XX: can do some more efficient matrix multiplication, I think
                G_k.row(iN) = vectorise(Gsum).t();
            }
        } 
        else { // if N_k==0, do same with all data, but return empty z
            size_t n_all = Y.n_rows;
            
            // This is clumsy, but apparently required for reproducibility,
            // when initializing with randn<mat>(num_particles, n_k) results 
            // are not consistent across platforms.
            mat temp_z_k( num_particles, n_all );
            for (size_t row=0; row<num_particles; row++) 
                for (size_t col=0; col<n_all; col++) 
                    temp_z_k(row,col) = randn();
            
            rowvec mean_y = mean(Y, 0);
            mat absz = abs(temp_z_k);
            vec mean_absz = mean(absz, 1);

            mat zmat = absz.each_col() - mean_absz;
            vec devz = sum(pow(zmat,2), 1);
            zmat.each_col() /= devz;
            mat ymat = Y.each_row() - mean_y;
            psi_k = zmat * ymat;
            
            xi_0k = -1.0*(psi_k % repmat(mean_absz, 1, p));
            xi_0k.each_row() += mean_y;
            vec mean_xi_0k = mean(xi_0k, 0).t();
            
            for (size_t j=0; j<J; j++) 
                xi_k.subcube(j,0,0,j,p-1,num_particles-1) = trans( xi_0k + mvrnormArma(num_particles, mean_xi_0k, E0_scaled) );
            
            for (size_t iN=0; iN<num_particles; iN++) {
                mat e = ( Y.each_row() - xi_0k.row(iN) );
                e -= ( repmat(psi_k.row(iN), n_all, 1) % repmat(abs(temp_z_k.row(iN).t()), 1, p) );  
                mat Gsum(p, p, fill::zeros);
                for (size_t i=0; i<n_all; i++) { Gsum += e.row(i).t() * e.row(i) / n_all; }
                G_k.row(iN) = vectorise(Gsum).t();
            }
            z_k.set_size( num_particles, 0 );
        } // end N_k==0
        
        for (size_t iN=0; iN<num_particles; iN++) {
            // mat E = inv_sympd(rWishartArma(invE0, e0));  
            mat E = inv_sympd(rWishartArmaChol(cholInvE0, e0));  
            E_k.row(iN) = vectorise(E).t();
        }
        
        Rcpp::List particles = Rcpp::List::create(
            Rcpp::Named( "z" ) = z_k,
            Rcpp::Named( "psi" ) = psi_k,
            Rcpp::Named( "xi" ) = xi_k,
            Rcpp::Named( "xi0" ) = xi_0k,
            Rcpp::Named( "G" ) = G_k,
            Rcpp::Named( "E" ) = E_k
            );
        
        all_particles(k) = particles;
    } // end k for loop
    return all_particles;
}


void PMC::sampleXi(const mat& Y_k, const uvec& C_k, const uvec& N_k, Rcpp::List& particles, mat& log_dQ, const size_t pp) {
    // Given the arguments, this function returns a population of MC draws 
    // for the values of the variable Xi, in the p-variate skew-N model.
    size_t nk = Y_k.n_rows;
    mat E = Rcpp::as<mat>(particles["E"]);
    mat xi0 = Rcpp::as<mat>(particles["xi0"]);
    
    cube xi(J, p, num_particles, fill::zeros);
    mat log_dxi(num_particles, J);
    
    if (nk == 0) {
        // Rcout << "empty xi" << endl;
        double sgn, ld;
        vec eigval; mat eigvec;
        for (size_t iN=0; iN<num_particles; iN++) {
            mat tempE = reshape(E.row(iN), p, p);
            log_det(ld, sgn, tempE);
            print_log_det(ld, sgn, "tempE", "sampleXi()");
            eig_sym(eigval, eigvec, tempE) ;
            mat invSqrtL_invU = solve(eigvec.t(), diagmat(1.0/sqrt(eigval))).t();
            // mat invSqrtL_invU = diagmat(1.0/sqrt(eigval)) * inv(eigvec);
            vec xi0_iN = xi0.row(iN).t();
            for (size_t j=0; j<J; j++) {
                // xi.slice(iN).row(j) = trans( mvnrnd(xi0.row(iN).t(), tempE) );
                xi.slice(iN).row(j) = mvrnormArma(1, xi0_iN, tempE);
                vec normedXi = invSqrtL_invU * trans(xi.slice(iN).row(j) - xi0.row(iN));
                log_dxi(iN, j) = as_scalar(-log2pi * (p/2.0) - 0.5*normedXi.t()*normedXi - 0.5*sgn*ld); 
            } // end j loop
        } // end iN loop
        // Rcout << "end empty xi" << endl;
    } 
    else { // if nk>0:

        // mat z = Rcpp::as<mat>(particles["z"]);
        mat absz = abs(Rcpp::as<mat>(particles["z"]));
        mat psi = Rcpp::as<mat>(particles["psi"]);
        mat G = Rcpp::as<mat>(particles["G"]);

        for (size_t iN=0; iN<num_particles; iN++) {

            xi.slice(iN) = repmat( xi0.row(iN), J, 1 );
            log_dxi.row(iN).fill(0);
            mat EiN = reshape(E.row(iN),p,p);
            add_diag_until_sympd(EiN, "sampleXi(): EiN is not sympd()");
            mat invE = inv_sympd(EiN);
            mat GiN = reshape(G.row(iN),p,p);
            add_diag_until_sympd(GiN, "sampleXi(): GiN is not sympd()");
            mat invG = inv_sympd(GiN);

            mat zin = absz.row(iN);
            for (size_t j=0; j<J; j++) {
                size_t njk = N_k(j);
                if (njk == 0) {
                    xi.slice(iN).row(j) = mvnrnd(xi0.row(iN).t(), EiN).t();
                    log_dxi(iN, j) = as_scalar(dmvnrm_arma_precision(xi.slice(iN).row(j), xi0.row(iN), EiN) );
                } else { // if njk>0
                    uvec jk_idx = find(C_k==j);
                    mat Yjk = Y_k.rows(jk_idx);
                    rowvec mean_y = mean(Yjk, 0);
                    rowvec zjk = zin.cols(jk_idx);
                    vec m = invE * xi0.row(iN).t() + zeta*njk*invG * trans( mean_y - psi.row(iN) * mean(zjk) );
                    mat v = inv_sympd(invE + zeta*njk*invG);
                    xi.slice(iN).row(j) = mvnrnd(v * m, v).t();
                    log_dxi(iN, j) = as_scalar(dmvnrm_arma_precision(
                                                                xi.slice(iN).row(j), trans(v * m),
                                                                invE + zeta*njk*invG ) );
                }
            } // end j loop

        } // end iN (particles) loop
    } // end nk>0 loop
    
    // for (size_t iN=0; iN<num_particles; iN++) {
    //     mat xtx = xi.slice(iN).t() * xi.slice(iN);
    //     if (!xtx.is_sympd()) {
    //         Rcout << "xtx(iN=" << iN << ") is not sympd. ";
    //     }
    // }

    particles["xi"] = xi;
    vec vec_log_dxi = sum(log_dxi, 1);
    log_dQ.col(pp) = vec_log_dxi;
    
}

void PMC::sampleG(const mat& Y_k, const uvec& C_k, const uvec& N_k, Rcpp::List& particles, mat& log_dQ, const size_t pp) {
    
    size_t nk = Y_k.n_rows;
    
    mat G(num_particles, pow(p,2));
    vec log_dG(num_particles);
    mat g(p,p);
    
    if (nk==0) {
        // Rcout << "empty G" << endl;
        for (size_t iN=0; iN<num_particles; iN++) {
            // g = inv_sympd(rWishartArma(invLamb, m0 ));
            mat invg = rWishartArmaChol(cholInvLamb, m0);
            add_diag_until_sympd(invg, "sampleG(): invg (nk==0) is not sympd()");
            g = inv_sympd(invg);
            G.row(iN) = vectorise(g).t();
            log_dG(iN) = dIWishartArmaS(g, m0, Lamb, true, ldLamb, sgnLamb);
            // log_dG(iN) = dIWishartArma(g, m0, Lamb);
        }
        // Rcout << "end empty G" << endl;
    } else { // if nk>0

        cube xi = Rcpp::as<cube>(particles["xi"]);
        mat xi0 = Rcpp::as<mat>(particles["xi0"]);
        mat absz = abs(Rcpp::as<mat>(particles["z"]));
        mat psi = Rcpp::as<mat>(particles["psi"]);

        mat ek(nk,p);
        mat zp(nk,p);
        mat Lambda_k(p,p);
        mat Sk(p,p);

        for (size_t iN=0; iN<num_particles; iN++ ) {
            Sk.fill(0);
            for (size_t j=0; j<J; j++) {
                size_t njk = N_k(j);
                if ( njk > 0 ) {
                    uvec jk_idx = find(C_k==j);
                    rowvec xi_j = xi.subcube(j, 0, iN, j, p-1, iN);
                    mat Y_jk = Y_k.rows(jk_idx);
                    mat ejk =  Y_jk.each_row() - xi_j;
                    vec zrow = trans( absz.row(iN) );
                    mat zpj = repmat( zrow(jk_idx) , 1 ,p);
                    zpj.each_row() %= psi.row(iN);
                    ejk -= zpj;
                    Sk += ejk.t() * ejk;
                }
            }
            if (is_Lamb_Zero) {
                Lambda_k = zeta * Sk;
                Rcout << "is_Lamb_Zero is true";
            }
            else
                Lambda_k = invLamb + zeta * Sk;
            
            add_diag_until_sympd(Lambda_k, "sampleG(): Lambda_k is not sympd()");
            mat inv_Lamb_k = inv_sympd(Lambda_k);
            mat invg = rWishartArma(inv_Lamb_k, ceil(zeta * nk) + m0 );
            add_diag_until_sympd(invg, "sampleG(): invg (nk>0) is not sympd()");
            g = inv_sympd(invg);
            G.row(iN) = vectorise(g).t();
            log_dG(iN) = dIWishartArma(g, nk+m0, Lambda_k);
            // log_dG(iN) = dIWishartArma(g, zeta*nk+m, Lambda_k);
        }
    }

    particles["G"] = G;
    log_dQ.col(pp) = log_dG;
}


void PMC::samplePsi(const mat& Y_k, const uvec& C_k, const uvec& N_k, Rcpp::List& particles, mat& log_dQ, const size_t pp) {
    
    size_t nk = Y_k.n_rows;
    size_t p = Y_k.n_cols;
    
    cube xi = Rcpp::as<cube>(particles["xi"]);
    mat G = Rcpp::as<mat>(particles["G"]);
    mat xi0 = Rcpp::as<mat>(particles["xi0"]);
    mat absz = abs(Rcpp::as<mat>(particles["z"]));
    vec sums_z2 = sum(pow(absz,2), 1);
    
    mat psi(num_particles, p);
    vec log_dpsi(num_particles);
    
    if (nk == 0) {
        // Rcout << "empty psi" << endl;
        // add sampling from prior on n-ball
        vec x(p);
        double logDetOmega;
        vec h;
        double ld, sgn;
        for (size_t iN=0; iN<num_particles; iN++) { 

            for (size_t pp=0; pp<p; pp++)  x(pp) = randn(); 
            
            double r = norm(x);
            double u = randu();

            if (false) {
                // sample from Sigma prior (take psi*psi element here to be zero)
                mat ginv = rWishartArmaChol(cholInvLamb, m0);
                // mat ginv = rWishartArma(invLamb, m0);
                // mat ginv = rWishartArma(invLamb, m, false, "invLamb");
                h = diagvec(inv(ginv));
                // Let D = sqrt( diagmat(g) ), where g = inv(ginv). Then -2*log|D| = accu(log(diagvec(g))). 
                // Then |inv(D) * inv(ginv) * inv(D)| = |inv(D)| * |inv(ginv)| * |inv(D)| = (1/|ginv|) * (1/|D|)^2,
                // so its log equals - log|ginv| - 2*log|D|.
                log_det(ld, sgn, ginv); 
                print_log_det(ld, sgn, "ginv", "samplePsi()"); 
                logDetOmega = - ld - accu(log(h));
            } else {  // XX: change back to this if possible  
                // sample from Sigma prior (take psi*psi element here to be zero)
                mat g = iwishrnd(Lamb, m0, cholLamb);  
                h = diagvec(g);
                log_det(ld, sgn, g); 
                print_log_det(ld, sgn, "g", "samplePsi()"); 
                logDetOmega = ld - accu(log(h));
            }

            // vec delta = pow(u, 1.0/p) * (1 / r) * pow(detOmega, p/2.0) * x;  
            vec delta = pow(u, 1.0/p) * (1 / r) * exp(logDetOmega*p/2.0) * x;  
            psi.row(iN) = trans(sqrt(h) % delta);  // psi.row(iN) = trans(diagmat(h) * delta);
            double logSphere = (p/2.0) * log(M_PI) - lgamma(p/2.0+1.0);
            log_dpsi(iN) = 1 / (logSphere * logDetOmega);
        }
        // Rcout << "end empty psi" << endl;
    } 
    else { // if nk>0:

        rowvec mpsi(p);
        for (size_t iN=0; iN<num_particles; iN++) {
            mpsi.fill(0);
            for (size_t j=0; j<J; j++) {
                size_t njk = N_k(j);
                if (njk > 0) {
                    uvec jk_idx = find(C_k==j);
                    rowvec xi_j = xi.subcube(j, 0, iN, j, p-1, iN);
                    mat Y_jk = Y_k.rows(jk_idx);
                    Y_jk.each_row() -= xi_j;
                    vec zrow = absz.row(iN).t();
                    Y_jk.each_col() %= zrow(jk_idx);
                    mpsi += sum(Y_jk, 0);
                } 
            } // end j loop
            mpsi /= sums_z2(iN);
            mat vpsi = reshape(G.row(iN) / (zeta*sums_z2(iN)), p, p);
            add_diag_until_sympd(vpsi, "samplePsi(): vpsi is not sympd()");
            psi.row(iN) = arma::mvnrnd(mpsi.t(), vpsi).t();
            log_dpsi(iN) = as_scalar( dmvnrm_arma_precision(psi.row(iN), mpsi, inv_sympd(vpsi)) );
        } // end iN (particles) loop

    } // end nk>0 if

    particles["psi"] = psi;
    log_dQ.col(pp) = log_dpsi;
    
}

void PMC::sampleZ(const mat& Y_k, const uvec& C_k, Rcpp::List& particles, mat& log_dQ, const size_t pp) {
    
    size_t nk = Y_k.n_rows;
    vec log_dZ(num_particles);
    log_dZ.fill(0); 
    mat z(num_particles, nk);
    
    if (nk > 0) {
        cube xi = Rcpp::as<cube>(particles["xi"]);
        mat psi = Rcpp::as<mat>(particles["psi"]);
        mat G = Rcpp::as<mat>(particles["G"]);
        vec tp(nk); vec ldZ(nk); vec u(nk);
        
        for (size_t iN=0; iN<num_particles; iN++) {
            mat GiN = reshape(G.row(iN), p, p);
            add_diag_until_sympd(GiN, "sampleZ(): GiN is not sympd()");
            mat invG = inv_sympd(GiN);
            double v = 1.0/(1.0 + zeta*as_scalar(psi.row(iN) * invG * psi.row(iN).t()));
            double sv = sqrt(v);
            u = randu(nk);
            // Rcout << "randu(nk)" << u(0) << endl;
            for (size_t i=0; i<nk; i++ ) {
                mat xi_j = xi.subcube(C_k(i), 0, iN, C_k(i), p-1, iN);
                double m = as_scalar(v * zeta * (psi.row(iN) * invG * (Y_k.row(i) - xi_j).t()));
                double absz = rtruncnormArma(m, sv, 0);
                z(iN, i) = (u(i) < 0.5 ? -absz : absz);  // z(iN, i) = sgn * absz;
                tp(i) = R::pnorm(0, -m, sv, true, true);
                ldZ(i) = R::dnorm(absz, m, sv, true);
            }
            log_dZ(iN) = accu( ldZ - tp - log(2) );
        }
    }

    particles["z"] = z;
    log_dQ.col(pp) = log_dZ;
    
}


void PMC::sampleXi0(const mat& Y_k, const uvec& N_k, Rcpp::List& particles, mat& log_dQ, const size_t pp) {
    size_t nk = Y_k.n_rows;

    mat psi = Rcpp::as<mat>(particles["psi"]);
    mat G = Rcpp::as<mat>(particles["G"]);
    mat E = Rcpp::as<mat>(particles["E"]);
    cube xi = Rcpp::as<cube>(particles["xi"]);
    mat absz = abs(Rcpp::as<mat>(particles["z"]));

    mat xi0(num_particles, p);
    xi0.fill(0);
    vec log_dxi0(num_particles);
    
    if(nk == 0) {
        xi0 = mvrnormArmaChol(num_particles, b0, cholB0);  // xi0 = mvrnormArma(num_particles, b0, B0);
        log_dxi0 =  dmvnrm_arma_precision(xi0, b0.t(), invB0 );
    } else { // if nk>0:
        for (size_t iN=0; iN<num_particles; iN++) {
            mat GiN = reshape(G.row(iN), p, p);
            add_diag_until_sympd(GiN, "sampleXi0(): GiN is not sympd()");
            mat invG = inv_sympd(GiN);
            uvec jk_idx = find(N_k>0);
            mat EiN = reshape(E.row(iN), p, p);
            add_diag_until_sympd(EiN, "sampleXi0(): EiN is not sympd()");
            mat invE = inv_sympd(EiN);
            mat inv_vxi0 = invB0 + zeta*jk_idx.n_elem*invE;
            add_diag_until_sympd(inv_vxi0, "sampleXi0(): inv_vxi0 is not sympd()");
            mat vxi0 = inv_sympd(inv_vxi0);
            mat xi_slice = xi.slice(iN);
            vec mean_xi = trans( mean(xi_slice.rows(jk_idx), 0) );
            vec mxi0 = vxi0 * (invB0*b0 + zeta * jk_idx.n_elem * invE * mean_xi);
            xi0.row(iN) = mvrnormArma(1, mxi0, vxi0);
            log_dxi0(iN) =  as_scalar( dmvnrm_arma_precision(xi0.row(iN), mxi0.t(), inv_vxi0) );
        } // end iN (particles) loop
    } // end nk>0 loop


    particles["xi0"] = xi0;
    log_dQ.col(pp) = log_dxi0;

}

void PMC::sampleE(const uvec& N_k, Rcpp::List& particles, mat& log_dQ, const size_t pp) {
    
    size_t nk = accu(N_k);
    
    mat E(num_particles, pow(p,2));
    vec log_dE(num_particles);
    
    if (nk==0) {
        mat e;
        for (size_t iN=0; iN<num_particles; iN++) {
            if (true) {
                while (!iwishrnd(e, E0, e0, cholE0)) {
                    Rcout << "sampleE(): iwishrnd(e, E0, e0, cholE0) is false. sampling again." << endl;
                }
            } else {
                e = inv_sympd(rWishartArmaChol( cholInvE0, e0 ));
            }
            E.row(iN) = vectorise(e).t();
            log_dE(iN) = dIWishartArmaS(e, e0, E0, true, ldE0, sgnE0);
        }
        
    } 
    else { // if nk>0

        uvec jk_idx = find(N_k>0);
        cube xi = Rcpp::as<cube>(particles["xi"]);
        mat xi0 = Rcpp::as<mat>(particles["xi0"]);
        double ek = e0 + jk_idx.n_elem;

        for (size_t iN=0; iN<num_particles; iN++) {
            mat xi_slice = xi.slice(iN).rows(jk_idx);
            xi_slice.each_row() -= xi0.row(iN);
            mat mE = invE0 + xi_slice.t() * xi_slice;
            std::string iNstr = " iN=";
            iNstr = iNstr.append(std::to_string(iN));
            add_diag_until_sympd(mE, iNstr.append("sampleE(): mE is not sympd()"));
            mat mEinv = inv_sympd(mE);
            mat e;
            if (true) {        
                while (!iwishrnd(e, mEinv, ek)) {
                    Rcout << "sampleE(): iwishrnd(e, mEinv, ek) is false. sampling again." << endl;
                }
            } else {
                e = inv( rWishartArma( mE, ek ) );
            }
            add_diag_until_sympd(e, iNstr.append("sampleE(): e is not sympd()"));
            E.row(iN) = vectorise(e).t();
            log_dE(iN) = dIWishartArma(e, ek, mEinv);
        }

    }

    particles["E"] = E;
    log_dQ.col(pp) = log_dE;

}


// ----------------------------------------------------------------------------------

arma::vec PMC::logPriorDens( Rcpp::List& particles ) {

    cube xi = Rcpp::as<cube>(particles["xi"]);
    mat xi0 = Rcpp::as<mat>(particles["xi0"]);
    mat psi = Rcpp::as<mat>(particles["psi"]);
    mat G = Rcpp::as<mat>(particles["G"]);
    mat E = Rcpp::as<mat>(particles["E"]);

    vec logDetSigma(num_particles);
    vec logPriorG(num_particles);
    vec logPriorE(num_particles);

    double logpxi = 0;

// #pragma omp parallel for 
    for (size_t iN=0; iN<num_particles; iN++ ) {
        mat Gpsi = reshape(G.row(iN), p, p) + psi.row(iN).t() * psi.row(iN);
        add_diag_until_sympd(Gpsi, "logPriorDens(): Gpsi is not sympd()");
        double ld, sgn;
        log_det(ld, sgn, Gpsi);
        // if (sgn <= 0)  { 
        //     Rcout << "Is Gpsi symmetric - " << Gpsi.is_symmetric() << ",  ";
        //     Rcout << "Is Gpsi sympd - " << Gpsi.is_sympd() << ",  ";
        //     Rcout << "Sign of det(Gpsi) is " << sgn;
        //     Rcout << "log_det(Gpsi) is " << ld;
        //     // mat abs_psi_iN = abs(psi_iN);
        //     // Rcout << "abs(min/max) of psi_iN: " << abs_psi_iN.min() << " " << abs_psi_iN.max() << " - ";
        //     // mat abs_pre_Sigma = abs(pre_Sigma);
        //     // Rcout << "abs(min/max) of abs_pre_Sigma: " << abs_pre_Sigma.min() << " " << abs_pre_Sigma.max() << " - ";
        //     mat abs_Gpsi = abs(Gpsi);
        //     double abs_Gpsi_min = abs_Gpsi.min();
        //     Rcout << "  abs(min/max) of Gpsi: " << abs_Gpsi_min << " " << abs_Gpsi.max() << " - ";
        //     Rcout << "  adding eye(p,p) * abs_Gpsi_min * 1e-09 to Gpsi to try to make it nonsingular  " << endl;
        //     Gpsi += eye(p,p) * abs_Gpsi_min * 1e-09;
        //     Rcout << "Is new Gpsi symmetric - " << Gpsi.is_symmetric() << ",  ";
        //     Rcout << "Is new Gpsi sympd - " << Gpsi.is_sympd() << ",  ";
        //     log_det(ld, sgn, Gpsi);
        //     Rcout << "Sign of new det(Gpsi) is " << sgn << ",  ";
        //     Rcout << "New log_det(Gpsi) is " << ld;
        //     // Rcout << "  attempting to avert overflow issues.  ";
        //     // log_det(ld, sgn, Gpsi / abs_Gpsi_min);
        //     // Rcout << "sign of det(Gpsi / abs_Gpsi_min) is " << sgn;
        //     // ld -= p*log(abs_Gpsi_min);
        // }
        logDetSigma(iN) = ld;  // = log(det(Sigma));

        mat tempE = reshape(E.row(iN), p, p);
        mat invtempE = inv(tempE); 
        rowvec xi0_iN = xi0.row(iN);
        for (size_t j=0; j<J; j++) {
            mat xi_iN_j = xi.slice(iN).row(j);
            // logpxi += as_scalar( dmvnrm_arma_fast(xi_iN_j, xi0_iN, tempE) );  // add back in?
            logpxi += as_scalar( dmvnrm_arma_precision(xi_iN_j, xi0_iN, invtempE) );  
        }

        // logPriorE(iN) = dIWishartArma(tempE, e0, E0, true, false, "tempE", "E0");
        // logPriorE(iN) = dIWishartArmaWithlds(tempE, e0, E0, true, false, "tempE", "E0", ldE0, sgnE0);
        logPriorE(iN) = dIWishartArmaS(tempE, e0, E0, true, ldE0, sgnE0);
        if (m0==0) {
            logPriorG(iN) = -((p+1.0)/2.0) * logDetSigma(iN);
        } else {
            // logPriorG(iN) = dIWishartArma(Sigma, m, Lamb, true, false, "Sigma", "Lamb");
            // logPriorG(iN) = dIWishartArmaWithlds(Sigma, m, Lamb, true, false, "Sigma", "Lamb", ldLamb, sgnLamb);
            logPriorG(iN) = dIWishartArmaS(Gpsi, m0, Lamb, true, ldLamb, sgnLamb);
        }
    }

    double logSphere = (p/2.0) * log(M_PI) - lgamma(p/2.0+1.0);

    vec logPriorDens = logpxi - logSphere - 0.5*logDetSigma + logPriorG + logPriorE;

    return logPriorDens;
}

arma::vec PMC::logPostDens(const mat& Y_k, const uvec& C_k, const uvec& N_k, Rcpp::List& particles) {
    size_t nk = Y_k.n_rows;
    vec log_lik(num_particles, fill::zeros);  // loglikelihood.fill(0);

    if (nk > 0) {

        cube xi = Rcpp::as<cube>(particles["xi"]);
        mat psi = Rcpp::as<mat>(particles["psi"]);
        mat G = Rcpp::as<mat>(particles["G"]);
        mat absz = abs(Rcpp::as<mat>(particles["z"]));  // (npart x ??) matrix

#pragma omp parallel for 
        for (size_t iN=0; iN<num_particles; iN++) {
            mat g = reshape(G.row(iN), p, p);
            double log_det_g, sgn;
            log_det(log_det_g, sgn, g);
            print_log_det(log_det_g, sgn, "g", "logPostDens()");
            g = inv(g);  // save space  
            rowvec absz_iN = absz.row(iN);
            mat e = Y_k - absz_iN.t() * psi.row(iN);
            for (size_t j=0; j<J; j++) {
                size_t njk = N_k(j);
                if (njk > 0) {
                    rowvec xi_j = xi.subcube(j, 0, iN, j, p-1, iN);
                    uvec jk = find(C_k==j);
                    mat e_jk = e.rows(jk);
                    e_jk.each_row() -= xi_j;
                    mat ee = e_jk.t() * e_jk;
                    // double accu_g_ee = accu(g % ee);   // PsiVec(iN) = accu(inv(g) % ee);  
                    // vec absz_iN_jk = absz_iN(jk);
                    double sums_z2 = dot(absz_iN(jk), absz_iN(jk)); 
                    log_lik(iN) -= 0.5 * as_scalar(njk*log_det_g + accu(g % ee) + sums_z2);
                }
            }
        }
        // loglik
        double loglik_normalizing_const = - nk * (p+1.0)/2.0 * log(2.0*M_PI);
        log_lik += loglik_normalizing_const;
    } // end nk>0 


    vec log_prior = logPriorDens(particles);
    // Numerator of the unnormalized importance weights
    vec log_pi = log_prior + zeta * log_lik;

    return log_pi;
}

Rcpp::List PMC::iter(const size_t k, const umat& N, Rcpp::List& all_particles) {
    
    uvec T_k = arma::find(T==k);
    mat Y_k = Y.rows(T_k);
    uvec C_k = C(T_k);
    Rcpp::List particles = all_particles[k];
    
    size_t nk = Y_k.n_rows;

    // Proposal step (with permuted sweeps):

    mat log_dQ(num_particles, 8, fill::zeros);
    sampleZ(Y_k, C_k, particles, log_dQ, 1);  // drawList = sampleZ(Y_k, C_k, particles);

    uvec parPerm = randsamp(5,3,7);  
    uvec N_k = N.col(k);  // so that I can pass by reference in the sampleXX() functions
    // Rcout << "**iter(): before sample loop" << endl;
    for (size_t pp : parPerm) {  
        switch (pp) {
            case 3: sampleXi(Y_k, C_k, N_k, particles, log_dQ, pp); break;
            case 4: sampleG(Y_k, C_k, N_k, particles, log_dQ, pp); break;
            case 5: samplePsi(Y_k, C_k, N_k, particles, log_dQ, pp); break;
            case 6: sampleXi0(Y_k, N_k, particles, log_dQ, pp); break;
            case 7: sampleE(N_k, particles, log_dQ, pp); break;
        }
    }
    
    vec log_pitilde;
    if (nk>0) {
        log_pitilde = logPostDens(Y_k, C_k, N.col(k), particles);
    } else {
        log_pitilde = logPriorDens(particles);
    }
    vec log_q = sum(log_dQ, 1);
    // Rcout << "q " << log_q.t() << endl;
    vec log_iw = log_pitilde - log_q;
    // Rcout << "log iw raw " << log_iw.t() << endl;
    double cnst = max(log_iw);  // Constant needed for the computation of the marginal likelihood
    double log_py = cnst + log(accu(exp(log_iw-cnst))) - log(num_particles);
    vec iw_bar = exp(log_iw);
    vec log_iw_b = log_iw - log(num_particles) - log_py;
    vec iw_b = exp(log_iw_b);  // unnormalized weights
    // Rcout << "log iw unnormalized " << iw_b.t() << endl;
    // Rcout << "iw unnormalized " << exp(iw_b.t()) << endl;
    vec iw = iw_b/accu(iw_b); // normalized weights
    // Rcout << "iw final " << iw.t() << endl;
    double perplexity = exp(-accu(iw % log(iw))) / num_particles;

    
    // Rcout << "end copmute weights" << endl;
    
    
    // if (nk==0) Rcout << "start empty sample with replacement " << endl;
    
    uvec resamp(num_particles);
    for (size_t iN=0; iN<num_particles; iN++) {
        resamp(iN) = sampling(iw);
    }
    
    uvec uresamp = unique(resamp);
    size_t nResampled = uresamp.n_elem;

    // // Resampling step
    // Rcout << "start resample" << endl;
    cube MMM = Rcpp::as<cube>(particles["xi"]);
    cube MM = MMM;
    for (size_t i=0; i<num_particles; i++) {
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

    all_particles[k] = particles;

    // Rcout << "end resampling" << endl;
    
    return Rcpp::List::create(
        // Rcpp::Named( "particles" ) = particles,
        Rcpp::Named( "log_py" ) = log_py,
        Rcpp::Named( "nResampled" ) = nResampled,
        Rcpp::Named( "perplexity" ) = perplexity  );
}

void PMC::sampleT(const arma::cube& xi, const arma::cube& Omega, const arma::mat& alpha, const arma::mat& logW)
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
    
    NumericVector U = runif(n);
#pragma omp parallel for
    for (size_t i=0; i<n; i++) {
        // vec prob = PT.row(i).t();
        vec probsum = cumsum(PT.row(i).t());
        double x = U(i) * probsum.back();
        
        size_t k=0;
        while (k<(K-1) && probsum(k)<x) { k++; }
        T(i) = k;
    }

}

void PMC::clearNonSaves() {
    // Clear memory before the hefty operation of returning the chain.
    // Make sure each object's capacity (not just size) also becomes zero.
    // .capacity()
    // .shrink_to_fit() vs swap() strategy
    // .set_size() vs .reset()
    // To forcefully release memory at any point, use .reset(); note that in normal use this is not required. 
    Rcout << "start of clearNonSaves()" << endl;
    Y.reset();
    psiX.reset();
    // psiX_uniq.reset();
    // psiX_map.reset();
    C.reset();
    T.reset();

    // Lamb.reset(); invLamb.reset(); cholLamb.reset(); cholInvLamb.reset();
}

//--------------------------------------------------------

Rcpp::List PMC::get_chain()
{
    Rcout << "start of get_chain()" << endl;
    clearNonSaves();
    return Rcpp::List::create(
        Rcpp::Named( "t" ) = saveT,
        Rcpp::Named( "z" ) = saveZ,
        // Rcpp::Named( "W" ) = saveW,
        Rcpp::Named( "xi" ) = saveXi,
        Rcpp::Named( "xi0" ) = saveXi0,
        Rcpp::Named( "psi" ) = savePsi,
        Rcpp::Named( "G" ) = saveG,
        Rcpp::Named( "E" ) = saveE,
        Rcpp::Named( "log_py" ) = saveLog_py,
        Rcpp::Named( "perplexity" ) = savePerplexity,
        Rcpp::Named( "nResampled" ) = saveNResampled,
        Rcpp::Named( "Omega" ) = saveOmega,
        Rcpp::Named( "alpha" ) = saveAlpha
    );
}

void PMC::get_chain_v2(Rcpp::List& chain, bool to_clr_saves)
{
    Rcout << "start of get_chain_v2()" << endl;
    clearNonSaves();

    chain["t"] = saveT;
    if (to_clr_saves)  saveT.reset();
    chain["z"] = saveZ;
    if (to_clr_saves)  saveZ.reset();
    // chain["W"] = saveW;
    // if (to_clr_saves)  saveW.reset();
    chain["xi"] = saveXi;
    if (to_clr_saves)  saveXi.reset();
    chain["xi0"] = saveXi0;
    if (to_clr_saves)  saveXi0.reset();
    chain["psi"] = savePsi;
    if (to_clr_saves)  savePsi.reset();
    chain["G"] = saveG;
    if (to_clr_saves)  saveG.reset();
    chain["E"] = saveE;
    if (to_clr_saves)  saveE.reset();
    chain["log_py"] = saveLog_py;
    if (to_clr_saves)  saveLog_py.reset();
    chain["perplexity"] = savePerplexity;
    if (to_clr_saves)  savePerplexity.reset();
    chain["nResampled"] = saveNResampled;
    if (to_clr_saves)  saveNResampled.reset();
    chain["Omega"] = saveOmega;
    if (to_clr_saves)  saveOmega.reset();
    chain["alpha"] = saveAlpha;
    if (to_clr_saves)  saveAlpha.reset();

}


