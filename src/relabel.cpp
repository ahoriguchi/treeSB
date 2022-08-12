#include "armaMunkres.h"
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
Rcpp::List relabel(const Rcpp::List res)
{
    Rcpp::List chain = clone(Rcpp::as<Rcpp::List>(res["chain"]));
    Rcpp::List prior = Rcpp::as<Rcpp::List>(res["prior"]);
    int K = Rcpp::as<int>(prior["K"]);
    Rcout << "K=" << K << "\t";
    umat t_relabel = Rcpp::as<umat>(chain["t"]);
    int T = t_relabel.n_rows; 
    Rcout << "T=" << T << "\t";
    int N = t_relabel.n_cols;
    Rcout << "N=" << N << "\t";
    urowvec refZ = t_relabel.row(T-1);
    
    // Rcout << unique(t_relabel) << endl;
    // mat z_relabel = Rcpp::as<mat>(chain["z"]);
    cube W_relabel;
    bool useW = chain.containsElementNamed("W");
    if (useW) {
        W_relabel = Rcpp::as<cube>(chain["W"]);
    }
    cube Wsmall_relabel = Rcpp::as<cube>(chain["Wsmall"]);
    cube xi_relabel = Rcpp::as<cube>(chain["xi"]);
    size_t J = xi_relabel.n_rows;
    Rcout << "J=" << J << "\t";
    size_t p = xi_relabel.n_cols / K;
    Rcout << "p=" << p << endl;
    cube xi0_relabel = Rcpp::as<cube>(chain["xi0"]);
    cube psi_relabel = Rcpp::as<cube>(chain["psi"]);
    cube G_relabel = Rcpp::as<cube>(chain["G"]);
    cube E_relabel = Rcpp::as<cube>(chain["E"]);
    // umat S_relabel = Rcpp::as<umat>(chain["S"]);
    mat log_py_relabel = Rcpp::as<mat>(chain["log_py"]);
    mat perplexity_relabel = Rcpp::as<mat>(chain["perplexity"]);
    mat nResampled_relabel = Rcpp::as<mat>(chain["nResampled"]);
    cube Omega_relabel = Rcpp::as<cube>(chain["Omega"]);
    cube alpha_relabel = Rcpp::as<cube>(chain["alpha"]);
    cube gamma = Rcpp::as<cube>(chain["gamma"]);
    
    // Rcout << "defined relabels" << endl;
    
    uvec refZobs(K, fill::zeros);
    for (size_t i=0; i<N; i++) {
        refZobs(refZ(i))++;
    }
    
    // Rcout << "did refZobs" << endl;
    
    mat cost(K,K);
    umat permut;
    umat brmat(T-1,K);
    
    for (size_t s=0; s<(T-1); s++) {

        urowvec currZ = t_relabel.row(s);
        size_t refClass, curClass;
        // Find relabeling cost matrix:
        for (size_t h=0; h<K; h++) {
            curClass = refZobs(h);
            for (size_t j=0; j<K; j++) {
                cost(h,j) = curClass;
            }
        }
        
        for (size_t i=0; i<N; i++) {
            refClass = refZ(i);
            curClass = currZ(i);
            // Rcout << refClass <<" "<< curClass << endl;
            cost(refClass,curClass) -= 1.0;
        }
        
        permut = hungarian_cc(cost);
        // Rcout << permut << endl;
        // Rcout << "did permut" << endl;    
        
        urowvec back_relabeling(K);
        for (size_t col=0; col<K; col++) {
            for (size_t row=0; row<K; row++) {
                if (permut(row,col)==1) {
                    back_relabeling(col) = row;
                }
            }
        }
        
        urowvec t_copy = t_relabel.row(s);
        for (size_t i=0; i<N; i++) {
            t_relabel(s, i) = back_relabeling(t_copy(i));
        }
        
        // Rcout << "did t" << endl;
        
        mat W_copy;
        if (useW)  W_copy = W_relabel.slice(s);
        mat Wsmall_copy = Wsmall_relabel.slice(s);
        mat xi_copy = xi_relabel.slice(s);
        mat xi0_copy = xi0_relabel.slice(s);
        mat psi_copy = psi_relabel.slice(s);
        mat G_copy = G_relabel.slice(s);
        mat E_copy = E_relabel.slice(s);
        // urowvec S_copy = S_relabel.row(s);
        rowvec log_py_copy = log_py_relabel.row(s);
        rowvec perplexity_copy = perplexity_relabel.row(s);
        rowvec nResampled_copy = nResampled_relabel.row(s);
        mat Omega_copy = Omega_relabel.slice(s);
        mat alpha_copy = alpha_relabel.slice(s);
        for (size_t k=0; k<K; k++) {
            // uvec idx = find(relabeling==k);
            // int kk = relabeling(idx(0));
            size_t kk = back_relabeling(k);
            if (useW)  W_relabel.slice(s).col(k) = W_copy.col(kk);
            Wsmall_relabel.slice(s).col(k) = Wsmall_copy.col(kk);
            xi_relabel.subcube(0,k*p,s,J-1,k*p+p-1,s) = 
                xi_copy.submat(0,kk*p,J-1,kk*p+p-1);
            xi0_relabel.slice(s).col(k) = xi0_copy.col(kk);
            psi_relabel.slice(s).col(k) = psi_copy.col(kk);
            G_relabel.subcube(0,k*p,s,p-1,k*p+p-1,s) = 
                G_copy.submat(0,kk*p,p-1,kk*p+p-1);
            E_relabel.subcube(0,k*p,s,p-1,k*p+p-1,s) = 
                E_copy.submat(0,kk*p,p-1,kk*p+p-1);
            log_py_relabel(s, k) = log_py_copy(kk);
            perplexity_relabel(s, k) = perplexity_copy(kk);
            nResampled_relabel(s, k) = nResampled_copy(kk);
            Omega_relabel.subcube(0,k*p,s,p-1,k*p+p-1,s) = 
                Omega_copy.submat(0,kk*p,p-1,kk*p+p-1);
            alpha_relabel.slice(s).col(k) = alpha_copy.col(kk);
            brmat(s, k) = kk;
        }

    }

    if (useW) {
        return Rcpp::List::create(
            Rcpp::Named( "t" ) = t_relabel+1,
            Rcpp::Named( "W" ) = W_relabel,
            Rcpp::Named( "Wsmall" ) = Wsmall_relabel,
            Rcpp::Named( "xi" ) = xi_relabel,
            Rcpp::Named( "xi0" ) = xi0_relabel,
            Rcpp::Named( "psi" ) = psi_relabel,
            Rcpp::Named( "G" ) = G_relabel,
            Rcpp::Named( "E" ) = E_relabel,
            // Rcpp::Named( "S" ) = S_relabel,
            // Rcpp::Named( "varphi" ) = Rcpp::as<vec>(chain["varphi"]),
            // Rcpp::Named( "a0" ) = Rcpp::as<vec>(chain["a0"]),
            Rcpp::Named( "log_py" ) = log_py_relabel,
            Rcpp::Named( "perplexity" ) = perplexity_relabel,
            Rcpp::Named( "nResampled" ) = nResampled_relabel,
            Rcpp::Named( "Omega" ) = Omega_relabel,
            Rcpp::Named( "alpha" ) = alpha_relabel,
            Rcpp::Named( "gamma" ) = gamma,
            Rcpp::Named( "brmat" ) = brmat
        );
    }
    
    return Rcpp::List::create(
        Rcpp::Named( "t" ) = t_relabel+1,
        // Rcpp::Named( "W" ) = W_relabel,
        Rcpp::Named( "Wsmall" ) = Wsmall_relabel,
        Rcpp::Named( "xi" ) = xi_relabel,
        Rcpp::Named( "xi0" ) = xi0_relabel,
        Rcpp::Named( "psi" ) = psi_relabel,
        Rcpp::Named( "G" ) = G_relabel,
        Rcpp::Named( "E" ) = E_relabel,
        // Rcpp::Named( "S" ) = S_relabel,
        // Rcpp::Named( "varphi" ) = Rcpp::as<vec>(chain["varphi"]),
        // Rcpp::Named( "a0" ) = Rcpp::as<vec>(chain["a0"]),
        Rcpp::Named( "log_py" ) = log_py_relabel,
        Rcpp::Named( "perplexity" ) = perplexity_relabel,
        Rcpp::Named( "nResampled" ) = nResampled_relabel,
        Rcpp::Named( "Omega" ) = Omega_relabel,
        Rcpp::Named( "alpha" ) = alpha_relabel,
        Rcpp::Named( "gamma" ) = gamma,
        Rcpp::Named( "brmat" ) = brmat
    );
}
