#include <RcppArmadillo.h>
#define NDEBUG 1
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(RcppNumerical)]]
#include <RcppNumerical.h>
#include "RNG.h"
#include "PolyaGamma.h"

// See 
// https://github.com/RcppCore/RcppArmadillo/issues/116
// for reason to include #define NDEBUG 1

using namespace Numer;
using namespace Rcpp;
using namespace arma;
using namespace std;

const double log2pi = std::log(2.0 * M_PI);

arma::mat mvrnormArma(int n, arma::vec mu, arma::mat sigma) 
{
    int ncols = sigma.n_cols;
    // arma::mat Y = arma::randn(n, ncols);
    
    // This is clumsy, but apparently reuiqred for reproducibility,
    // when initializing with randn<mat>(num_particles, n_k) results 
    // are not consistent across platforms. 
    arma::mat Y(n, ncols);
    for (int row=0; row<n; row++) {
        for (int col=0; col<ncols; col++) {
            Y(row,col) = randn();
        }
    }
    // Rcout << "Y = arma::randn(n, ncols)" << Y(0,0) << endl;
    
    
    return arma::repmat(mu, 1, n).t() + Y * arma::chol(sigma);
}

arma::mat rWishartArma(arma::mat Sigma, int df)
{
    int p = Sigma.n_rows;
    vec m(p); m.zeros();
    mat X = mvrnormArma(df, m, Sigma );
    return X.t()*X;
}

arma::vec dmsnArma(arma::mat y, arma::rowvec xi, arma::mat omega, arma::vec alpha, bool logd = false) {

    int p = y.n_cols;
    // int m = y.n_rows;
    double logconst = log(2);
    
    mat Y = trans(y.each_row() - xi);
    vec Q = sum(arma::solve(omega, Y) % Y, 0).t();
    // vec Q = trans( sum( ( inv_sympd(omega) * Y ) % Y, 0) );
    vec L = trans(Y.each_col() / sqrt(omega.diag())) * alpha;

    vec logPDF = ( log( arma::normcdf(L) ) - 0.5 * Q);
    logPDF += logconst;
    double ldo, sgno;
    log_det(ldo, sgno, omega);
    logPDF -= 0.5 * (p * log(2.0 * datum::pi) + ldo);

    return (logd ? logPDF : exp(logPDF));
}

int sampling(vec probs) {  

    int Kf = probs.n_elem;
    double x = R::runif(0.0, sum(probs));    
    double psum = 0;

    for (int k=0; k<Kf; k++) {  
        psum += as_scalar(probs(k));
        if (x < psum) {
            return k;
        }
    }

    return Kf-1;
}


/* Exponential rejection sampling (a,inf) */
double ers_a_inf(double a) {
    const double ainv = 1.0 / a;
    double x, rho;
    do {
        x = -ainv * log(arma::randu<double>()) + a;
        // Rcout << "log(arma::randu<double>())" << x << " ";
        rho = exp(-0.5 * pow((x - a), 2));
    } while (arma::randu<double>() > rho);
    return x;
}

/* Normal rejection sampling (a,inf) */
double nrs_a_inf(double a) {
    double x = -DBL_MAX;
    while (x < a) {
        x = arma::randn<double>();
    }
    return x;
}

double rtruncnormArma(double mu, double sigma, double a) {
    static const double t = 0.45;
    const double alpha = (a - mu) / sigma;
    if (alpha < t) {
        return mu + sigma * nrs_a_inf(alpha);
    } else {
        return mu + sigma * ers_a_inf(alpha);
    }
}

arma::vec dmvnrm_arma_precision(arma::mat x, arma::rowvec mean, arma::mat omega, bool logd = true) { 

    int n = x.n_rows;
    int xdim = x.n_cols;
    arma::vec out(n);
    arma::mat rooti = trimatu(arma::chol(omega));
    double rootisum = arma::sum(log(rooti.diag()));
    double constants = -(static_cast<double>(xdim)/2.0) * log2pi;
    
    for (int i=0; i < n; i++) {
        arma::vec z = rooti * arma::trans( x.row(i) - mean) ;    
        out(i)      = constants - 0.5 * arma::sum(z%z) + rootisum;     
    }  
    
    return (logd ? out : exp(out));
}

double dIWishartArmaHelp(arma::mat& W, const double v, arma::mat& S, const bool logd = true, const double ldS=0, const double sgnS=0) {
    
    double ldW, sgnW;  // sign of det, not of logdet
    log_det(ldW, sgnW, W);  // use log_det_sympd()?
    
    int p = S.n_rows;
    double log_gammapart = 0;
    for (int i=0; i<p; i++) {
        log_gammapart += lgamma((v - i)/2.0);
    }
    double log_denom = log_gammapart + (v * p/2.0) * log(2.0) + (p * (p - 1)/4.0) * log(M_PI);
    mat hold = arma::solve(W.t(), S.t()).t();  // mat hold = S * inv_sympd(W);
    double log_num = (v/2) * ldS + (-(v + p + 1)/2.0) * ldW - 0.5 * trace(hold);  // double log_num = (v/2) * log(det(S)) + (-(v + p + 1)/2.0) * log(det(W)) - 0.5 * trace(hold);
    double log_out = log_num - log_denom;
    return (logd ? log_out : exp(log_out));
}

double dIWishartArma(arma::mat& W, const double v, arma::mat& S, const bool logd = true) {
    
    double ldS, sgnS;  // sign of det, not of logdet
    log_det(ldS, sgnS, S);  // use log_det_sympd()?
    
    return dIWishartArmaHelp(W, v, S, logd, ldS, sgnS);
}

// double dIWishartArma(arma::mat W, double v, arma::mat S, bool logd = true) {
    
//     double ldW, sgnW;  // sign of det, not of logdet
//     log_det(ldW, sgnW, W);  // use log_det_sympd()?
    
//     double ldS, sgnS;  // sign of det, not of logdet
//     log_det(ldS, sgnS, S);  // use log_det_sympd()?
    
//     int p = S.n_rows;
//     double log_gammapart = 0;
//     for (int i=0; i<p; i++) {
//         log_gammapart += lgamma((v - i)/2.0);
//     }
//     double log_denom = log_gammapart + (v * p/2.0) * log(2.0) + (p * (p - 1)/4.0) * log(M_PI);
//     mat hold = arma::solve(W.t(), S.t()).t();
//     // mat hold = S * inv_sympd(W);
//     double log_num = (v/2) * ldS + (-(v + p + 1)/2.0) * ldW - 0.5 * trace(hold);
//     // double log_num = (v/2) * log(det(S)) + (-(v + p + 1)/2.0) * log(det(W)) - 0.5 * trace(hold);
//     double log_out = log_num - log_denom;

//     return (logd : log_out ? exp(log_out));
//     // if (logd == false) {
//     //     log_out = exp(log_out);
//     // }
//     // return(log_out);
// }

uvec randsamp(int n, int min, int max) {
    int r,i=n;
    uvec a(n);
    while (i--) {
        r = (max-min+1-i);
        // a[i] = min += (r ? randi()%r : 0);
        a[i] = min += (r ? randi()%r : 0);
        min++;
    }
    while (n>1) {
        // r = a[i=randi()%n--];
        r = a[i=randi()%n--];
        a[i]=a[n];
        a[n]=r;
    }
    return a;
}


class SNlog2Phi: public Func
{
private:
    double xi;
    double om;
    double al;
public:
    SNlog2Phi(double xi_, double om_, double al_) : xi(xi_), om(om_), al(al_) {}

    double operator()(const double& x) const
    {
        double SN = 2 * arma::normpdf(1.0/(1.0-pow(x,2)), xi, sqrt(om)) *
                                            arma::normcdf(al/(1.0-pow(x,2)), xi, sqrt(om));
        double g = log( 2 * arma::normcdf(1.0/(1.0-pow(x,2))) );
        double J = (1+pow(x,2))/pow((1.0-pow(x,2)),2);
        return SN * g * J;
    }
};

double Eint( double xi, double om, double al ) {
    
    SNlog2Phi f(xi, om, al);
    double err_est;
    int err_code;
    double z = integrate(f, -1.0, 1.0, err_est, err_code);

    return z;
}

// [[Rcpp::export]]
double KL(  arma::vec xi_1, 
                        arma::vec xi_2, 
                        arma::mat Omega_1, 
                        arma::mat Omega_2,
                        arma::vec alpha_1,
                        arma::vec alpha_2  )
{
    // Normal part
    int p = xi_1.n_elem;
    double val_1, val_2;
    double sign_1, sign_2;
    
    log_det(val_1, sign_1, Omega_1);
    log_det(val_2, sign_2, Omega_2);
    mat xi(p,1); 
    xi.col(0) = (xi_1 - xi_2);
    mat invOm2 = inv_sympd(Omega_2);
    double DK0 = 0.5 * (trace( invOm2 * Omega_1 ) + as_scalar( xi.t() * invOm2 * xi ) - p + val_2 - val_1 );
    
    // cout << DK0 << endl;
    // Skew part
    vec delta_1 = (Omega_1 * alpha_1) / sqrt(1.0 + as_scalar(alpha_1.t() * Omega_1 * alpha_1));
    vec delta_2 = (Omega_2 * alpha_2) / sqrt(1.0 + as_scalar(alpha_2.t() * Omega_2 * alpha_2));
    
    double xi_W11 = 0;
    double om_W11 = as_scalar( alpha_1.t() * Omega_1 * alpha_1 );
    double al_W11 = as_scalar( alpha_1.t() * delta_1 / sqrt( om_W11 - pow(alpha_1.t() * delta_1, 2)) );
    double E_W11 = Eint(xi_W11, om_W11, al_W11);
    // cout << E_W11 << endl;
    
    double xi_W21 = as_scalar( alpha_2.t() * (xi_1 - xi_2)  );
    double om_W21 = as_scalar( alpha_2.t() * Omega_1 * alpha_2 );
    double al_W21 = as_scalar( alpha_2.t() * delta_1 / sqrt( om_W21 - pow(alpha_2.t() * delta_1, 2)) );
    double E_W21 = Eint(xi_W21, om_W21, al_W21);
    // cout << E_W21 << endl;
    
    double a = as_scalar( sqrt(2/M_PI) * xi.t() * invOm2 * delta_1 );
    
    return (DK0 + a + E_W11 - E_W21);
    
}


double rPG(double b=1.0, double c=0.0) {
    return 0.0;
}

double rPG1z(double z) {
    return 0.0;
}

arma::colvec rpg(arma::colvec shape, arma::colvec scale) {
    // C++-only interface to PolyaGamma class
    // draws random PG variates from arma::vectors of n's and psi's
    RNG r;
    PolyaGamma pg;
// #ifdef USE_R
//   GetRNGstate();
// #endif
    int d = shape.n_elem;
    colvec result(d);
    for(int i=0; i<d; i++) {
        result[i] = pg.draw(shape(i), scale(i), r);
    }
// #ifdef USE_R
//   PutRNGstate();
// #endif
    return result;
}




