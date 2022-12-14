calibrate <- function(x)
{
  C = x$data$C - 1
  Z = x$chain$t - 1
  if (!is.matrix(Z)) Z=matrix(Z, ncol=1)
  ns  = dim(x$chain$xi0)[3]
  K = dim(x$chain$xi0)[2]

  output = calib(x$data$Y,
                 matrix(C,ncol=1),
                 Z,
                 x$chain$xi, dim(x$chain$xi),
                 x$chain$xi0, dim(x$chain$xi0) )
  colnames(output$Y_cal) = colnames(x$data$Y)
  return(output)

}

calibrateNoDist <- function(x)
{
  C = x$data$C - 1
  Z = x$chain$t - 1
  if (!is.matrix(Z)) Z=matrix(Z, ncol=1)
  ns  = dim(x$chain$xi0)[3]
  K = dim(x$chain$xi0)[2]

  output = calibNoDist(x$data$Y,
                 matrix(C,ncol=1),
                 Z,
                 x$chain$xi, dim(x$chain$xi),
                 x$chain$xi0, dim(x$chain$xi0) )
  colnames(output$Y_cal) = colnames(x$data$Y)
  return(output)

}

relabelChain = function(res) {
  res$chain$t = res$chain$t - 1
  relabeled_chain = relabel(res)
  res$chain = relabeled_chain
  res
}

# Recover stochastic representation parameters from distribution parameters:
transform_params = function(Omega, alpha) {
  n = NROW(Omega)
  m = NCOL(Omega)
  if (n!=m) stop("Omega is not sqaure")
  if (length(alpha)!=n) stop("alpha is of wrong length")
  if (n==1) {
    omega = sqrt(Omega)
  } else {
    omega = sqrt(diag(diag(Omega)))
  }
  omega_inv = solve(omega)
  OmegaBar = omega_inv %*% Omega %*% omega_inv
  alpha = matrix(alpha, ncol=1)
  numer = OmegaBar %*% alpha
  denom = as.numeric(sqrt(1 + t(alpha) %*% OmegaBar %*% alpha))
  delta = numer/denom
  if (n==1) {
    psi = sqrt(Omega) * delta
  } else {
    psi = sqrt(diag(Omega)) * delta
  }
  alpha.tilde = delta / sqrt(1-delta^2)
  if (n==1) {
    inv.Delta = 1/c(sqrt(1-delta^2))
  } else {
    inv.Delta = diag(1/c(sqrt(1-delta^2)))
  }
  Omega.epsilon = inv.Delta %*% OmegaBar %*% inv.Delta - alpha.tilde%*%t(alpha.tilde)
  return(list(delta=delta, omega=omega, Omega.epsilon=Omega.epsilon,
              psi=psi, Sigma=(Omega - psi%*%t(psi))))
}


Mode = function(x) {
  ux = unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

summarizeChain = function( res ) {
  chainSummary = list()
  K = res$prior$K
  chain = res$chain
  p = ncol(res$data$Y)
  J = length(unique(res$data$C))
  ns = res$pmc$nsave

  xi_raw = rowMeans(chain$xi[, , 1:ns], dims=2)
  chainSummary$xi0 = rowMeans(chain$xi0[, , 1:ns], dims=2)
  chainSummary$psi = rowMeans(chain$psi[, , 1:ns], dims=2)
  Omega_raw = rowMeans(chain$Omega[, , 1:ns], dims=2)
  Sigma_raw = rowMeans(chain$G[, , 1:ns], dims=2)
  E_raw = rowMeans(chain$E[, , 1:ns], dims=2)
  chainSummary$alpha = rowMeans(chain$alpha[, , 1:ns], dims=2)
  chainSummary$W = rowMeans(chain$W[, , 1:ns], dims=2)
  # tmp <- array(c(1:9, 1:9, 9:1), c(3,3,3))
  # tmp
  # rowSums(tmp, dims=2)
  chainSummary$xi = array(0, dim=c(J,p,K))
  chainSummary$Omega = array(0, dim=c(p,p,K))
  chainSummary$Sigma = array(0, dim=c(p,p,K))
  chainSummary$E = array(0, dim=c(p,p,K))
  for (k in 1:K) {
    indsk = (1+p*(k-1)):(p*k)
    chainSummary$xi[, , k] = xi_raw[, indsk]
    chainSummary$Omega[, , k] = Omega_raw[, indsk]
    chainSummary$Sigma[, , k] = Sigma_raw[, indsk]
    chainSummary$E[, , k] = E_raw[, indsk]
  }



  chainSummary$meanvec = array(0, c(J, p, K))
  chainSummary$meanvec0 = matrix(0, p, K)
  for (k in 1:K) {
    del.om = transform_params(chainSummary$Omega[, , k], chainSummary$alpha[, k])
    # chainSummary$psi[,k] = del.om$psi
    chainSummary$Sigma[,,k] = del.om$Sigma
    toadd = del.om$omega %*% del.om$delta*sqrt(2/pi)
    for (j in 1:J) {
      chainSummary$meanvec[j, , k] = chainSummary$xi[j, , k] + toadd
    }
    chainSummary$meanvec0[, k] = chainSummary$xi0[, k] + toadd
  }

  chainSummary$t = apply(chain$t, 2, Mode)
  # chainSummary$a0 = mean(chain$a0)

  chainSummary
}

