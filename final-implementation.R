dlmcmc_modified <- function(X, Y, newdata, newY, J, tau0, tauz, nepoch) {
  p = ncol(X)
  n = nrow(X)
  
  # Initializations
  B = matrix(0, ncol = p, nrow = p)
  W0 = 1
  b0 = 0
  lambda = rep(0, n * J)
  Z0 = rnorm(n)
  
  # Store error sequence
  error_seq = c()
  
  for(epoch in 1:nepoch) {
    # Update Z using current weights
    Z = SampleZ(Z0, Y, W0, tau0, tauz, J, "gaussian", b0, lambda)
    
    # Update W0 using the current Z values
    res = SampleW0(Y, Z0, "gaussian", J, W0, tau0)
    W0 = res$W0
    b0 = res$b0
    lambda = res$lambda
    
    # Calculate predictions for test data
    yhat = nn_forward(newdata, B)
    error_seq = c(error_seq, mean((yhat - newY)^2))
    
    # B update using SGD - Not present in the provided functions, so here's a basic SGD
    B = sgd_update(X, Z, B, eta = 0.01)
  }
  
  return(list(B = B, yhat = yhat, W0 = W0, b0 = b0, error_seq = error_seq))
}

# Using the function
dat = friedman_dat(100, 5)
result = dlmcmc_modified(dat$X, dat$Y, dat$X, dat$Y, J = 10, tau0 = 1, tauz = 1, nepoch = 10)
