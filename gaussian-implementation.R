friedman_dat <- function(n, p, sigma=1, seed=1000){
  set.seed(seed)
  X = matrix(runif(n * p), ncol = p)
  Y = 10 * sin(pi* X[ ,1] * X[,2]) + 20 * (X[,3] -.5)^2 + 10 * X[ ,4] + 5 * X[,5] + rnorm(n,sd=sigma)
  return(list(Y=Y, X=X))
}

J_seq <- c(2,5,10)
p_seq <- c(10,50,100,1000)
n <- 100 # no of observations
learning_rate <- 0.01
batch_size <- 32
T <- 100
tau0 <- 1.0 # eg value
tauz <- 1.0 # eg value
nhid <- 10 # eg value
delta <- 0.5

# iterate over J and p
for(p in p_seq){
  # generate the dataset
  data <- friedman_dat(n, p)
  X <- data$X
  Y <- data$Y
  
  for(J in J_seq){
    # Step1: Initialize
    W0 <- 1.0 # eg initialization
    b0 <- 0.0
    Z0 <- rnorm(n*J)
    
    for(t in 1:T){
      # Step2: Update the weights in the top layer
      results <- SampleW0_mean(Y, Z0, family="gaussian", W0=W0, tau0=tau0)
      W0 <- results$W0
      b0 <- results$b0
      
      # Step3: Update the deep learner with SGD step
      model <- dnn_model(p=p, nhid=nhid, L=1, delta=delta, family="gaussian")
      model %>% fit(X, Z0, epochs=1, batch_size=batch_size)
      
      # Update B
      # simply extracting the weights
      B <- model$get_weights()[[1]]
      
      # Step4: Update Z0
      Z0 <- SampleZ(Z=Z0, Y=Y, W0=W0, tau0=tau0, tauz=tauz, J=J, family="gaussian", b0=b0, lambda=NULL)
    }
    
    # final prediction
    y_hat <- model %>% predict(X) * W0 + b0
    
    # eval
    mse_val <- mse(Y, y_hat)
    cat(sprintf("For p=%d and J=%d, MSE: %f\n", p, J, mse_val))
  }
}
