predict <- function(X, W, b){
  return(X %*% W + b)
}

mse_loss <- function(Y_true, Y_pred){
  return(mean((Y_true - Y_pred)^2))
}

compute_gradients <- function(X, Y_true, Y_pred){
  n <- dim(X)[1]
  dW <- -2/n * t(X) %*% (Y_true - Y_pred)
  db <- -2 * mean(Y_true - Y_pred)
  return(list(dW=dW, db=db))
}

# SGD function
sgd_step <- function(X, Y, W, b, learning_rate=0.01){
  # Randomly sample a mini-batch
  indices <- sample(1:nrow(X), batch_size)
  X_batch <- X[indices, , drop=FALSE]
  Y_batch <- Y[indices]
  
  # compute predictions for it
  Y_pred <- predict(X_batch, W, b)
  
  # compute gradient
  grads <- compute_gradients(X_batch, Y_batch, Y_pred)
  
  # update weights,bias
  W <- W - learning_rate * grads$dW
  b <- b - learning_rate * grads$db
  
  return(list(W=W, b=b))
}

p <- 10
nhid <- 10
batch_size <- 32
learning_rate <- 0.01

# initialize 
W <- matrix(rnorm(p * nhid), ncol=nhid)
b <- rep(0, nhid)

# create a dataset
data <- friedman_dat(1000, p)
X <- data$X
Y <- data$Y

# sgd for few epocs
for(epoch in 1:100){
  for(i in 1:(nrow(X)/batch_size)){
    results <- sgd_step(X, Y, W, b, learning_rate)
    W <- results$W
    b <- results$b
  }
  Y_pred <- predict(X, W, b)
  cat(sprintf("Epoch: %d, Loss: %f\n", epoch, mse_loss(Y, Y_pred)))
}

B <- W # post training, W itself will be the B matrix
