# library(reticulate)
# use_python("/apps/python/3.6/3.6.3/bin/python3")
# py_config()

# comparison of different J's
source("dlmcmc.R")
library(sfsmisc)
#library(ggplot2)
library(reshape2)
library(parallel)
library(doParallel)


cl <- makeCluster(1)
registerDoParallel(cl)

J_seq=c(2,5,10)

n=100
p_seq=c(10,50,100,1000)

data_type="friedman"
optimizer="adam"
# a regression example

friedman_dat<-function(n,p,sigma=1,seed=1000){
  set.seed(seed)
  X = matrix(runif(n * p), ncol = p)
  Y = 10 * sin(pi* X[ ,1] * X[,2]) +20 * (X[,3] -.5)^2 + 10 * X[ ,4] + 5 * X[,5] + rnorm(n,sd=sigma)
  return(list(Y=Y,X=X))
}

doRegression<-function(seed,L=1,delta=0.5,tau0=1,tauz=1,optimizer="adam"){
  set.seed(seed)
  n=100
  p_seq=c(10,50,100,1000)
  J_seq=c(2,5,10)
  plot_mat_all=NULL
  for(i in 1:length(p_seq)){
    p=p_seq[i]
    dat=friedman_dat(n,p,seed=seed)
    # 70% training and 30% testing
    idx.train=sample(1:n,size=round(0.7*n))
    X.train=dat$X[idx.train,]
    Y.train=dat$Y[idx.train]
    X.test=dat$X[-idx.train,]
    Y.test=dat$Y[-idx.train]
    
    mcmc_mat<-NULL
    for(j in 1:length(J_seq)){
      J=J_seq[j]
      mcmc_fit<-dlmcmc(X.train,Y.train,newdata = X.test,newY=Y.test,family="gaussian",
                       tau0=tau0,tauz=tauz,J=J,nepoch=10,L=L,delta=delta,optimizer=optimizer)
      mcmc_mat=rbind(mcmc_mat,data.frame(time_seq=cumsum(mcmc_fit$time_seq),error_seq=mcmc_fit$error_seq, J=J))
    }
    
    nn_fit<-nn_res(X.train,Y.train,X.test,Y.test,family="gaussian",L=L,delta=delta,optimizer=optimizer)
    
    plot_mat=mcmc_mat
    colnames(plot_mat)[3]<-"method"
    plot_mat$method<-as.factor(plot_mat$method)
    levels(plot_mat$method)<-c("SDA-GR(J=2)","SDA-GR(J=5)","SDA-GR(J=10)")
    
    plot_mat=rbind(plot_mat,data.frame(time_seq=cumsum(nn_fit$time_seq),error_seq=nn_fit$error_seq,method="DL"))
    colnames(plot_mat)[1:2]<-c("time","mse")
    plot_mat$epoch=rep(0:10,4)
    
    plot_mat$dim=p
    
    plot_mat_all=rbind(plot_mat_all,plot_mat)
  }
  return(plot_mat_all)
}

result_1<-mclapply(1:50,doRegression,L=1,tau0 =0.1,tauz = 1,optimizer=optimizer,mc.cores=1,mc.preschedule=FALSE)
result_2<-mclapply(1:50,doRegression,L=2,tau0 =0.1,tauz = 1,optimizer=optimizer,mc.cores=1,mc.preschedule=FALSE)
stopCluster(cl)

# save.image(paste0(ra"friedman_1000_",optimizer,".Rdata"))

