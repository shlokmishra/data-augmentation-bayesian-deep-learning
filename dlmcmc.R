# dl-mcmc

SampleZ<-function(Z,Y,W0,tau0,tauz,J,
                  family,
                  b0,
                  # lambda is (n*J)*1 vector
                  lambda
                  ){
  
  # add the cases when Y is a matrix
  if(is.vector(Y)){
    n=length(Y)
  }else{
    n=nrow(Y)
  }
  
  
  if(family=="gaussian"){
    # whether Y is multivariate
    if(is.vector(Y)){
      # Y is univariate
      mu=((W0)*(tauz^2)*(Y-b0)+tau0^2*Z)/(W0^2*tauz^2+tau0^2)
      sigma=(tau0*tauz)/sqrt(W0^2*tauz^2+tau0^2)
    }else{
      # Y is multivariate
      mu=((tauz^2)*(sweep(Y,1,b0))%*%W0+tau0^2*Z)/(sum(W0^2)*tauz^2+tau0^2)
      sigma=(tau0*tauz)/sqrt(sum(W0^2)*tauz^2+tau0^2)
    }
    #stacked new J
    newZ0=rnorm(J*n,mean=mu,sd=sigma)
  }
  else if(family=="binomial"){
    # if it is binomial, b0=NULL,need lambda
    Ys=rep(Y,J)
    Zs=rep(Z,J)
    mu=(W0*tauz^2*Ys+tau0^2*Zs*lambda)/(W0^2*tauz^2+tau0^2*lambda)
    sigma=(tauz*tau0*sqrt(lambda))/sqrt(tau0^2*tauz^2+tau0^2*lambda)
    newZ0=rnorm(J*n,mean=mu,sd=sigma)
  }
  return(newZ0)
}

SampleZ_mean<-function(Z,Y,W0,tau0,tauz,
                  family,
                  b0,
                  # lambda is (n*J)*1 vector
                  lambda
){
  # add the cases when Y is a matrix
  if(is.vector(Y)){
    n=length(Y)
  }else{
    n=nrow(Y)
  }
  
  if(family=="gaussian"){
    mu=((W0^2)*(tauz^2)*(Y-b0)+tau0^2*Z)/(W0^2*tauz^2+tau0^2)
  }
  else if(family=="binomial"){
    # if it is binomial, b0=NULL,need lambda
    Ys=Y
    Zs=Z
    mu=(W0*tauz^2*Ys+tau0^2*Zs*lambda)/(W0^2*tauz^2+tau0^2*lambda)
  }
  return(mu)
}

SampleW0<-function(Y,Z0,
                   family,
                   J,
                   W0,
                   tau0){
  
  if(is.vector(Y)){
    n=length(Y)
    Ys=rep(Y,J)
  }else{
    n=nrow(Y)
    Ys=Y[rep(1:n,J),]
  }
  
  if(family=="gaussian"){
    if(is.vector(Y)){
      fit=lm(Ys~Z0)
      b0=fit$coefficients[1]
      W0=fit$coefficients[2]
    }else{
      fit=apply(Ys,2,function(y) lm(y~Z0))
      b0=sapply(fit,function(x) x$coefficients[1])
      W0=sapply(fit,function(x) x$coefficients[2])
    }
  
    lambda=NULL
  }else if(family=="binomial"){
    Z0s=Z0
    require(statmod)
    lambda=1/rinvgauss(n*J,mean=1/abs(1-Ys*Z0s*W0),shape=1/(tau0^2))
    sigma2=1/sum(Ys^2*Z0s^2/lambda)
    mu=sum(Ys*Z0s*(1+lambda)/lambda)*sigma2
    W0=rnorm(1,mean=mu,sd=sqrt(sigma2))
    b0=NULL
  }
  return(list(W0=W0, b0=b0, lambda=lambda))
}

SampleW0_mean<-function(Y,Z0,
                   family,
                   W0,
                   tau0){
  n=length(Y)
  Ys=Y
  if(family=="gaussian"){
    fit=lm(Ys~Z0)
    b0=fit$coefficients[1]
    W0=fit$coefficients[2]
    lambda=NULL
  }else if(family=="binomial"){
    Z0s=Z0
    require(statmod)
    lambda=abs(1-Ys*Z0s*W0)
    sigma2=1/(tau0^2*sum(Ys^2*Z0s^2/lambda))
    mu=sum(Ys*Z0s*(1+lambda)/lambda)*sigma2
    if(is.na(mu[1])){
      print("change")
      lambda=rep(0.05,n)
      sigma2=1/(tau0^2*sum(Ys^2*Z0s^2/lambda))
      mu=sum(Ys*Z0s*(1+lambda)/lambda)*sigma2
    }else{
      W0=mu
    }
    b0=NULL
  }
  return(list(W0=W0, b0=b0, lambda=lambda))
}

dnn_model<-function(p,nhid,L=1,delta=0.5,family,optimizer="rmsprop"){
  model<-keras_model_sequential()
  if(L!=1 && length(nhid)==1){
    nhid=rep(nhid,L)
  }
  if(L!=1 && length(delta)==1){
    delta=rep(delta,L)
  }
  for(ll in 1:L){
    if(family=="gaussian"){
      if(ll==1){
        model%>%layer_dense(units=nhid[ll],activation='relu',input_shape=p) 
      }else{
        model%>%layer_dense(units=nhid[ll],activation='relu') 
      }
    }else if(family=="binomial"){
      if(ll==1){
        model%>%layer_dense(units=nhid[ll],activation='relu',input_shape=p) 
      }else{
        model%>%layer_dense(units=nhid[ll],activation='relu') 
      }
    }
     model%>%layer_dropout(rate=delta[ll]) 
  }
  model%>%layer_dense(units=1)
  
  model %>% compile(
    loss="mse",
    optimizer=optimizer,
    metrics=list("mean_absolute_error")
  )
  
  model
}



mse=function(x,y){
  return(mean((x-y)^2))
}

dlmcmc<-function(X,
                 Y,
                 newdata=NULL,
                 newY=NULL,
                 # "gaussian" for regression model
                 # "binomial" for classification model
                 family=c("gaussian","binomial"),
                 tau0=1e-02,
                 tauz=1e-02,
                 J=10,
                 # number of hidden variables in Keras
                 nhid=64,
                 # number of layers
                 L=1,
                 # dropout rate
                 delta=0.5,
                 nepoch=10,
                 optimizer="rmsprop"){
  require(keras)
  require(statmod)
  require(tictoc)
  
  if(is.vector(Y)){
    n=length(Y)
    q=1
  }else{
    n=nrow(Y)
    q=ncol(Y)
  }

  p=ncol(X)
  Xs=X[rep(1:n,J),]
  W0=rep(1,q)
  b0=rep(0,q)
  lambda=rep(0,n*J)
  
  dnn<-dnn_model(p,nhid,L,delta,family=family,optimizer=optimizer)



  time_seq=0
  error_seq=c()
  # add the initialization error
  
  if(is.null(newdata)){
    newdata=X
    newY=Y
  }
  yhat=dnn%>%predict(newdata)
  yhat=yhat%*%W0
  
  # storing all the yhat
  yhat_list=list()
  
  if(family=="gaussian"){
    if(q==1){
      yhat=yhat+b0
    }else{
      yhat=sweep(yhat,1,b0,"+")
    }
    yhat_list[[1]]=yhat

    error_seq=c(error_seq,mse(yhat,newY))
  }else{
    yhat=ifelse(yhat>0,1,-1)
    yhat=as.vector(yhat)
    error_seq=c(error_seq,mean(yhat!=newY))
    yhat_list[[1]]=yhat
  }

  
  for(i in 1:nepoch){
    tic()
    print(paste0("epoch: ",i))
    Z0hat=dnn%>%predict(X)
    Z0=SampleZ(Z0hat,Y,
               W0=W0,
               tau0 = tau0,
               tauz = tauz,
               J=J,
               family=family,
               b0=b0,
               lambda=lambda)
    W0_fit=SampleW0(Y,Z0,family=family,J=J,W0=W0,tau0=tau0)
    W0=W0_fit$W0
    b0=W0_fit$b0
    lambda=W0_fit$lambda
    dnn %>% fit(
      Xs,
      Z0,
      epochs=1,
      batch_size = 1280,
      validation_split=0.2
    )
    tt=toc()
    time_seq=c(time_seq,(tt$toc-tt$tic))

    if(is.null(newdata)){
      newdata=X
      newY=Y
      }
    yhat=dnn%>%predict(newdata)
    yhat=yhat%*%W0
    if(family=="gaussian"){
      if(q==1){
        yhat=yhat+b0
      }else{
        yhat=sweep(yhat,1,b0,"+")
      }
      yhat_list[[i+1]]=yhat
      error_seq=c(error_seq,mse(yhat,newY))
    }else{
      yhat=ifelse(yhat>0,1,-1)
      yhat=as.vector(yhat)
      yhat_list[[i+1]]=yhat
      error_seq=c(error_seq,mean(yhat!=newY))
    }
    
    }

  return(list(W0=W0,b0=b0,dnn=dnn,yhat=yhat,yhat_list=yhat_list,time_seq=time_seq,error_seq=error_seq))
}

# nn for regression
reg_nn<-function(p,nhid,L=1,delta=0.5,optimizer="rmsprop"){
  if(L!=1 && length(nhid)==1){
    nhid=rep(nhid,L)
  }
  if(L!=1 && length(delta)==1){
    delta=rep(delta,L)
  }
  model<-keras_model_sequential() 
  for(ll in 1:L){
    if(ll==1){
      model%>%layer_dense(units=nhid[ll],activation='relu',input_shape=p) 
    }else{
      model%>%layer_dense(units=nhid[ll],activation='relu') 
    }
      model%>%layer_dropout(rate=delta[ll]) 
  }
  model%>%layer_dense(units=1)
  
  
  model %>% compile(
    loss="mse",
    optimizer=optimizer,
    metrics=list("mean_absolute_error")
  )
  
  model
}

# nn for binary classification
bin_nn<-function(p,nhid,L=1,delta=0.5,optimizer="rmsprop"){
  model<-keras_model_sequential() 
  if(L!=1 && length(nhid)==1){
    nhid=rep(nhid,L)
  }
  if(L!=1 && length(delta)==1){
    delta=rep(delta,L)
  }
  for(ll in 1:L){
    if(ll==1){
      model%>%layer_dense(units=nhid[ll],activation='relu',input_shape=p) 
    }else{
      model%>%layer_dense(units=nhid[ll],activation='relu') 
    }
    model%>%layer_dropout(rate=delta[ll]) 
  }
  model%>%layer_dense(units=1,activation="sigmoid")
  
  model %>% compile(
    loss="binary_crossentropy",
    optimizer=optimizer,
    metrics=c("accuracy")
  )
  model
}

nn_res=function(X.train,Y.train,X.test,Y.test,
                family,
                nhid=64,
                nepoch=10,
                L=1,
                delta=0.5,
                optimizer="rmsprop"){
  p=ncol(X.train)
  require(tictoc)
  require(keras)
  error_seq=c()
  time_seq=c()
  if(family=="gaussian"){
    dnn=reg_nn(p,nhid,L,delta,optimizer=optimizer)
    # add initialization
    time_seq=0
    yhat=dnn%>%predict(X.test)
    error_seq=c(error_seq,mse(yhat,Y.test))
    
    for(k in 1:nepoch){
      tic()
      dnn%>% fit(
        X.train,
        Y.train,
        epochs=1,
        batch_size = 1280,
        validation_split=0.2
      )
      tt=toc()
      time_seq=c(time_seq,(tt$toc-tt$tic))
      yhat=dnn%>%predict(X.test)
      error_seq=c(error_seq,mse(yhat,Y.test))
    }
  }else if(family=="binomial"){
    # force the label to be 0 and 1
    # Y.train=to_categorical(ifelse(Y.train>0,0,1),num_classes = 2)
    # Y.test=to_categorical(ifelse(Y.test>0,0,1),num_classes = 2)
    Y.train=ifelse(Y.train>0,0,1)
    Y.test=ifelse(Y.test>0,0,1)
    dnn=bin_nn(p,nhid,L,delta,optimizer=optimizer)
    
    # add initialization
    time_seq=0
    score=dnn%>%evaluate(X.test,Y.test)
    error_seq=c(error_seq,(1-score[2]))
    
    for(k in 1:nepoch){
      tic()
      dnn%>% fit(
        X.train,
        Y.train,
        epochs=1,
        batch_size = 1280,
        validation_split=0.2
      )
      tt=toc()
      time_seq=c(time_seq,(tt$toc-tt$tic))
      score=dnn%>%evaluate(X.test,Y.test)
      error_seq=c(error_seq,(1-score[2]))
    }
    
  }
  return(list(time_seq=time_seq,error_seq=error_seq, dnn=dnn))
}

