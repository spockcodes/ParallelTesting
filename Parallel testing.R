library(doParallel)
library(xgboost)

#cluster
cl <- makeCluster(4, setup_timeout = 0.5)
registerDoParallel(cl)

#core
registerDoParallel(cores = 6)

#trivial example
foreach(i=1:3) %dopar% sqrt(i)

x <- iris[which(iris[,5] != "setosa"), c(1,5)]
trials <- 10000

#sequential
ptime.s <- system.time({
    r <- foreach(icount(trials), .combine=cbind) %do% {
      ind <- sample(100, 100, replace=TRUE)
      result1 <- glm(x[ind,2]~x[ind,1], family=binomial(logit))
      coefficients(result1)
    }
  })

ptime.s

#parallel
ptime.p <- system.time({
  r <- foreach(icount(trials), .combine=cbind) %dopar% {
    ind <- sample(100, 100, replace=TRUE)
    result1 <- glm(x[ind,2]~x[ind,1], family=binomial(logit))
    coefficients(result1)
  }
})

stopCluster(cl)


ptime.p

boost.time <- system.time(xgb.fit1 <- xgb.cv(
  data = features_train_dq,
  label = response_train_dq,
  nrounds = 100,
  nfold = 10,
  #nthreads = 4, #increase multi-threading
  objective = "count:poisson",
  base_score = mean(response_train_dq),  
  verbose = 0,               # silent,
  early_stopping_rounds = 10 # stop if no improvement for 10 consecutive trees
)
)

boost.time
