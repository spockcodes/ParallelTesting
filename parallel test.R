library(xgboost)

set.seed(222)
N <- 2*10^5
p <- 350
x <- matrix(rnorm(N  * p), ncol = p)
y <- rnorm(N)

system.time(mymodel <- xgboost(
  nthread = 4,
  data = x,
  label = y, 
  nrounds = 5, 
  objective = "reg:linear", 
  tree_method = "exact",
  max_depth = 10,
  min_child_weight = 1, 
  eta = 1, 
  subsample = 0.66, 
  colsample_bytree = 0.33
))




#################################

data(agaricus.train, package='xgboost')
data(agaricus.test, package='xgboost')
train <- agaricus.train
test <- agaricus.test
# fit model
bst <- xgboost(data = train$data, 
               label = train$label, 
               max.depth = 2, 
               eta = 1, 
               nrounds = 2,
               nthread = 1, 
               objective = "binary:logistic"
               )
# predict
pred <- predict(bst, test$data)
