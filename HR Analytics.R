#Get working directory
getwd()

#Import hr analytics data set
hr_data <- read.csv("hr_train1.csv", header = TRUE, sep = ",")

#split into testing and training dataset
subset <-  sample(nrow(hr_data), nrow(hr_data) * 0.8)
hr_train = hr_data[subset,]
hr_test = hr_data[-subset,]

#Get summary of training dataset for exploratory analysis
summary(hr_train)

#Train a logistic regression model w/ all variables
hr_glm0 <- glm(target~., family=binomial, data=hr_train)
summary(hr_glm0)
AIC(hr_glm0)
BIC(hr_glm0)

#Prediction
hist(predict(hr_glm0))
pred_resp <- predict(hr_glm0, type="response")
hist(pred_resp)

#Prediction of model and Misclassification rate
table(hr_train$target, (pred_resp > 0.5)*1, dnn=c("Truth", "Predicted"))

#In-sample Prediction
library(ROCR)
pred_glm0_train <- predict(hr_glm0, type = "response")
pred <- prediction(pred_glm0_train, hr_train$target)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)
unlist(slot(performance(pred, "auc"), "y.values"))

#Out-of-sample Prediction
pred_glm0_test <- predict(hr_glm0, newdata = hr_test, type="response")
pred <- prediction(pred_glm0_test, hr_test$target)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)
unlist(slot(performance(pred, "auc"), "y.values"))

#Variable selection with backwards BIC approach
hr_glm_back_BIC <- step(hr_glm0, k=log(nrow(hr_train)))
AIC(hr_glm_back_BIC)
BIC(hr_glm_back_BIC)

#variable selection with backwards AIC approach
hr_glm_back <- step(hr_glm0)
AIC(hr_glm_back)
BIC(hr_glm_back)


#Random Forest
library(randomForest)
hr_train$target <- as.character(hr_train$target)
hr_train$target <- as.factor(hr_train$target)
rf_classifier<- randomForest(target~., data=hr_train, ntree=500, mtry=2, importance=TRUE)
varImpPlot(rf_classifier)
rf_classifier