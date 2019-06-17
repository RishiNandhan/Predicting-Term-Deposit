#install the necessary packages
install.packages("plyr")
library(plyr)
install.packages("tidyverse")
library(tidyverse)
install.packages("reshape") #used to reshape variables into heatmap
library(reshape)
install.packages("GGally") #extension of ggplot2 package
library(GGally)
library(MASS)
library(leaps)
library(randomForest)
library(caret)
library(class)
#set working directory
setwd("E://")
#load the data
bank<-read.csv("bank.csv")
#check the dimension of data
dim(bank)
#structure of the data
str(bank)
#summary of the data
summary(bank)
#check for NA values
sum(is.na(bank))
#check for duplicate entries
sum(duplicated(bank,incomparables = F))
#check for outliers
sum(bank$balance<0)
#since 688 observations takes only around 6 percent of our data we can remove those observations.
bank.df<-subset(bank,bank$balance>=0)
summary(bank.df)
#normalize the data
bank.df[c(1,6,10,12,13,14,15)]<-scale(bank.df[c(1,6,10,12,13,14,15)])
summary(bank.df)
#check for corelation of variables plot inform of heatmap
cor.mat<-round(cor(bank.df[c(1,6,12,13,14,15)]),2) 
cor.mat
melted.cor.mat<-melt(cor.mat)
ggplot(melted.cor.mat,aes(x=X1, y=X2, fill=value))+
  geom_tile()+
  geom_text(aes(x=X1, y=X2, label=value))

#spliting the data into training and testing data
set.seed(21)
ind<-sample(2, nrow(bank.df),replace = T, prob = c(0.8,0.2))
train<-bank.df[ind==1,]
test<-bank.df[ind==2,]
#check if we need to do oversampling
table(train$deposit)

##########################################
#Knn model
#for knn all variables should be numeric
#convert all factor variables to numeirc
train.knn<-train
test.knn<-test
str(train.knn)
#convert the training factor variables to numeric 
train.knn$job<-as.numeric(train.knn$job)
train.knn$marital<-as.numeric(train.knn$marital)
train.knn$education<-as.numeric(train.knn$education)
train.knn$default<-as.numeric(train.knn$default)
train.knn$housing<-as.numeric(train.knn$housing)
train.knn$loan<-as.numeric(train.knn$loan)
train.knn$contact<-as.numeric(train.knn$contact)
train.knn$month<-as.numeric(train.knn$month)
train.knn$poutcome<-as.numeric(train.knn$poutcome)
train.knn$deposit<-as.numeric(train.knn$deposit)

#convert the test factor variables to numeric
test.knn$job<-as.numeric(test.knn$job)
test.knn$marital<-as.numeric(test.knn$marital)
test.knn$education<-as.numeric(test.knn$education)
test.knn$default<-as.numeric(test.knn$default)
test.knn$housing<-as.numeric(test.knn$housing)
test.knn$loan<-as.numeric(test.knn$loan)
test.knn$contact<-as.numeric(test.knn$contact)
test.knn$month<-as.numeric(test.knn$month)
test.knn$poutcome<-as.numeric(test.knn$poutcome)
test.knn$deposit<-as.numeric(test.knn$deposit)

str(train.knn)
#set seed and implement knn
set.seed(25)
# 1st model where K=1
knn.pred1<-knn(train.knn,test.knn,train.knn$deposit,k=1)
table1<-table(knn.pred1,test.knn$deposit)
table1
knn_accuracy_1<-sum(diag(table1))/sum(table1)
knn_accuracy_1

#2nd model where K=10
knn.pred2<-knn(train.knn,test.knn,train.knn$deposit,k=10)
table2<-table(knn.pred2,test.knn$deposit)
table2
knn_accuracy_2<-sum(diag(table2))/sum(table2)
knn_accuracy_2

#3rd model where K=100
knn.pred3<-knn(train.knn,test.knn,train.knn$deposit,k=100)
table3<-table(knn.pred3,test.knn$deposit)
table3
knn_accuracy_3<-sum(diag(table3))/sum(table3)
knn_accuracy_3

############################################################
#selection of variables
# Using exhaustive search
regfit.full<-regsubsets(deposit~., data = train, nvmax = 16,method = "exhaustive")
regfit.summary<-summary(regfit.full)
names(regfit.summary)

#plotting adjusted R2 values and selecting number of variables.
regfit.summary$adjr2
max.adjr2 <- max(regfit.summary$adjr2)
std.adjr2 <- sd(regfit.summary$adjr2)
plot(regfit.summary$adjr2 ,xlab =" Number of Variables ",ylab=" Adjusted RSq",type="l",main = "Adjusted Rsq values for different variables")
abline(h = max.adjr2 + 0.2 * std.adjr2, col = "red", lty = 1)
abline(h = max.adjr2 - 0.2 * std.adjr2, col = "red", lty = 1)
#plotting cp values and selecting number of variables.
regfit.summary$cp
min.cp <- min(regfit.summary$cp)
std.cp <- sd(regfit.summary$cp)
plot(regfit.summary$cp ,xlab =" Number of Variables ",ylab=" cp",type="l",main = "CP values for different variables")
abline(h = min.cp + 0.2 * std.cp, col = "red", lty = 1)
abline(h = min.cp - 0.2 * std.cp, col = "red", lty = 1)
#plotting bic values and selecting number of variables
regfit.summary$bic
min.bic<-min(regfit.summary$bic)
std.bic<-min(regfit.summary$bic)
plot(regfit.summary$bic, xlab="Number of Variables",ylab="bic",type="l",main = "BIC values for different variables")
abline(h=min.bic + 0.2*std.bic, col="red", lty=1)
abline(h=min.bic - 0.2*std.bic, col="red",lty=1)

#selection of variables from bic
regfit.full<-regsubsets(deposit~.,data = bank.df, nvmax = 16, method = "exhaustive")
coeffs_bic<-coef(regfit.full, id=4)
names(coeffs_bic)

#selection of variables from cp
regfit.full<-regsubsets(deposit~.,data = bank.df, nvmax = 16, method = "exhaustive")
coeffs_cp<-coef(regfit.full, id=11)
names(coeffs_cp)

#model using all variable
glm1<-glm(formula = deposit~.,data = train,family = "binomial")
summary(glm1)

#prediction using training data
p1<-predict(glm1,train,type = "response")
pred1<-ifelse(p1>0.5,1,0)
tab1<-table(predicted=pred1,actual=train$deposit)
accuracy_logit_train_1<-sum(diag(tab1))/sum(tab1)
accuracy_logit_train_1
#predicition using test data
p11<-predict(glm1,test,type = "response")
pred11<-ifelse(p11>0.5,1,0)
tab11<-table(predicted=pred11,actual=test$deposit)
accuracy_logit_test_1<-sum(diag(tab11))/sum(tab11)
accuracy_logit_test_1

# models using cp variables
glm2<-glm(formula = deposit~ housing + loan + contact + month + duration + poutcome,data = train,family = "binomial")
summary(glm2)

#prediction using training data
p2<-predict(glm2,train,type = "response")
pred2<-ifelse(p2>0.5,1,0)
tab2<-table(predicted=pred2,actual=train$deposit)
accuracy_logit_train_2<-sum(diag(tab2))/sum(tab2)
accuracy_logit_train_2

p22<-predict(glm2,test,type = "response")
pred22<-ifelse(p22>0.5,1,0)
tab22<-table(predicted=pred22,actual=test$deposit)
accuracy_logit_test_2<-sum(diag(tab22))/sum(tab22)
accuracy_logit_test_2

#model using bic variables
glm3<-glm(formula = deposit~ housing+contact+duration+poutcome,data=train,family = "binomial")
p3<-predict(glm3,train,type = "response")
pred3<-ifelse(p3>0.5,1,0)
tab3<-table(predicted=pred3,actual=train$deposit)
accuracy_logit_train_3<-sum(diag(tab3))/sum(tab3)
accuracy_logit_train_3

p33<-predict(glm3,test,type = "response")
pred33<-ifelse(p33>0.5,1,0)
tab33<-table(predicted=pred33,actual=test$deposit)
accuracy_logit_test_3<-sum(diag(tab33))/sum(tab33)
accuracy_logit_test_3

####################################################3
#Random Forest
set.seed(876)
#random forest model with all varaiables
rf1<-randomForest(deposit~., data = train,
                  importance=T, proximity=T)
rf1
#predict with training data and plot confusion matrix
pp1<-predict(rf1, train)
confusionMatrix(pp1, train$deposit)
#predict with testing data and plot confusion matrix
pp11<-predict(rf1,test)
confusionMatrix(pp11,test$deposit)
#importance of each predictors
varImpPlot(rf1)
#random forest model with bic variables
rf2<-randomForest(deposit~housing+contact+duration+poutcome, data = train,
                 importance=T, proximity=T)
rf2
#predict with training data and plot confusion matrix
pp2<-predict(rf2, train)
confusionMatrix(pp2, train$deposit)
#predict with testing data and plot confusion matrix
pp22<-predict(rf2,test)
confusionMatrix(pp22,test$deposit)
# importance of each predictors
varImpPlot(rf2)
###########################################################

