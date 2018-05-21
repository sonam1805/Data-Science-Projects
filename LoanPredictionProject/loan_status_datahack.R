train_loan_data=read.csv(file.choose())
test_loan_data=read.csv(file.choose())
summary(train_loan_data)
summary(test_loan_data)
####combine test and train data for data cleaning
test_loan_data$Loan_Status=NA
test_loan_data$IS_trainset=FALSE
train_loan_data$IS_trainset=TRUE
loan_full_data=rbind(train_loan_data,test_loan_data)
dim(loan_full_data)
head(loan_full_data)
#Data Cleaning
library(ggplot2)
ggplot(data=loan_full_data,aes(x=Gender,fill=Married))+geom_bar(position="dodge")
loan_full_data[loan_full_data$Gender=='',"Gender"]<-'Male'


ggplot(data=loan_full_data,aes(x=loan_full_data$LoanAmount,col="red"))+geom_histogram()
loan_full_data[is.na(loan_full_data$LoanAmount),"LoanAmount"]<-126


ggplot(loan_full_data,aes(x=Credit_History))+geom_bar()
loan_full_data[is.na(loan_full_data$Credit_History),"Credit_History"]<-1

ggplot(loan_full_data,aes(x=Self_Employed,fill=Education))+geom_bar(position = "dodge")
loan_full_data[loan_full_data$Self_Employed=='',"Self_Employed"]<-'No'

ggplot(loan_full_data,aes(x=Dependents,fill=Married))+geom_bar(position = "dodge")
loan_full_data[loan_full_data$Dependents=='',"Dependents"]<-0


ggplot(loan_full_data,aes(x=Married,fill=Gender))+geom_bar(position = "dodge")
loan_full_data[loan_full_data$Married=='',"Married"]<-'Yes'



ggplot(loan_full_data,aes(x=Loan_Amount_Term))+geom_histogram()
loan_full_data[is.na(loan_full_data$Loan_Amount_Term),"Loan_Amount_Term"]<-360
library(car)
loan_full_data$Loan_Amount_Term<-recode(loan_full_data$Loan_Amount_Term,"'350'='360';'6'='60'")

library(plyr)
loan_full_data<-mutate(loan_full_data,TotalIncome=ApplicantIncome+CoapplicantIncome)
loan_full_data$LoanAmountByTotIncome<-loan_full_data$LoanAmount/loan_full_data$TotalIncome
summary(loan_full_data)

##split the traning and test data
loan.train=loan_full_data[loan_full_data$IS_trainset=="TRUE",]
summary(loan.train)
loan.test=loan_full_data[loan_full_data$IS_trainset=="FALSE",]
summary(loan.test)


##########################rpart model
###Rec_model <- rpart(Loan_Status~ Credit_History + Property_Area + Education + LoanAmount, loan.train, method = "class")
library(rpart)
dim(loan.train)
614*0.7
train_mod=loan.train[1:429,]
test_mod=loan.train[429:614,]
Rec_model <- rpart(Loan_Status~., train_mod, method = "class")
pred <- predict(Rec_model,test_mod, type = "class")


##validation to testdata
conf <- table(test_mod$Loan_Status, pred)
print(conf)

accuracy <- sum(diag(conf))/sum(conf)
print(accuracy)



####################### Decision tree algorithm
library("party")
Dtree_model=ctree(train_mod$Loan_Status~.,data = train_mod)
plot(Dtree_model)

## validation
test_mod$prid=predict(Dtree_model,test_mod)
View(test_mod)
install.packages("ROCR")
library("ROCR")
score=prediction(as.numeric(test_mod$prid),as.numeric(test_mod$Loan_Status))
performance(score,"auc")
plot(performance(score,"tpr","fpr"),col="green")

#####################logostic regression model

model <- glm(Loan_Status~.,family=binomial(link='logit'),data=train_lg)
summary(model)
anova(model, test="Chisq")
test_lg=test_mod[,-1]
prid=predict(model,test_mod)

library("ROCR")
score=prediction(as.numeric(prid),as.numeric(test_mod$Loan_Status))
performance(score,"auc")
plot(performance(score,"tpr","fpr"),col="green")

