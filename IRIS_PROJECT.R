
#Loading data into R
iris_data=read.csv("fileLocation/iris.csv")

#Data understanding
head(iris_data)
summary(iris_data)
dim(iris_data)

install.packages("GGally")
library(GGally)

#plotting features against each other like corelation based on species, histograms, scatter plots etc 
ggpairs(iris_data, aes(colour = Species, alpha = 0.4))


#Data spliting into train and test data
set.seed(123)##random initialization
train=sort(sample(nrow(iris_data),0.7*nrow(iris_data)))
train_data=iris_data[train,]   ## training dataset
test_data=iris_data[-train,]

summary(train_data$Species)
summary(test_data$Species)
# Design models

#Decision tree algorithm 
library(rpart)
Dec_tree=rpart(train_data$Species~.,data=train_data)
prid=predict(Dec_tree,test_data)
library(caret)
confusionMatrix(prid,as.numeric(test_data$Species))

#KNN Algorthm 
iris_label=train_data$Species
class(iris_label)
iris_lable_t=test_data$Species
library("class")
dat_train <- train_data[ ,-5]         #leave your target variable out 
dat_test <- test_data[ ,-5]       
head(dat_test)
iris_test_pred1 <- knn(train = dat_train, test = dat_test,cl= iris_label,k = 3,prob = TRUE) 
install.packages("gmodels")
library(gmodels)
CrossTable(x = iris_lable_t, y = iris_test_pred1,prop.chisq=FALSE) 
#A total of 1 out of 45, or nearly 3 percent of Species were incorrectly classified by the kNN classifier
# therefore 95 percent accuracy 
#95% of accuracy is showing by the model 
#> (12+13+19)/45
#[1] 0.9777778


#SVM model
library("e1071")
x <- subset(train_data, select=-Species)
y <- train_data$Species
svm_model <- svm(x,y)
summary(svm_model)
x1=subset(test_data,select=-Species)
y1=test_data$Species
pred <- predict(svm_model,x1)
table(pred,y1)
#A total of 0 out of 45, or nearly 0 percent of Species were incorrectly classified by the SVM classifier
# therefore 100 percent accuracy 
#95% of accuracy is showing by the model 
#> (12+14+19)/45
#[1] 1