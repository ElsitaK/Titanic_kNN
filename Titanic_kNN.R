#TITANIC DATASET!
#classification of survival using k-NN algorithm
#Presentation for RC on 07/09/2020

#packages 
library(readr)
library(dplyr)
library(ggplot2)
library(class)
library(psych)


setwd("/Users/elsita/Documents/Job Search/Data Science/Kaggle/Titanic")

#the data from: https://www.kaggle.com/c/titanic/overview
train <- read_csv("train.csv")
test <- read_csv("test.csv")


#Exploratory Analysis
str(train)
#labeled - Survived
#10 features - categorical and numeric
summary(train)
#there are missing values


#Visualization
#2 numeric features
#Age
train %>% ggplot(aes(x=Age, color=factor(Survived))) +
  geom_histogram(binwidth = 5)

#Fare
train %>% ggplot(aes(x=Fare, color=factor(Survived))) +
  geom_histogram(bins = 10)
#outliers?

#BOTH in Scatterplot - eg Carthesian space
train %>% 
  filter(Fare > 0) %>%
  ggplot(aes(x=log10(Fare), y=Age, color=factor(Survived))) +
  geom_point() +
  scale_x_continuous(limits = c(0.5,3))
#removed the observations of Fare = 0 to avoid log(0) = -Inf
#should these be removed from analysis? Fare of zero is not a fare...??


####################################################################################
#Data Cleaning
#not going to worry about this for now
#probably some other steps/ best practices here
#not the subject of the presentation

####################################################################################
#kNN Classification

#Data Pre-processing for kNN

#2 features for starters
train_2_feat <- train %>% select(c(Age, Fare, Survived)) #select necessary variables
head(train_2_feat) #quick check

#Remove observations with missing values
#ie remove all rows with NAs
#again, not necessarily best practice, but just trying to get the model to work as an example
train_2_feat <- na.omit(train_2_feat)

#Scale features
#numeric variables must be on same scale to be comparable
#in order to do this - make normalization function
normalize <- function(x){
  return((x - min(x)) / (max(x) - min(x)))
}

train_2_feat <- train_2_feat %>% mutate(Age = normalize(Age), Fare = normalize(Fare))

#viz - now both features on scale from 0 to 1
train_2_feat %>% ggplot(aes(x=Fare, y=Age, color = factor(Survived))) +
  geom_point()


#Classes should be factors (not numeric)
train_2_feat$Survived <- as.factor(train_2_feat$Survived)


#Split training dataset
#in order to evaluate model performance before prediction on novel data
#must divide into training and validation

set.seed(123) #reproducible
#split into 66% train / 33% validation
train_sample_size <- floor(0.66 * nrow(train_2_feat))
train_indices <- sample(seq_len(nrow(train_2_feat)), size = train_sample_size)

#training set
train_f2 <- train_2_feat[train_indices, -3] #471 obs

#validation set
val_f2 <- train_2_feat[-train_indices, -3] #243 obs

#save labels
labs_train_f2 <- train_2_feat[train_indices, 3] #for use in model
labs_val_f2 <- train_2_feat[-train_indices, 3] #for comparison after


#Classify Using kNN
#FINALLY!!!

#get a vector of predictions! 
pred_f2_k1 <- knn(train = train_f2, test = val_f2, cl = labs_train_f2$Survived) #seems that cl must be a vector not a dataframe
print(pred_f2_k1) #vector of predictions

#our labels from the validation set
print(labs_val_f2$Survived) #vector of actual outcomes

#% Correctly Classified Instances
mean(pred_f2_k1 == labs_val_f2$Survived)  #TRUE = 1, FALSE = 0

#Confusion Matrix
table(pred_f2_k1, labs_val_f2$Survived) #creates matrix of the predicted vector answers vs the actual vector answers



#####################################################################################
#Different values of k
#it is customary to choose an odd k when number of features is even
#NO SET RULE ON HOW TO CHOOSE K

#k=3
pred_f2_k3 <- knn(train = train_f2, test = val_f2, cl = labs_train_f2$Survived, k=3) 
#k=5
pred_f2_k5 <- knn(train = train_f2, test = val_f2, cl = labs_train_f2$Survived, k=5) 
#k=7
pred_f2_k7 <- knn(train = train_f2, test = val_f2, cl = labs_train_f2$Survived, k=7) 
#k=9
pred_f2_k9 <- knn(train = train_f2, test = val_f2, cl = labs_train_f2$Survived, k=9) 
#k=11
pred_f2_k11 <- knn(train = train_f2, test = val_f2, cl = labs_train_f2$Survived, k=11) 


#do other values of k improve model performance?
mean(pred_f2_k1 == labs_val_f2$Survived)
mean(pred_f2_k3 == labs_val_f2$Survived)
mean(pred_f2_k5 == labs_val_f2$Survived)
mean(pred_f2_k7 == labs_val_f2$Survived)
mean(pred_f2_k9 == labs_val_f2$Survived)
mean(pred_f2_k11 == labs_val_f2$Survived)


#Elbow method attempt
perf <- as.data.frame(c(1,3,5,7,9,11)) 
names(perf)[1] <- "k"
perf$error[1] <- 1 - (mean(pred_f2_k1 == labs_val_f2$Survived))
perf$error[2] <- 1 - (mean(pred_f2_k3 == labs_val_f2$Survived)) #ELBOW
perf$error[3] <- 1 - (mean(pred_f2_k5 == labs_val_f2$Survived))  
perf$error[4] <- 1 - (mean(pred_f2_k7 == labs_val_f2$Survived)) 
perf$error[5] <- 1 - (mean(pred_f2_k9 == labs_val_f2$Survived)) 
perf$error[6] <- 1 - (mean(pred_f2_k11 == labs_val_f2$Survived)) 


perf %>% ggplot(aes(x=k, y=error)) +
  geom_point() +
  geom_line()



######################################################################################
#Adding Categorical Variables
#more features, yay!

#select 4 features
train_4_feat <- train %>% select(c(Age, Fare, Sex, Pclass, Survived)) #select necessary variables


##Pre-processing of Data
train_4_feat <- na.omit(train_4_feat)

#Distance Functions assume numeric data
#must convert categorical to numeric

#Binary Classification eg Yes vs No - easily changed to 0s and 1s

#Gender - 2 levels
train %>% ggplot(aes(x=Sex, fill = factor(Survived))) + 
  geom_bar()

train_4_feat$Sex <- ifelse(train_4_feat$Sex == "female", 1, 0)

#Non-binary classification: dummy coding
#categorical variables are recorded into a set of separate binary variables 

#Passenger Class - 3 levels
train %>% ggplot(aes(x=Pclass, fill = factor(Survived))) + 
  geom_bar()

dummy_features <- as.data.frame(dummy.code(train_4_feat$Pclass))
head(dummy_features) #not sure why it didnt rename using original feature name
names(dummy_features)[1] <- "Pclass_3"
names(dummy_features)[2] <- "Pclass_1"
names(dummy_features)[3] <- "Pclass_2"
head(dummy_features)

#append to dataset
train_4_feat <- cbind(train_4_feat, dummy_features)
train_4_feat$Pclass <- NULL #remove original categorical variable column

#scale
train_4_feat <- train_4_feat %>% mutate(Age = normalize(Age), Fare = normalize(Fare))
#labels as factors
train_4_feat$Survived <- as.factor(train_4_feat$Survived)

#checks
str(train_4_feat) #all numeric except labels
head(train_4_feat) #all features on scale 0 to 1


#continue as before
#training set
train_f4 <- train_4_feat[train_indices, -4] #get features except 4th which is the labels

#validation set
val_f4 <- train_4_feat[-train_indices, -4]

#save labels
labs_train_f4 <- train_4_feat[train_indices, 4] #for use in model
labs_val_f4 <- train_4_feat[-train_indices, 4] #for comparison after


#classify
#get a vector of predictions 
pred_f4_k1 <- knn(train = train_f4, test = val_f4, cl = labs_train_f4) #above converted it to a vector automatically?
print(pred_f4_k1) #vector of predictions

#our labels from the validation set
print(labs_val_f4) #vector of actual outcomes

#% Correctly Classified Instances
mean(pred_f4_k1 == labs_val_f4)  #Nice

#Confusion Matrix
table(pred_f4_k1, labs_val_f4)

#try different values of k
#get predictions
pred_f4_k3 <- knn(train = train_f4, test = val_f4, cl = labs_train_f4, k=3)
pred_f4_k5 <- knn(train = train_f4, test = val_f4, cl = labs_train_f4, k=5)
pred_f4_k7 <- knn(train = train_f4, test = val_f4, cl = labs_train_f4, k=7)

#comparisons
mean(pred_f4_k1 == labs_val_f4)
mean(pred_f4_k3 == labs_val_f4)
mean(pred_f4_k5 == labs_val_f4)
mean(pred_f4_k7 == labs_val_f4)

