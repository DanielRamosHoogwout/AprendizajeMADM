library(ISLR)
require("caret")
require("randomForest")
require("gbm")
set.seed(2021)

#Cargamos los datos
cancer <- read.csv("archive/data.csv")
#View(cancer)

summary(cancer)
#Podemos observar que los datos estan "balanceados" ya que diagnosis_number 
#tiene un a media de 0.62 y sus valores van entre 0 y 1
attach(cancer)
#No nos interesa la variable "id" ni "X"
cancer <- subset(cancer, select = -c(id))
cancer <- subset(cancer, select = -c(X))
#Diagnosis esta dividido entre B (Benigno) y M (Malgino)
#B = 1 y M = 0
#cancer$diagnosis_number = ifelse(diagnosis=="B",1,0)
#View(cancer)

#Simplemente hacemos la regresión logística para que que margen de mejora tenemos
#Regresión logística
glm.fit = glm(diagnosis_number~radius_mean + perimeter_mean, data = cancer, family = binomial)
summary(glm.fit)
glm.probs=predict(glm.fit,type="response")
glm.probs[1:5]
glm.pred=ifelse(glm.probs>0.5,"B","M")
table(glm.pred,diagnosis)
mean(glm.pred==diagnosis)
#Se puede observar que la predicción con la regresión logística es muy alta.


###########################################
#Comparacion de Random Forest con Boosting#
###########################################


# create a list of 75% of the rows in the original dataset we can use for training
validation_index <- createDataPartition(cancer$diagnosis, p=0.75, list=FALSE)
# select 50% of the data for validation
validation <- cancer[-validation_index,]
# use the remaining 25% of data to training and testing the models
dataset <- cancer[validation_index,]

#I change dfWN1 to dataset - to keep the same as above.
ntrain=length(dataset$diagnosis)    
train.ext=createFolds(dataset$diagnosis,k=5,returnTrain=TRUE)
test.ext=lapply(train.ext,function(x) (1:ntrain)[-x])

# I reduce the grid to save time here. 

fitControl <- trainControl(method = 'cv', number = 5, summaryFunction=defaultSummary)

#gbmGrid <-  expand.grid(interaction.depth = c(2,4,6),
#n.trees = (0:50)*50,
#                        n.trees = seq(500,1500,500),
#                        shrinkage = seq(.01, .03,.01),
#                        n.minobsinnode = 10)

gbmGrid <-  expand.grid(interaction.depth = c(1,4,7,10),
                        #n.trees = (0:50)*50,
                        #n.trees = seq(500,2000,500),
                        n.trees = c(500,1000,2000),
                        #shrinkage = seq(.005, .05,.005),
                        shrinkage = c(.005, .02,.05),
                        n.minobsinnode = 10)

gbmGrid

getModelInfo()$rf$parameters
#mtry max:
ncol(cancer)-1

rfGrid <-  expand.grid(mtry = c(2,3,4,6,9,12))
rfGrid

#Store accurancy
gbmacc <- 0
rfacc <- 0

#Store best tune.
#nrow is 5, because of using 5 inner folds, ncol depends on how many tuning parameters.
gbmtune = matrix(nrow = 5, ncol = 4)
dimnames(gbmtune) = list(c("k1", "k2", "k3", "k4", "k5"),         # row names 
                         c("n.trees", "interaction.depth", "shrinkage", "n.minobsinnode")) # column names 
rftune = matrix(nrow = 5, ncol = 1)
dimnames(rftune) = list(c("k1", "k2", "k3", "k4", "k5"),         # row names 
                        c("mtry")) # column names 

for (i in 1:5){
  
  usedata = cancer[train.ext[[i]],]
  valdata = cancer[test.ext[[i]],]
  
  fit.gbm <- train(diagnosis~., data=usedata, method = 'gbm', trControl=fitControl, tuneGrid=gbmGrid, metric='Accuracy', distribution='bernoulli')
  fit.gbm
  
  boost.caret.pred <- predict(fit.gbm,valdata)
  
  gbmacc[i]=mean(boost.caret.pred==valdata$diagnosis)
  gbmtune[i,1:4] = as.matrix(fit.gbm$bestTune)
  
  fit.rf <- train(diagnosis~., data=usedata, method = 'rf', trControl=fitControl, tuneGrid=rfGrid, metric='Accuracy', distribution='bernoulli')
  fit.rf
  
  rf.caret.pred <- predict(fit.rf,valdata)
  
  rfacc[i]=mean(rf.caret.pred==valdata$diagnosis)
  rftune[i,1] = as.matrix(fit.rf$bestTune)
  
}

gbmacc #0.9883721 0.9534884 0.9651163 0.9764706 0.9761905
rfacc #0.9651163 0.9302326 0.9069767 0.9764706 0.9761905

mean(gbmacc) #0.9719276
mean(rfacc) #0.9509973

gbmtune
rftune

#Elegimos boosting
fit.gbm <- train(diagnosis~., data=dataset, method = 'gbm', trControl=fitControl, tuneGrid=gbmGrid, metric='Accuracy', distribution='bernoulli')
fit.gbm

boost.caret.pred <- predict(fit.gbm,validation)
table(boost.caret.pred,validation$diagnosis)
#   B  M
#B 85  1
#M  4 52
mean(boost.caret.pred==validation$diagnosis)
#0.9647887

#-------------------------------------------------------------------------------
#review gbmtune to choose a model to estimate on complete train-data.
#We observe that the boosting model has a higher accuracy, so we choose this one.

#Estimamos el boosting con los mejores parámetros
cancer$diagnosis = ifelse(cancer$diagnosis == "B",1,0)
dataset$diagnosis = ifelse(dataset$diagnosis == "B",1,0)
validation$diagnosis = ifelse(validation$diagnosis == "B",1,0)

BST=gbm(diagnosis~.,data=dataset,distribution="bernoulli",n.trees=500,cv.folds=5,shrinkage=0.02,interaction.depth=10)
# check performance using 5-fold cross-validation
best.iter <- gbm.perf(BST,method="cv")
print(best.iter)
summary(BST,n.trees=best.iter)

predBST = predict(BST,n.trees=500, newdata=validation,type='response')
p.predBST=ifelse(predBST > 0.5,1,0)

table(p.predBST,validation$diagnosis)
mean(p.predBST==validation$diagnosis)
