library(boot)
dataset = read.csv2("CV_RS_quantils.csv")
attach(dataset)

set.seed(1)
cv.error10=rep(0,10)
for(i in 1:10){
  glm.fit <- glm(q99 ~ poly((1/(n^-(1/2))),degree=i))
  cv.error10[i]=cv.glm(dataset,glm.fit,K=10)$delta[1]
}
cv.error10 #Es el que tiene menor error

#2
#Probablemente aumentando el pmax el error seguiria disminuyendo

#3
#Hacemos un shuffle de los datos y lo comprobamos
set.seed(1)

rows <- sample(nrow(dataset))
df <- dataset[rows, ]
detach(dataset)
attach(df)
cv.error10=rep(0,10)
for(i in 1:10){
  glm.fit <- glm(q99 ~ poly((1/(n^-(1/2))),degree=i))
  cv.error10[i]=cv.glm(dataset,glm.fit,K=10)$delta[1]
}
cv.error10

#Por tanto el orden inicial de las observaciones si que da problemas a la hora
#de hacer CV




