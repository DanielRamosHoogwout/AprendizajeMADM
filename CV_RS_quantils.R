library(boot)
df = read.csv2("CV_RS_quantils.csv")

set.seed(1)
cv.error10=rep(0,10)
for(i in 1:10){
  glm.fit <- glm(q99 ~ poly((1/(n^(1/2))),degree=i),data = df)
  cv.error10[i]=cv.glm(df,glm.fit,K=10)$delta[1]
}
cv.error10 #Es el que tiene menor error

#2
#Poner gráfica con el codo y listo.
#Probablemente aumentando el pmax el error seguiria disminuyendo,
#pero tiene un coste computacional muy alto.
plot(1:10,cv.error10,type="b", xlab= 'Orden', ylab = 'Error CV')

#3
#Hacemos un shuffle de los datos y lo comprobamos
set.seed(1)
rows <- sample(nrow(df))
df <- df[rows, ]
cv.error10=rep(0,10)
for(i in 1:10){
  glm.fit <- glm(q99 ~ poly((1/(n^(1/2))),degree=i), data = df)
  cv.error10[i]=cv.glm(df,glm.fit,K=10)$delta[1]
}
cv.error10
plot(1:10,cv.error10,type="b", xlab= 'Orden', ylab = 'Error CV')
#Por tanto el orden inicial de las observaciones no es relevante a la hora
#de hacer CV
