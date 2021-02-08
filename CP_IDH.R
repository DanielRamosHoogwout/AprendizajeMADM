dataset = read.csv2("data_HDI_2019.csv")
# 1. Realiza una anàlisis de componenetes principales de los datos anteriores

#Primero eliminamos las filas con valores faltantes
dataset = na.omit(dataset)

#Eliminamos la primera columna y lo transformo en una matriz
df_mat <- as.matrix(dataset[ ,-1])

#Aplicamos el análisis de componentes principales
pr.out=prcomp(df_mat, scale=TRUE)

# 2. Interpreta la primera y la segunda componente principal a partir de los vectores de cargas.

pr.out$rotation
#La primera componente principal:
#tiene asociaciones positivas con HDI, LEB, EYEDU, MYEDU, GNIpc, IHDI, P2EDU_M.
#Y asociaciones negativas con CHI, IN_LE, IN_EDU, GII, ABR.
#Este componente recoge principalmente la desigualdad.

#El segundo componente principal:
#Tiene asociaciones positivas con INC_40_POOR y negativas con IN_INC, INC_10_RICH, INC_1_RICH, GINI
#Asi que este componente recoge mayoritariamente información sobre la renta.

# 3. Interpreta el biplot de la primera y segunda componente principal. ¿Qué puedes decir de los scores?
biplot(pr.out, scale=0)
#Se obtiene una representación bidimensional de las dos primeras componentes.
#Los paises (scores) estan representados en números, cuanto más cerca se encuentre de un
#indice (vector de cargas) mayor será su puntuación y viceversa.
#Como se ha dicho anteriormente cuanto más abajo este menos renta tendrá y viceversa.
#Cuanto más hacia la derecha este el país, más desigual será y viceversa.

# 4. ¿Con cuantas componentes te quedarías para representar de forma resumida toda la información contenida en los datos?
pr.out$sdev
#Escogería 3 componentes ya que la desviación del 4 componente es menor de 1.
screeplot(pr.out, type = "l", main = "Varianzas de los Componentes Principales",
          col = "blue", cex.main = 0.8)
#Aunque según la regla del codo podríamos escoger 2

# 5. Calcula las componentes principales de las variables LEB, EYEDU, MYEDU y GNIpc. 
# Estudia sus vectores de carga y comenta dichos resultados en función de cómo pondera
# el IDH los diferentes indicadores. ¿Con cuantas componentes principales te quedarías?
dataset2 = read.csv2("data_HDI_2019.csv")
df_mat2 <- as.matrix(dataset2[ ,-1])
new_mat <- df_mat2[,2:5]
pr.out.2=prcomp(new_mat, scale=TRUE)
pr.out.2$rotation
#La primera componente recoge muy bien que la ONU da el mismo peso a estos indicadores
#La segunda recoge principalmente el PIB per capita
biplot(pr.out.2, scale=0)
pr.out.2$sdev
#Con el criterio de la varianza, nos quedariamos con una sola componente
screeplot(pr.out.2, type = "l", main = "Varianzas de los Componentes Principales",
          col = "blue", cex.main = 0.8)
#En cambio con la regla del codo, cogeriamos 2 componentes principales
