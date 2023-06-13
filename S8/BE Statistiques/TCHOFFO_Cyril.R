
# 1 - Chargement librairies et donnees

data=read.table(file="Data.txt",header=TRUE)

names(data)
summary(data)
dim(data)


data$DD <- as.factor(data$DD)
data$RR <- as.factor(data$RR)
summary(data)

# 2 Régréssion simple 


lm.out1 <- lm(O3~O3v,data)
summary(lm.out1)

# nuage de points correspondant 


x11()
plot(data$O3v,data$O3,xlab="ozone maximale mesurée le jour j-1",ylab="ozone maximale mesurée le jour j",pch='+')
abline(lm.out1, col="red")

# Analyse des performances du modèle

# sur ce modèle on observe que la regression linéaire passe au centre des
# valeurs moyennes mais que beaucoup de valeurs sont très éloignées de la droite surtout les valeurs pour lesquelles l'ozone maximale mesurée le jour j-1 est grande (à partir de 100 ou 120 )
#le modèle  ne parait donc pas linéaire, le modèle n'est pas adapté à ces données 


# les prédicteurs sont significatifs (puisque la p_value < 0.05)

# le modèle n'est tout de meme pas mal puisqu'il permet d'expliquer 46/100 de variance du prédictand 
# mais p etre enrichie aves de nouveaux préditeurs




# 3 modèle sans interaction 

# comportement de la variable 03

dataPluie <- subset(data, RR=="Pluie")
dataSec <- subset(data, RR=="Sec")

t.test(dataPluie$O3,dataSec$O3,var.equal=T)


# les moyennes sont significativement différentes pour ces deux jeux de données 
# avec les valeurs de 73.39 pour RR "Pluie" et 
# 100.84 pour RR "Sec"

x11()
par(mfrow=c(1,2))
hist(dataPluie$O3)
hist(dataSec$O3)

# l'histogramme  O3 
# on peut observer qu'on trouve en  des valeurs de O3 plus élevées pour les  modalités de RR "Pluie" et "Sec" 
# entre 45 et 80 pour "Pluie" et entre 55 et 100 pour "Sec". les valeurs sont plus denses dans ces zones

# Modèle exploitant les 6 prédicteurs  disponibles 

lm.out2=lm(O3~O3v+T+N+FF+DD+RR, data)
summary(lm.out2)

# IMPACT DE N sur le prédictand 

# il a un effet significatif sur le prédictand car sa p-value < 0.05 
# il est donc non nul et égal à -2.20773 
# de ce fait il entraine une dimunition de O3 

# INTERPRATATION DES RESULTATS RELATIFS AU FACTEUR DD 

# Pour le facteur DD, le modèle prend ici prends comme valeur de référence la valeur Est, l'expression
# de la donnée associée à DD correspond donc à l'influence des  modalités Nord , Ouest , Sud sur le prédictand 
# par rapport à la modalité Est (effet différentiel D'avec la modalité de référence) . La p-value est  << 0.05 pour les modalités Nord et Sud , donc la différence d'avec la modalité Est 
# a de l'importance pour le modèle , ces parametres pour ce predicteur doivent  rester non nul (DDNord , DDSud) et on les donnent les valeurs 
#  -13.92927 et  14.45585 respectivement. On peut donc dire que la modalalité Nord fait diminuer O3 et la modalité Sud fait augmenter 03. 
# Quant à la modalité Ouest son effet différentiel d'avec la modalité Est de référenre est jugée
# non significative. Donc on peut là  considérer comme égale à cette modalité. Elle n'a pas d'influence sur le prédictand.  



# Hypothèses du cardre théorique du modèle linéaire gaussien 


x11()
plot(fitted(lm.out2),residuals(lm.out2),main="hypothèse homoscedasticite",xlab="Valeurs ajustees (Y*)",ylab="Residus")
# pour l'hypothèse d'homoscédasticité on observe ici que les valeurs ne sont pas vraiment regroupées autours de mêmes
# droites, et qu'elles se dispersent. Il faudrait appliquer une modification sur certains prédictants afin que 
# les variances des erreurs soient très proches.
# L'hypothèse d'homoscédasticité n'est donc pas entièrement satisfaite. 

x11()
qqnorm(residuals(lm.out2))
# On observe un graphique linéaire, et des valeurs qui sont en majorité contenues entre les quantiles -1 et 1.
# Les distributions sont donc très proches de normales, on peut donc confirmer l'hypothèse de normalité.

x11()
acf(residuals(lm.out2))
# On cherche à connaitre la corrélation entre les variables, on observe ici que peu de valeurs dépassent de 
# la ligne d'en haut, on peut donc conclure que les variables sont bien indépendantes les unes des autres, ce qui 
# valide l'hypothèse d'indépendance. 

x11() 
plot(fitted(lm.out2),data$quality,xlab="valeur prévues",ylab="valeur observées",pch="+",main="Hypothese de linearite")
# les valeurs sont très dispercées du centre , on a donc peu de linéarité. 


# On peut conserver les prédicteurs dont la p-value est inférieur au niveau alpha = 0.05
# qui sont : O3v, T, N, DD ( puisque on a 2 de ses modalités qui sont significatives)  , RR 
# tous sauf FF donc 



library(MASS)

lm.outBIC=stepAIC(lm.out2)
summary(lm.outBIC)
# Ce modèle propose de garder des  prédicteurs précédents dont la p-value était inférieure
# à 0.05 à savoir : O3v, T , N , DD. Mais pas RR. 
# IL GARDE DONC 4 prédicteurs 
# Multiple R-squared:  0.8358 pour lm.out2 et  Multiple R-squared:  0.8322 pour lm.outBIC,
# Ce modèle explique aussi mieux la variabilité du prédictand et est moins complexe car il  a moins de prédicteurs. 




# 4 -


lm.outBICint <- stepAIC(lm.out2, scope = formula(lm.out2) ~ .^2)

# Afficher le modèle Im.outBICint
summary(lm.outBICint)
# dimension du modèle : 14 
# de plus  la dimension du modèle est grande, ce qui peut mener à du sur-apprentissage des données 
# le modèle est plus complexe car il a plus de prédicteurs (14 ) mais explique tout de meme mieux la variablité du prédictand. (87/100)
# il est meilleur. complexe mais robuste  
# le predicteur DD n'est plus significatif  , O3v , FF est devenu significatif

# 5 - 

x11()
plot(data$O3[1:200], xlab="Indice",ylab="O3")
points(fitted(lm.out1),col="red",pch="+",cex=1.2)

points(fitted(lm.outBICint),col="blue",pch="+",cex=1.2)

# Le modèle lm.outBICint prévoit bien des valeurs de quality dans la moyenne des valeurs observées, 

#le  modèle  lm.outBICint ajuste effectivement mieux les valeurs que le modèle lm.out1 



# 6 - 


# fonction calculant le BIAIS et le RMSE
scores=function(obs,prev) {
  rmse=sqrt(mean((prev-obs)**2))
  biais=mean(prev-obs)
  print("Biais  RMSE") 
  return(round(c(biais,rmse),3))
}


# création de data de test et d'apprentissage
nappr=ceiling(0.8*nrow(data))
ii=sample(1:nrow(data),nappr)
jj=setdiff(1:nrow(data),ii)
datatest=data[jj,]
datapp=data[ii,]  


# transformation des modèles sur les données d'apprentissage
lm.out1=lm(formula(lm.out1),datapp)
lm.out2=lm(formula(lm.out2),datapp)
lm.outBIC=lm(formula(lm.outBIC),datapp)
lm.outBICint=lm(formula(lm.outBICint),datapp) 


scores(datapp$O3,fitted(lm.out1))
# "Biais  RMSE"
#  0.000 20.481
scores(datatest$O3,predict(lm.out2,datatest))
# "Biais  RMSE"
# 0.609  10.925
scores(datapp$O3,fitted(lm.outBIC))
# "Biais  RMSE"
#  0.000 11.722
scores(datatest$O3,predict(lm.outBICint,datatest))
# "Biais  RMSE"
# 4.134 15.338
scores(datapp$O3,fitted(lm.outBICint))

scores(datatest$O3,predict(lm.outBICint,datatest))

scores(datapp$O3,fitted(lm.outBIC))

scores(datatest$O3,predict(lm.outBIC,datatest))


# On observe que pour tous les scores sur les fichiers d'apprentissage les modèles sont non biaisées,
# mais ils le sont pour les données de test, le moins biaisé est le modèle BICint.
# Pour le RMSE, il est moins élevé pour les données de test que pour les données d'apprentissage 


# Adapatation de CV.R du TP2 , et analyse de robustesse des modèles  
RMSE=function(obs,pr){
  return(sqrt(mean((pr-obs)^2)))}

k=100 # nb iterations

tab=matrix(nrow=k,ncol=8)

for (i in 1:k) {
  
  nappr=ceiling(0.8*nrow(data))
  ii=sample(1:nrow(data),nappr)
  jj=setdiff(1:nrow(data),ii)
  datatest=data[jj,]
  datapp=data[ii,]
  
  # Estimation des modeles
  lm.out1=lm(formula(lm.out1),datapp)
  lm.out2=lm(formula(lm.out2),datapp)
  lm.outBIC=lm(formula(lm.outBIC),datapp)
  lm.outBICint=lm(formula(lm.outBICint),datapp)
  
  
  # Scores sur apprentissage
  tab[i,1]=RMSE(datapp$O3,predict(lm.out1))
  tab[i,2]=RMSE(datapp$O3,predict(lm.out2))
  tab[i,4]=RMSE(datapp$O3,predict(lm.outBIC))
  tab[i,3]=RMSE(datapp$O3,predict(lm.outBICint))

  
  # Scores sur test
  tab[i,5]=RMSE(datatest$O3,predict(lm.out1,datatest))
  tab[i,8]=RMSE(datatest$O3,predict(lm.out2,datatest))
  tab[i,6]=RMSE(datatest$O3,predict(lm.outBIC,datatest))
  tab[i,7]=RMSE(datatest$O3,predict(lm.outBICint,datatest))
  
  
}

x11()
boxplot(tab,col=c(rep("blue",4),rep("red",4)),xlab="bleu=apprentissage - rouge=test",
        names=c("lm.out1","lm.out2","lm.outBIC","lm.outBICint","lm.out1","lm.out2","lm.outBIC","lm.outBICint"),main=("Modele lineaire gaussien - Score RMSE"))


