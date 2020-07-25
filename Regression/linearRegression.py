import numpy as np 
import math

#tabe haye formul ha ro taarif mikonim
#chand ta nemoone dade migirim az voroodi


print("Enter the Xs list separated by space: ")
Xtrain = [int(x) for x in input().split()]
#Xtrain = [1,2,3,4]

print("Enter the Ys list separated by space: ")
Ytrain = [int(y) for y in input().split()]
#Ytrain = [3,4,7,12]

#parameters 
learningRate = .01
tekrar =1000

#regression e khatti bood: y = beta0 + x*beta1
def formule_e_kolli(b0,b1,x):
	return (b0 + b1*x)

#tabe e hazina barabar hast ba jam e majzoor haye regression menhaye meghdar e asli
def betaHat(b0, b1):
	loss = 0

	#zip function xtrain o y train o jofti dorost mikone
	for i, j in zip(Xtrain,Ytrain):
		temp = math.pow((formule_e_kolli(b0,b1,i) - j),2)
		loss += temp
	return loss


#hesab kardan e moshtagh addadi nebast be b0 ya b1
def moshtagh(withRespectTo, b0, b1):
	h = 1./1000.
	if (withRespectTo == "beta0"):
		rise = betaHat(b0 + h, b1) - betaHat(b0,b1)
	else: #ba dar nazar gereftan beta1
		rise = betaHat(b0 , b1 + h) - betaHat(b0,b1)
	run = h 
	shib = rise/run
	return shib


#update e vaznha ba gereftan moshtagh az hazineha hast  ba dar nazar gereftan beta 
def betaUpdate(withRespectTo, b0, b1):
	if (withRespectTo == "beta0"):
		b0 = b0 - learningRate*(moshtagh(withRespectTo, b0, b1))
		return b0
	else: #pas beta1 hast
		b1 = b1 - learningRate*(moshtagh(withRespectTo, b0, b1))
		return b1
	

#Random meghdar e beta ra hesab mikonim
beta0 = np.random.uniform(0,1)
beta1 = np.random.uniform(0,1)
#Test value

Xtest = int(input("enter test number:\n"))
#Xtest = 6

for i in range(0,tekrar):
	beta0 = betaUpdate("beta0", beta0, beta1)
	beta1 = betaUpdate("beta1", beta0, beta1)
print (formule_e_kolli(beta0,beta1,Xtest))
