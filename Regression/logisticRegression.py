import numpy as np 
import math

#tabe haye formul ha ro taarif mikonim
#chand ta nemoone dade migirim az voroodi


print("Enter the Xs list separated by space: ")
Xtrain = [int(x) for x in input().split()]

print("Enter the Ys list separated by space: ")
Ytrain = [int(y) for y in input().split()]

#parameters 
numTrainingExamples = len(Xtrain)
learningRate = .1
tekrar =1000

#formule e logostic regression bood 1/(1+e^(-b0+b1*x))
def formule_e_kolli(b0,b1,x):
	return (1/(1+np.exp(-(b0 + b1*x))))

#tabe ye hazine
def costFunction(b0, b1):
	loss = 0
	for i, j in zip(Xtrain,Ytrain):
		temp = (-j*math.log(formule_e_kolli(b0,b1,i))) - (1-j)*math.log(1 - formule_e_kolli(b0,b1,i))
		loss += temp
	return loss/numTrainingExamples

#hesab kardan e moshtagh addadi nebast be b0 ya b1
def moshtagh(withRespectTo, b0, b1):
	h = 1./1000.
	if (withRespectTo == "beta0"):
		rise = costFunction(b0 + h, b1) - costFunction(b0,b1)
	else: #nesbat be beta1 
		rise = costFunction(b0 , b1 + h) - costFunction(b0,b1)
	run = h
	shib = rise/run
	return shib

#beta ro update mikonim 
def betaUpdate(withRespectTo, b0, b1):
	if (withRespectTo == "beta0"):
		b0 = b0 - learningRate*(moshtagh(withRespectTo, b0, b1))
		return b0
	else: #pas beta1 hast
		b1 = b1 - learningRate*(moshtagh(withRespectTo, b0, b1))
		return b1

#Random initialization of the beta values
beta0 = np.random.uniform(0,1)
beta1 = np.random.uniform(0,1)

Xtest = int(input("enter test number:\n"))

for i in range(0,tekrar):
	beta0 = betaUpdate("beta0", beta0, beta1)
	beta1 = betaUpdate("beta1", beta0, beta1)
print (formule_e_kolli(beta0,beta1,Xtest))
