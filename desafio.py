import numpy as np
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/allanspadini/numpy/dados/citrus.csv"

dados = np.loadtxt(url, delimiter=",", skiprows=1, usecols=np.arange(1, 6, 1))
print(f"dados: {dados}")

pesos = dados[:, 0]
diametros = dados[:, 1]

peso_laranja = pesos[:5000]
peso_grapefruit = pesos[5000:]

diametro_laranja = diametros[:5000]
diametro_grapefruit = diametros[5000:]

X_laranja = peso_laranja
Y_laranja = diametro_laranja
n = np.size(peso_laranja)
a_laranja = (n*np.sum(X_laranja*Y_laranja) - np.sum(X_laranja) *
             np.sum(Y_laranja))/(n*np.sum(X_laranja**2) - np.sum(X_laranja)**2)
b_laranja = np.mean(Y_laranja) - a_laranja*np.mean(X_laranja)
y_laranja = a_laranja*X_laranja + b_laranja

X_grapefruit = peso_grapefruit
Y_grapefruit = diametro_grapefruit
n = np.size(peso_grapefruit)
a_grapefruit = (n*np.sum(X_grapefruit*Y_grapefruit) - np.sum(X_grapefruit) *
                np.sum(Y_grapefruit))/(n*np.sum(X_grapefruit**2) - np.sum(X_grapefruit)**2)
b_grapefruit = np.mean(Y_grapefruit) - a_grapefruit*np.mean(X_grapefruit)
y_grapefruit = a_grapefruit*X_grapefruit + b_grapefruit


plt.plot(peso_laranja, diametro_laranja, 'r')
plt.plot(peso_grapefruit, diametro_grapefruit, 'g')
plt.plot(peso_laranja, y_laranja)
plt.plot(peso_grapefruit, y_grapefruit)


plt.xlabel("peso")
plt.ylabel("diametro")
plt.legend(['laranja', 'grapefruit', 'reg_laranja', 'reg_grapefruit'])

coef_angulares = np.random.uniform(0.01, 0.95, 100)
norma = []
b = 17
for i in range(100):
    norma = np.append(norma, np.linalg.norm(X_laranja*coef_angulares[i] + b))

print(f"norma: {norma}")
minimo = np.min(norma)

np.savetxt("normas_laranja.csv", norma, delimiter=',')

# plt.show()
