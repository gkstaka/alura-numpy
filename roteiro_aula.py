import numpy as np
import matplotlib.pyplot as plt
url = "https://raw.githubusercontent.com/allanspadini/numpy/dados/citrus.csv"

dados = np.loadtxt("apples_ts.csv", delimiter=",", usecols=np.arange(1, 88, 1))

# lista = [[1,2,3],[4,5,6],[7,8,9]]
# lista = np.array(lista)

print(f"dados.nd: {dados.ndim}")
print(f"dados.size: {dados.size}")
print(f"dados.shape: {dados.shape}")

dados_transposto = dados.T  # lista.transpose()
# datas = dados_transposto[:,0]
datas = np.arange(1, 88, 1)
# print(f"datas: {dados_transposto}")
precos = dados_transposto[:, 1:6]
moscow = precos[:, 0]
kaliningrad = precos[:, 1]
petersburg = precos[:, 2]
krasnodar = precos[:, 3]
ekaterinburg = precos[:, 4]

# print(f"moscow.shape: {moscow.shape}")
# plt.plot(datas,precos[:,0])
# plt.show()

# Multiple plots

"""moscow_ano1 = moscow[0:12]
moscow_ano2 = moscow[12:24]
moscow_ano3 = moscow[24:36]
moscow_ano4 = moscow[36:48]
print(f"moscow_ano1: {moscow_ano1}")

plt.plot(np.arange(1, 13, 1), moscow_ano1)
plt.plot(np.arange(1, 13, 1), moscow_ano2)
plt.plot(np.arange(1, 13, 1), moscow_ano3)
plt.plot(np.arange(1, 13, 1), moscow_ano4)
plt.legend(['ano1', 'ano2', 'ano3', 'ano4'])
"""

# Equality

"""print(f"np.array_equal(moscow_ano3, moscow_ano4) {
      np.array_equal(moscow_ano3, moscow_ano4)}")
print(f"np.allclose(moscow_ano3,moscow_ano4,0.01): {
      np.allclose(moscow_ano3, moscow_ano4, 0.01)}")
print(f"np.allclose(moscow_ano3,moscow_ano4, 10): {
      np.allclose(moscow_ano3, moscow_ano4, 10)}")
plt.show()
"""

# NaN

# plt.plot(datas, kaliningrad)
# plt.show()

# print(f"np.isnan(kaliningrad): {np.isnan(kaliningrad)}")
# print(f"sum(np.isnan(kaliningrad)): {sum(np.isnan(kaliningrad))}")

# kaliningrad[4] = np.mean([kaliningrad[3], kaliningrad[5]])


# moscow_mean = np.mean(moscow)
# kaliningrad_mean = np.mean(kaliningrad)
# print(f"moscow_mean: {moscow_mean}")
# print(f"kaliningrad_mean: {kaliningrad_mean}")

# Array diff
# y = 2*datas + 80
y = 0.52*datas + 80
diff = np.power(moscow-y,2)
soma = np.sum(diff)
sq = np.sqrt(soma)
print(f"sq: {sq}")
# ou 
print(f"np.linalg.norm(moscow-y): {np.linalg.norm(moscow-y)}")

# plt.plot(datas, y)
# plt.plot(datas, moscow)
# plt.show()

# Array mult - regressao

Y = moscow
X = datas
n = np.size(moscow)

a = (n*np.sum(X*Y) - np.sum(X)*np.sum(Y))/(n*np.sum(X**2) - np.sum(X)**2)
# print(f"a: {a}")
b = np.mean(Y) - a*np.mean(X)
# print(f"b: {b}")
# y = a*X + b
# print(f"np.linalg.norm(moscow-y): {np.linalg.norm(moscow-y)}")

# plt.plot(X, y)
# plt.plot(X, moscow)
# plt.plot(41.5, 41.5*a + b, '*r')
# plt.plot(100, 100*a + b, '*r')
# plt.show()

# Numeros aleatorios

r1 = np.random.randint(low=40,high=100, size=100)
print(f"r: {r1}")

np.random.seed(84)
coef_angulares = np.random.uniform(low=0.10,high =0.90, size=100)
norma= np.array([])
for i in range(100):
    norma = np.append(norma,np.linalg.norm(moscow-(coef_angulares[i]*X + b)))
print(f"norma: {norma}")

dados = np.column_stack([norma, coef_angulares])
print(f"dados.shape: {dados.shape}")

np.savetxt("dados.csv", dados, delimiter=',')