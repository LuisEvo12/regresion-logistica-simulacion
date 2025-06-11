import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

# --- Configuración inicial ---
np.random.seed(42)
num_samples = 100

# Matriz de características: constante + 2 variables
X = np.hstack((np.ones((num_samples, 1)), np.random.randn(num_samples, 2)))
beta_true = np.array([-1.0, 2.0, -1.0])
z = X @ beta_true

def sigmoid(t):
    return 1 / (1 + np.exp(-t))

# Probabilidades verdaderas y respuesta simulada
probabilities = sigmoid(z)
Y = np.random.binomial(n=1, p=probabilities)

# --- Gráfica 1: función sigmoide ordenada ---
plt.figure(figsize=(8, 5))
sorted_idx = np.argsort(z)
plt.plot(z[sorted_idx], probabilities[sorted_idx], color='blue', label=r'$\pi_i$ = sigmoide($x_i^T \beta$)')
plt.xlabel(r'$x_i^T \beta$')
plt.ylabel(r'$\pi_i$')
plt.title('Función logística (sigmoide)')
plt.grid(True)
plt.legend()

# --- Gráfica 2: función logit ---
def logit(p):
    return np.log(p / (1 - p))

plt.figure(figsize=(8, 5))
plt.scatter(z, logit(probabilities), color='green')
plt.xlabel(r'$x_i^T \beta$')
plt.ylabel(r'logit($\pi_i$)')
plt.title('Función logit: transformación lineal de probabilidades')
plt.grid(True)

# --- Gradiente y entrenamiento por gradiente descendente ---
def compute_gradient(beta, X, Y):
    pi = sigmoid(X @ beta)
    return X.T @ (Y - pi)

def logistic_regression_gd(X, Y, learning_rate=0.1, epochs=1000):
    beta = np.zeros(X.shape[1])
    loss_history = []
    epsilon = 1e-15

    for _ in range(epochs):
        pi = sigmoid(X @ beta)
        loss = -np.sum(Y * np.log(pi + epsilon) + (1 - Y) * np.log(1 - pi + epsilon))
        loss_history.append(loss)
        gradient = compute_gradient(beta, X, Y)
        beta += learning_rate * gradient

    return beta, loss_history

beta_gd, loss_history = logistic_regression_gd(X, Y)
print("Parámetros estimados (Gradiente Descendente):", beta_gd)
print("Parámetros reales:", beta_true)

# --- Gráfica 3: convergencia ---
plt.figure(figsize=(8, 5))
plt.plot(loss_history, color='purple')
plt.xlabel('Iteraciones')
plt.ylabel('Log-verosimilitud negativa')
plt.title('Convergencia - Gradiente Descendente')
plt.grid(True)

# --- Newton-Raphson ---
def logistic_regression_newton(X, Y, tol=1e-3, max_iter=100):
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        pi = sigmoid(X @ beta)
        g = X.T @ (Y - pi)
        H = X.T @ (np.diag(pi * (1 - pi))) @ X
        delta = np.linalg.pinv(H) @ g
        beta += delta
        if np.linalg.norm(g) < tol:
            break
    return beta

beta_nr = logistic_regression_newton(X, Y)
print("Parámetros estimados (Newton-Raphson):", beta_nr)

# --- Evaluación ---
prob_estimated = sigmoid(X @ beta_gd)
Y_pred = (prob_estimated >= 0.5).astype(int)
accuracy = np.mean(Y_pred == Y)
print(f"Precisión del modelo (GD): {accuracy:.3f}")

# --- Gráfica 4: predicción vs clase real ---
plt.figure(figsize=(8, 5))
scatter = plt.scatter(X @ beta_gd, prob_estimated, c=Y, cmap='bwr', edgecolor='k')
plt.xlabel(r'$x_i^T \hat{\beta}$')
plt.ylabel(r'$\hat{\pi_i}$')
plt.title('Probabilidades estimadas y etiquetas reales')
plt.colorbar(scatter, label='Etiqueta Y')
plt.grid(True)

# --- Gráfica 5: curva de potencia y p-valor ---
n = 28
powers = np.linspace(0, 1, 1000)
pvals = 1 - chi2.cdf(n * (2 * powers - 1)**2, df=1)

plt.figure(figsize=(8, 5))
plt.plot(powers, pvals, label='p-valor')
plt.xlabel('Potencia estadística')
plt.ylabel('p-valor')
plt.title('Relación entre potencia y p-valor')
plt.grid(True)
plt.legend()

plt.show()
