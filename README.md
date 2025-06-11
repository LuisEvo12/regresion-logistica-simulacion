# Regresión Logística Binaria: Simulación y Estimación

Este proyecto implementa un modelo de regresión logística binaria utilizando dos métodos de estimación: **gradiente descendente** y **Newton-Raphson**. Se incluyen simulaciones, visualizaciones gráficas de funciones logísticas, análisis de convergencia, y evaluación de desempeño.

##  Objetivo

Simular un conjunto de datos binarios y estimar los parámetros de un modelo logístico utilizando diferentes algoritmos. El objetivo es entender la forma de la función sigmoide y logit, comparar métodos de optimización, y visualizar métricas relevantes como la verosimilitud y la precisión.

---

##  Requisitos

- Python 3.8+
- Numpy
- Matplotlib
- Scipy

Instalar dependencias:
```bash
pip install numpy matplotlib scipy


Contenido del Código
1. Simulación de Datos
Se generan 100 muestras con dos variables independientes y un término constante.

Se usa una combinación lineal con parámetros reales (beta_true) y se transforma con una función sigmoide.

2. Funciones Matemáticas
sigmoid(t): Función logística.

logit(p): Transformación logit.

compute_gradient(): Gradiente del log-verosímil.

3. Entrenamiento
logistic_regression_gd(): Entrenamiento usando gradiente descendente.

logistic_regression_newton(): Entrenamiento con Newton-Raphson.

4. Evaluación
Precisión del modelo.

Comparación entre los parámetros reales y los estimados.

 Visualizaciones Generadas
Curva sigmoide: relación entre xᵀβ y π.

Transformación logit.

Convergencia: log-verosimilitud negativa por iteraciones.

Probabilidades vs clases reales.

Relación entre potencia estadística y p-valor.

Resultados
Parámetros reales: [-1, 2, -1]

Parámetros estimados (GD): calculados iterativamente.

Precisión del modelo: impresa en consola.

Licencia
Este proyecto está licenciado bajo la Licencia MIT. Consulta el archivo LICENSE para más detalles.

Autor
Luis Eduardo Villanueva Oliver
2025 — Proyecto de simulación y análisis estadístico con regresión logística.
