# Machine Learning From Scratch

I created a repository containing core machine learning algorithms implemented entirely from using standard Python and vectorized NumPy matrix math, completely avoiding high-level framework dependencies like scikit-learn.

## 📁 Repository Modules

### 1. Simple Linear Regression

**Objective:** Train a single-feature regression model on a custom housing dataset from scratch.
   
**Mathematical Core:** Implements a broadcasting squarred cost function to evaluate model error and utilizes an explicit gradient descent pipeline to compute partial derivatives and update weights.

**Engineering Focus:** Built to master vectorized data shapes, eliminate execution loop bottlenecks, and accurately track cost convergence down the gradient.



### 2. Multi-Feature Linear Regression 

**Objective:** A pure Python and NumPy implementation of multivariate linear regression built from scratch

**Core Mechanics:** The repository implements a modular machine learning pipeline broken down into distinct mathematical components

1. ***Prediction***: Computes the hypothesis $\hat{y}$ simultaneously for all examples using a vectorized dot product.
2. ***Mathematical Core:*** Implements a broadcasting squared cost function to evaluate model error and utilizes an explicit gradient descent pipeline to compute partial derivatives and update weights.
3. ***Optimization Loop***: Minimizes the cost function iteratively by updating parameters against a defined learning rate ($\alpha$)[cite: 15].

***Mathematical Matrix Geometry:*** To perform parallel calculations without inefficient nested sequential loops, the matrix dimensions are strictly aligned as follows:

* **Feature Matrix ($X$)**: Shape `(m, n)` — where $m$ is the number of training examples (e.g., 100) and $n$ is the number of features (e.g., 4).
* **Weight Vector ($w$)**: Shape `(n,)` — a 1D vector matching the number of features.
* **Bias ($b$)**: A scalar value uniformly broadcasted across all predictions.
* **Prediction Vector ($\hat{y}$)**: Shape `(m,)` — the resulting output containing a single prediction per example.

**The Gradient Transpose Requirement:** To calculate the weight gradients without shape mismatches, the feature matrix $X$ is explicitly transposed ($X^T$):

$$\text{Shape of } X^T: (n \times m) \quad \times \quad \text{Shape of Error } (\hat{y} - y): (m \times 1) \longrightarrow \text{Gradient Shape: } (n \times 1)$$
