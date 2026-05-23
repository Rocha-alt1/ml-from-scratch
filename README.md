# Machine Learning From Scratch

I created a repository containing core machine learning algorithms implemented entirely from using standard Python and vectorized NumPy matrix math, completely avoiding high-level framework dependencies like scikit-learn.

## 📁 Repository Modules

### 1. Linear Regression Optimization Engine 

**Objective:** Train a single-feature regression model on a custom housing dataset from scratch.
   
**Mathematical Core:** Implements a broadcasting squarred cost function to evaluate model error and utilizes an explicit gradient descent pipeline to compute partial derivatives and update weights.

**Engineering Focus:** Built to master vectorized data shapes, eliminate execution loop bottlenecks, and accurately track cost convergence down the gradient.