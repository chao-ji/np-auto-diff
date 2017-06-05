# reverse-mode-auto-differentiation

* Implemented backprop algorithm by auto differentiation using only Python and NumPy. 
* Created `Operation`-like constructs (similar to tensorflow) for arithmetic operations (e.g. matrix multiplication, softmax with cross-entropy)  
* Computed gradients of target variable (scaler) with respect to entries of matrix by multiplying gradients with respect to intermediate variables and the Jocobian matrix (local derivative)
* Demo on toy dataaset with 3-way classification
