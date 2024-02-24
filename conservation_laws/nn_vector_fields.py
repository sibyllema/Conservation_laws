from sage.all import var, PolynomialRing, R, ZZ, product

def vector_fields_for_q_layer_LNN(list_dim): #for example list_dim = [n1, n2, n3, n4]
    """
    return D and the vector fields associated to the reparametrization of linear neural network 𝜙(𝑈1,⋯,𝑈𝑞)=𝑈1×⋯×𝑈𝑞
    """
    D = int(np.array(list_dim)[:-1] @ np.array(list_dim)[1:].T)

    list_var = [var('x'+str(i+1)) for i in range(D)]
    R = PolynomialRing(ZZ, list_var) 
    
    matrix_list = []
    for i in range(len(list_dim)-1):
        matrix = []
        for j in range(list_dim[i]):
            line = []
            for k in range(list_dim[i+1]):
                L = [0]*D
                L[j * list_dim[i+1] + k + sum([list_dim[l] * list_dim[l+1] for l in range(i)])] = 1
                line.extend([R.monomial(*L)])
            matrix.append(line)
        matrix_list.append(np.array(matrix))
  
    prod = np.linalg.multi_dot(matrix_list)
    assert prod.shape == (list_dim[0], list_dim[-1])
    vec = []
    for i in range(list_dim[0]):
        for j in range(list_dim[-1]):
            vec.append(prod[i][j].gradient())
    vec = np.array(vec).T
    return D, vec

import itertools

from sage.parallel.decorate import parallel
import numpy as np
import itertools

# Create function for operations inside first loop
@parallel
def compute_gradient_1(dims, matrix_list, bias_list, num_mat):
    vec = []
    to_multiply = [matrix_list[0].copy()[:, dims[0]]]
    for i in range(len(dims) - 1):
        to_multiply.append(np.array(matrix_list[i + 1].copy()[dims[i], dims[i+1]]))
    to_multiply.append(np.array(bias_list.copy()[num_mat - 1][dims[-1]]))
    prod = np.array(product(to_multiply)).reshape(-1)
    for l in range(len(prod)):
        vec.append(prod[l].gradient())
    return vec

# Create function for operations inside second loop (last layer)
@parallel
def compute_gradient_2(dims, matrix_list):
    vec = []
    to_multiply = [matrix_list[0].copy()[:, dims[0]]]
    for i in range(len(dims) - 1):
        to_multiply.append(np.array(matrix_list[i + 1].copy()[dims[i], dims[i+1]]))
    prod = np.array(product(to_multiply)).reshape(-1, 1) @ np.array(matrix_list[-1].copy()[dims[-1], :]).reshape(1, -1)
    for l in range(prod.shape[0]):
        for m in range(prod.shape[1]):
            vec.append(prod[l][m].gradient())
    return vec

def vector_fields_ReLU_q_layers(list_dim, bias = False):
    """
    return D and the vector fields associated to the reparametrization of ReLU networks
    """
    
    D = np.array(list_dim)[:-1] @ np.array(list_dim)[1:].T
    dim_weights = D
    
    if bias:
        D += sum([list_dim[l] for l in range(1, len(list_dim)-1)])
        
    list_var = [var('x'+str(i+1)) for i in range(D)]
    R = PolynomialRing(ZZ, list_var) 
    
    # definition of the weight matrices
    matrix_list = []
    for i in range(len(list_dim)-1):
        matrix = []
        for j in range(list_dim[i]):
            line = []
            for k in range(list_dim[i+1]):
                L = [0]*D
                L[j * list_dim[i+1] + k + sum([list_dim[l] * list_dim[l+1] for l in range(i)])] = 1
                line.extend([R.monomial(*L)])
            matrix.append(line)
        matrix_list.append(np.array(matrix))
    
    # definition of the bias
    if bias:
        bias_list = []
        for i in range(1, len(list_dim)-1):
            line = []
            for j in range(list_dim[i]):
                L = [0]*D
                L[dim_weights + j + sum([list_dim[l] for l in range(1, i)])] += 1
                line.extend([R.monomial(*L)])
            bias_list.append(line)
    
    vec = []
    if bias:
        for num_mat in range(1, len(list_dim) - 1):
            dims_list = itertools.product(*[list(range(list_dim[i])) for i in range(1, num_mat+1)])
            # use sage parallelization
            results = compute_gradient_1([(dims, matrix_list, bias_list, num_mat) for dims in dims_list])
            vec += [item for sublist in results for item in sublist[1]]  # Flatten results

    dims_list = itertools.product(*[list(range(list_dim[i])) for i in range(1, len(list_dim) - 1)])
    # use sage parallelization
    results = compute_gradient_2([(dims, matrix_list) for dims in dims_list])
    vec += [item for sublist in results for item in sublist[1]]  # Flatten results
    
    return D, np.array(vec).T
    
D, vector_fields = vector_fields_ReLU_q_layers([2, 2, 5, 3], bias=False)