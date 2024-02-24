from sage.all import PolynomialRing, ZZ, var
import numpy as np

def known_conserved_functions_linear(list_dim):
    """
    return the number of independent conserved functions already known by the literature 
    for linear neural network case
    """
    assert len(list_dim) > 2, "Number of dimensions should be greater than 2"
    D = np.array(list_dim)[:-1] @ np.array(list_dim)[1:].T
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
    
    known_vec_fields = []
    for i in range(len(list_dim)-2):
        prod = matrix_list[i].T @ matrix_list[i] - matrix_list[i+1] @ matrix_list[i+1].T
        for j in range(list_dim[i+1]):
            for k in range(list_dim[i+1]):
                known_vec_fields.append(prod[j][k].gradient())
            
    known_vec_fields = np.array(known_vec_fields).T
    u, v = np.shape(known_vec_fields)
    evaluation = np.empty((u, v))
    value = np.random.rand(D)
    dic = {}
    for k in range(D):
        dic[f"x{k+1}"] = value[k]
    for i in range(u):
        for j in range(v):
            evaluation[i, j] = known_vec_fields[i][j](**dic)
    rank = np.linalg.matrix_rank(evaluation)
    return rank

def known_conserved_functions_ReLU(list_dim):
    """
    return the number of independent conserved functions known by the literature
    """
    return sum([list_dim[i] for i in range(1, len(list_dim)-1)])