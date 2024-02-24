import numpy as np
from conservation_laws.polynomials import *

def get_coef_equations(P, Vector_fields): 
    """
    return the list of the linear equations in the coefficients a_i of the polynomial P
    that the a_i must satisfy if  <gradP(.), v_i(.) > = 0.

    Parameters:
    -----------
    P: polynomial 
        polynomial with D indeterminates x_1, ..., x_D and with indeterminate coefficients a_1, ..., a_m
    Vector_fields: np.array 
        np.array of size (D, d) whose columns correspond to vector fields v_1, ..., v_d.
        Each element of the array is a polynomial with D indeterminates x_1, ..., x_D.
    
    Returns:
    -----------
    list_coef_equations: list
        list of the linear equations in the coefficients a_i, that the a_i must satisfy if  <gradP, v_i > = 0
    """

    #  calculate the system  <gradP, Vector_fields> (d linear equations in the coef a_i, polynomial in the x_i)
    polynomial_system = P.gradient() @ Vector_fields 
    
    # calculate the coefficients of these d polynomials (linear in the a_i)
    list_coef_equations = []
    for i in range(len(polynomial_system)):
        list_coef_equations.extend(polynomial_system[i].coefficients())
        
    return list_coef_equations

def build_matrix(k, Vector_fields, general = False):
    """
    Build the matrix associated to the linear system of equations returned by get_coef_equations.

    Parameters:
    -----------
    k: int
        degree of the polynomial
    Vector_fields: np.array 
        np.array of size (D, d) that corresponds to the vector fields.
    general: bool
        if True, we consider any polynomials (with D indeterminates) of degree k. If False, we consider only homogeneous polynomials of degree k.
    
    Returns:
    -----------
    Matrix: np.array
        np.array that corresponds to the matrix of the linear system of equations.
    """
    
    D = np.shape(Vector_fields)[0]
    if general == False:
        Ring_coef, Ring_ambiant = ring(k, D)
        #R, Ring, coef, variables_list, coefficients_list = ring(k, D)
        P = create_generic_homogeneous_polynomial(k, D)

    if general == True:
        R, Ring, coef, variables_list, coefficients_list = ring(k, D, general=True)
        # caution: coefficients_list is a list of lists in that case
        P = create_generic_polynomial(k, D)

    eq = get_coef_equations(P, Vector_fields)
    #coef, indetermiates = Ring_coef.gens(), Ring_ambiant.gens()

    N, M = len(eq), len(P.coefficients())

    coefficient_matrix = np.empty((N, M))
    for i in range(N):
        degree_list = list(eq[i].degrees())
        non_zero_coefficients = list(eq[i].coefficients())
        u = 0
        for j in range(M):
            if degree_list[j] != 0:
                degree_list[j] = non_zero_coefficients[u]
                u += 1
        coefficient_matrix[i, :] = degree_list
    matrix = coefficient_matrix.T 
    return matrix

def get_conserved_polynomials(k, Vector_fields, get_list_of_all_conserved_polynomial= False,  general = False):
    """
    return the number of independent polynomial conserved functions associated with the vector fields. 
    If get_list_of_all_conserved_polynomial == True, return a list of all conserved polynomials associated with the vector fields.
    
    Parameters:
    ----------
    k: int
        degree of the polynomial
    Vector_fields: np.array of size (D, d) 
        vector fields
    get_list_of_all_conserved_polynomial: bool
        if True, returns a list of all conserved polynomials associated with the vector fields
    general: bool
        if True, returns the list of all conserved polynomials of degree k associated with the vector fields
        if False, returns the list of all conserved  *homogeneous* polynomials of degree k associated with the vector fields
        
    Returns:
    ----------
    rank: int
        rank of the linear system 
    If get_list_of_all_conserved_polynomial == True:
        P:  list of all conserved polynomials associated with the vector fields (homogeneous or not)
    """
    L = build_matrix(k,Vector_fields, general)
    D = np.shape(Vector_fields)[0]
    indeterminates_list = [var('x'+str(i+1)) for i in range(D)]
    R = PolynomialRing(RR, indeterminates_list)
    Mat = matrix(RR, L)
    ker = Mat.kernel()
    ker_basis = ker.basis()
    
    # next we reconstruct all polynomials associated with each vector in the kernel basis
    kernel_dim = len(ker_basis)
    P = []
    grad_P = []
    if kernel_dim == 0:
        return 0
    if general == False:
        for i in range(kernel_dim):
            Q = 0
            coefficients_list = get_homogeneous_exponents(k, D)
            for j in range(len(coefficients_list)):
                Q += ker_basis[i][j]*R.monomial(*coefficients_list[j])
            P.append(Q)
            grad_P.append(Q.gradient())
        grad_P = np.array(grad_P).T
    if general == True:
        for i in range(kernel_dim):
            Q = 0
            m = 0
            for j in range(k+1):
                coefficients_list = get_homogeneous_exponents(j, D)
                for k in range(len(coefficients_list)):
                    Q += ker_basis[i][k + m]*R.monomial(*coefficients_list[k])
                m += len(coefficients_list)
            P.append(Q)
            grad_P.append(Q.gradient())
        grad_P = np.array(grad_P).T

    # then we evaluate at a point and compute the rank
    u, v = np.shape(grad_P)
    evaluation = np.empty((u, v))
    D = np.shape(Vector_fields)[0]
    value = np.random.rand((D))
    dic = {}
    for k in range(D):
        dic[f"x{k+1}"] = value[k]
    for i in range(u):
        for j in range(v):
            evaluation[i, j] = grad_P[i][j](**dic)
    rank = np.linalg.matrix_rank(evaluation)
      
    if get_list_of_all_conserved_polynomial == False:
        return rank
    else:
        return P, rank