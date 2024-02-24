from sage.all import var, PolynomialRing

def list_to_str(l):
    """
    Convert a list of integers into a string
    """
    res = 'a'
    for element in l:
        res += str(element)
    return res

def get_homogeneous_exponents(k, D):
    """
    Return the list of all lists of D integers whose sum is k, i.e., 
    the list of all lists of exponents of every monomial of degree k in D indeterminates.

    Parameters:
    -----------
    k : int 
        Degree of the polynomial
    D : int
        Number of indeterminates
    
    Returns:
    --------
    res : list 
        list of all lists of D integers whose sum is k
    """
    if D == 0: # if there is 0 indeterminate, there is no coef!
        return [] 
    if D == 1: # if there is 1 indeterminate, there is only one nonzero coefficient (the k-th one)
        return [[k]]
    if k == 0: # if the degree of the polynomial is zero there is only one coefficient a_{0, ..., 0}
        return [[0]*D]
    res = []
    for i in range(k+1):
        c = get_homogeneous_exponents(k-i, D-1)
        resint = [[i]+coef for coef in c]
        res += copy(resint)
    return res

def ring(k, D, general=False):
    """
    Parameters
    ----------
    k: int
        (maximal) degree of polynomials
    D: int
        number of indeterminates
    general : bool
        if False, we consider homogeneous polynomials. If True, polynomials are general.

    Returns
    -------
    Ring_coef : ring
        the polynomial ring RR[a_1, ..., a_m] with a_1, ..., a_m  the indeterminates and with m the dimension of the vector space
        of the polynomials of degree at most k (resp. homogeneous polynomials of degree k)
        with D indeterminates if general = True (resp. if general = False).
    Ring_ambiant : ring
        the polynomial ring RR[a_1, ..., a_m][x_1, ..., x_D] with x_1, ..., x_D  indeterminates and with coefficients in Ring_coef
    """ 
    list_var = [var('x'+str(i+1)) for i in range(D)] 

    if general == False:
        list_exponents = get_homogeneous_exponents(k, D)
        list_coef = [var(list_to_str(c)) for c in list_exponents]
        Ring_coef = PolynomialRing(RR, list_coef)
        Ring_ambiant = PolynomialRing(Ring_coef, list_var)
        return Ring_coef, Ring_ambiant

    if general == True:
        list_exponents = []
        for i in range(k+1):
            list_exponents.extend(get_homogeneous_exponents(i, D))
        list_coef = [var(list_to_str(c)) for c in list_exponents]
        Ring_coef = PolynomialRing(RR, list_coef)
        Ring_ambiant = PolynomialRing(Ring_coef, list_var)
        return Ring_coef, Ring_ambiant

def create_generic_homogeneous_polynomial(k, D):
    """
    Create a homogeneous polynomial of degree  𝑘 with 
    𝐷 indeterminates and indeterminate coefficients 
    """
    Ring_coef, Ring_ambiant =  ring(k, D)
    #R, Ring, coef, list_variables, list_coefs = ring(k, D)
    coef = Ring_coef.gens()
    list_exponents = get_homogeneous_exponents(k, D)
    P = 0
    for i in range(len(list_exponents)):
        for j in range(D):
            P += coef[i]* Ring_ambiant.monomial(*list_exponents[i])
    return P 

def create_generic_polynomial(k, D):
    """
    Create a polynomial of degree  𝑘 with  𝐷 indeterminates and indeterminate coefficients
    """
    Ring_coef, Ring_ambiant = ring(k, D, general=True)
    coef = Ring_coef.gens()
    P = 0
    u = 0
    for i in range(k+1):
        list_exponents = get_homogeneous_exponents(i, D)
        for j in range(len(list_exponents)):
            P += coef[j + u]* Ring_ambiant.monomial(*list_exponents[j])
        u += len(list_exponents)
    return P 
