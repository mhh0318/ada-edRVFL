def rescale(A):

    min_A=A.min(0)
    max_A=A.max(0)
    out=-1+((A-min_A)/(max_A-min_A))*2

    return out