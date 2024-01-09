import numpy as np
import scipy.optimize as opt


def constr1(om, const_val,cv2):
    ## om is 1*6 form
    #om_mat = om.reshape((2,3))
    results = []
    for row in om:
        results.append(sum(row)-const_val-cv2)
    return np.array(results)

def posc(om):
    return om

def obj_func(om, const_v ):
    return abs(np.prod(om) - const_v)

# form is 2*3
if __name__ == '__main__':
    x0_v = np.array([2,2,2,2,2,2]).reshape(2,3)
    res = opt.minimize(fun=obj_func,args=2,x0=x0_v,constraints=({'type':'ineq','fun':posc}))
    print(res)
