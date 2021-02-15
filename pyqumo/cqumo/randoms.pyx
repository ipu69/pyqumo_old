import numpy as np
from libcpp.vector cimport vector
from pyqumo.cqumo.randoms cimport RandomVariable as CxxRandomVariable, Randoms

cdef class Rnd:
    cdef int index
    cdef vector[double] samples
    cdef int cacheSize
    cdef object fn

    def __cinit__(self, object fn, int cache_size = 10000):
        self.cacheSize = cache_size
        self.samples = vector[double](cache_size, 0.0)
        self.index = cache_size
        self.fn = fn

    def __init__(self, fn, cache_size=10000):
        pass

    # noinspection PyAttributeOutsideInit
    cdef double eval(self):
        cdef int cacheSize = self.cacheSize
        cdef int index = self.index
        cdef object fn = <object>self.fn
        if index >= cacheSize:
            self.samples = fn(cacheSize)
            self.index = 0
        x = self.samples[self.index]
        self.index += 1
        return x

    def __call__(self):
        return self.eval()

    def __repr__(self):
        return f"<CyRnd: ->"


cdef class RandomsFactory:
    cdef Randoms* randoms

    def __init__(self):
        self.randoms = new Randoms()
    
    def __dealloc__(self):
        del self.randoms
    
    def createConstantVariable(self, value):
        cdef CxxRandomVariable *c_var = self.randoms.createConstant(value)
        var = Variable()
        var.set_variable(c_var)
        return var
    
    def createExponentialVariable(self, rate):
        cdef CxxRandomVariable *c_var = self.randoms.createExponential(rate)
        var = Variable()
        var.set_variable(c_var)
        return var

    def createNormalVariable(self, mean, std):
        cdef CxxRandomVariable *c_var = self.randoms.createNormal(mean, std)
        var = Variable()
        var.set_variable(c_var)
        return var
    
    def createUniformVariable(self, a, b):
        cdef CxxRandomVariable *c_var = self.randoms.createUniform(a, b)
        var = Variable()
        var.set_variable(c_var)
        return var

    def createHyperExponentialVariable(self, rates, weights):
        cdef vector[double] _rates = rates
        cdef vector[double] _weights = weights
        cdef CxxRandomVariable *c_var = \
            self.randoms.createHyperExp(_rates, _weights)
        var = Variable()
        var.set_variable(c_var)
        return var
    
    def createErlangVariable(self, shape, param):
        cdef CxxRandomVariable *c_var = self.randoms.createErlang(shape, param)
        var = Variable()
        var.set_variable(c_var)
        return var
    
    def createMixtureVariable(self, vars, weights):
        cdef vector[CxxRandomVariable*] _vars
        cdef vector[double] _weights = weights
        for var in vars:
            if not isinstance(var, Variable):
                classname = f"{Variable.__module__}.{Variable.__name__}"
                raise RuntimeError(f"var type {type(var)} is not {classname}")
            _vars.push_back((<Variable>var).get_variable())
        cdef CxxRandomVariable *c_var = \
            self.randoms.createMixture(_vars, _weights)
        ret_var = Variable()
        ret_var.set_variable(c_var)
        return ret_var
    
    def createAbsorbSemiMarkovVariable(self, vars, p0, trans, absorb_state):
        cdef vector[CxxRandomVariable*] c_vars
        cdef vector[double] c_initProbs = p0
        cdef vector[vector[double]] c_trans = trans
        cdef int c_absorbState = absorb_state

        # Fill variables:
        for var in vars:
            if not isinstance(var, Variable):
                classname = f"{Variable.__module__}.{Variable.__name__}"
                raise RuntimeError(f"var type {type(var)} is not {classname}")
            c_vars.push_back((<Variable>var).get_variable())
        
        cdef CxxRandomVariable *c_ret_var = \
            self.randoms.createAbsorbSemiMarkov(
                c_vars, c_initProbs, c_trans, c_absorbState
            )
        result = Variable()
        result.set_variable(c_ret_var)
        return result
    
    def createChoiceVariable(self, values, weights):
        cdef CxxRandomVariable *c_var = \
            self.randoms.createChoice(values, weights)
        var = Variable()
        var.set_variable(c_var)
        return var
    
    def createSemiMarkovArrivalVariable(self, vars, p0, all_trans_probs):
        cdef vector[CxxRandomVariable*] c_vars
        cdef vector[double] c_initProbs = p0
        cdef vector[vector[double]] c_allTransProbs = all_trans_probs

        # Fill variables:
        for var in vars:
            if not isinstance(var, Variable):
                classname = f"{Variable.__module__}.{Variable.__name__}"
                raise RuntimeError(f"var type {type(var)} is not {classname}")
            c_vars.push_back((<Variable>var).get_variable())
        
        cdef CxxRandomVariable *c_ret_var = \
            self.randoms.createSemiMarkovArrival(
                c_vars, c_initProbs, c_allTransProbs)
        result = Variable()
        result.set_variable(c_ret_var)
        return result


cdef class Variable:
    cdef CxxRandomVariable* variable

    def __init__(self):
        self.variable = NULL

    def __dealloc__(self):
        del self.variable
    
    cdef set_variable(self, CxxRandomVariable *variable):
        self.variable = variable

    cdef CxxRandomVariable *get_variable(self):
        return self.variable
    
    cpdef eval(self):
        return self.variable.eval()

    def __call__(self, size):
        return np.asarray([self.eval() for _ in range(size)])
