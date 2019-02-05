#import snana
from params import Parameter

class SN(Parameter):
    def __init__(self,param):
        super().__init__(param.name,param.sn_parameters,param.cosmology,param.telescope)

    def __call__(self):
        print('in call',self.name)
