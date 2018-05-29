from Telescope import Telescope

class Parameter:
    def __init__(self, name, sn_parameters, cosmology, Telescope):
        print('there we go',name)
        self._name=name
        self._sn_parameters=sn_parameters
        self._cosmology=cosmology
        self._telescope=Telescope

    @property
    def name(self):
        return self._name
    
    @property
    def sn_parameters(self):
        return self._sn_parameters
    
    @property
    def cosmology(self):
        return self._cosmology
    
    @property
    def telescope(self):
        return self._telescope
