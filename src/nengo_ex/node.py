import nengo

class ControllableNode(nengo.Node):
    def __init__(self, default_value=None, size_in=None, size_out=None, label=None, **kwargs):

        if size_in is None or size_in == 0:
            def output(t):
                return self.__value
        else:
            def output(t, x):
                if self.__value is None:
                    return x
                return self.__value
        
        self.__value = default_value
        super().__init__(output, size_in, size_out, label, **kwargs)

    @property
    def value(self):
        return self.__value
    
    @value.setter
    def value(self, value):
        self.__value = value