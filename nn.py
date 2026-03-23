from autograd import Value
import random
class Neuron:
    def __init__(self,n):
        self.n = n
        self.w =[Value(random.uniform(-1,1)) for _ in range(n)]
        self.b = Value(random.uniform(-1,1))
    def forward(self,inp):
        out = sum((wi*ii for ii,wi in zip(inp,self.w)),self.b)
        out = out.tanh()
        return out
    def backward(self):
        self.w = [Value(wi.data - 0.2 * wi.grad) for wi in self.w]
        self.b = Value(self.b.data - 0.2 * self.b.grad)

class Layer:
    def __init__(self,pre=1,n=1):
        self.ns = [Neuron(pre) for _ in range(n)]

    def forward(self,inp):
        out = [neu.forward(inp) for neu in self.ns]
        return out

    def backward(self,):
        for neu in self.ns:
            neu.backward()

class MLP:
        def __init__(self,input_num,layer_nums):
            self.loss = 0
            sz = [input_num]+layer_nums
            self.layers =  [Layer(sz[i],sz[i+1]) for i in range(len(layer_nums))]

        def __repr__(self):
            return f"loss : {self.loss}"
        
        def forward(self,inp):
            for layer in self.layers:
                inp = layer.forward(inp)
            return inp[0] if len(inp) == 1 else inp

        def backward(self):
            for layer in self.layers:
                layer.backward()
