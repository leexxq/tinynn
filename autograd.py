class Value:
    def __init__(self,val,_children=(),_op=''):
        self.data = val
        self.grad = 0.0
        self._prev = set(_children)
        self._backward = lambda:None
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data + other.data,(self,other),'+')
        def add_backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = add_backward
        return out

    def __mul__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data * other.data,(self,other),'*')
        def mul_backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = mul_backward
        return out

    def __pow__(self,p):
        assert isinstance(p,(int,float))
        out = Value(self.data ** p , (self,),f'**{p}')
        def pow_backward():
            self.grad += out.grad * p * self.data ** (p - 1)
        out._backward = pow_backward
        return out
    
    def __truediv__(self,other):
        return self * other**-1

    def __neg__(self):
        return -1 * self
    
    def __sub__(self,other):
        return -other + self

    def __rsub__(self,other):
        return -self + other

    def __rtruediv__(self,other):
        return other * self**-1 
    
    def __radd__(self,other):
        return self.__add__(other)
    
    def __rmul__(self,other):
        return self.__mul__(other)

    def sin(self):
        import math
        out = Value(math.sin(self.data),(self,),'sin')
        def sin_backward():
            self.grad += out.grad * math.cos(self.data)
        
        out._backward = sin_backward
        return out

    def cos(self):
        import math
        out = Value(math.cos(self.data),(self,),'cos')
        def cos_backward():
            self.grad -= out.grad * math.sin(self.data)

        out._backward = cos_backward
        return out
    
    def tanh(self):
        val = self.data
        
        import math
        
        val = (math.exp(2*val) - 1)/(math.exp(2*val) + 1) 
        def tanh_backward():
            self.grad += out.grad * (1 - val**2)
        out =  Value(val,(self,),'tanh')
        out._backward = tanh_backward
        return out
   
    def exp(self):
        import math
        out = Value(math.exp(self.data),(self,),'exp')
        def exp_backward():
            self.grad += out.grad * out.data
        out._backward = exp_backward
        return out        
    
    def backward(self):
        visited = set()
        topo = []
        def recursion(root):
            if root not in visited:
                root.grad=0
                visited.add(root)
                for node in root._prev:
                    recursion(node)
                topo.append(root)
        recursion(self)
        topo.reverse()
        self.grad = 1
        for node in topo:
            node._backward()