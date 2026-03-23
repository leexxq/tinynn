from nn import MLP
from autograd import Value

x = [[2.0,3.0,-1.0],[3.0,-1.0,0.5],[0.5,1.0,1.0],[1,0,1.0,1.0]]
y = [1.0,0,1.0,0]
mlp = MLP(3,[4,4,4,1])
y_pred = [mlp.forward(xi) for xi in x]
print(y_pred)
loss = sum((y-yp)**2 for y,yp in zip(y,y_pred))
print(loss)
loss.backward()
mlp.layers[0].ns[0].w[0]
mlp.layers[0].ns[0].w[0].grad
mlp.backward()
for _ in range(1000):
    y_pred = [mlp.forward(xi) for xi in x]
    loss = sum((y-yp)**2 for y,yp in zip(y,y_pred))
    loss.backward()
    mlp.backward()
