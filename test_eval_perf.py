import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

class Simple(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = mx.random.normal((1024, 1024))
    def __call__(self, x):
        return x @ self.w

model = Simple()
optimizer = optim.Adam(learning_rate=0.01)

def loss_fn(model, x):
    y = model(x)
    return mx.sum(y)

step_fn = nn.value_and_grad(model, loss_fn)

@mx.compile
def step(model_state, opt_state, x):
    model.update(model_state)
    optimizer.state = opt_state
    loss, grads = step_fn(model, x)
    optimizer.update(model, grads)
    return loss, model.state, optimizer.state

x = mx.random.normal((1024, 1024))

t0 = time.time()
for _ in range(1000):
    loss, ms, os = step(model.state, optimizer.state, x)
    model.update(ms)
    optimizer.state = os
    mx.eval(model.state, optimizer.state, loss)
print("With model.state:", time.time() - t0)

t0 = time.time()
for _ in range(1000):
    loss, ms, os = step(model.state, optimizer.state, x)
    model.update(ms)
    optimizer.state = os
    mx.eval(model.parameters(), optimizer.state, loss)
print("With model.parameters():", time.time() - t0)
