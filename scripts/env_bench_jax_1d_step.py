import jax, jax.numpy as jnp, time
jax.config.update("jax_enable_x64", False)
def init(key,sizes):
    ps=[]
    for a,b in zip(sizes[:-1],sizes[1:]):
        key,k=jax.random.split(key); ps.append((jax.random.normal(k,(a,b))*jnp.sqrt(1/a), jnp.zeros(b)))
    return ps
def net(p,x):
    h=jnp.array([x])
    for W,b in p[:-1]: h=jnp.tanh(h@W+b)
    W,b=p[-1]; return (h@W+b)[0]
uxx=jax.grad(jax.grad(net,1),1)
def loss(p,X): return jnp.mean(jax.vmap(lambda x: uxx(p,x))(X)**2)
step=jax.jit(jax.value_and_grad(loss))
p=init(jax.random.PRNGKey(0),[1,32,32,32,1]); X=jnp.linspace(0,1,256)
l,g=step(p,X); jax.block_until_ready(g)
t=time.time()
for i in range(500):
    l,g=step(p,X); p=jax.tree_util.tree_map(lambda a,b:a-1e-3*b,p,g)
jax.block_until_ready(p); print("ms/step f32", (time.time()-t)/500*1000)
