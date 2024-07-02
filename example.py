from model import Model
from helpers.basic_classes import KroneckerArgs, ReLUArgs, GeneralHeadArgs, Task, PoolType
import torch

from helpers.general import seed_everything


seed_everything(42)

# hyperparams
n_neighbors = 5
dynamic_knn = True
add_relus = True
eps = 0.0
share = False
negative_slope = 0.0
k = 6
in_channel = 128
add_linears = True
u_shape = True
z_align = False
pool_type = PoolType.MEAN
drop_out = 0.0

kronecker_args = KroneckerArgs(n_neighbors=n_neighbors, dynamic_knn=dynamic_knn)
relu_args = ReLUArgs(add=add_relus, eps=eps, share=share, negative_slope=negative_slope)
general_head_args = GeneralHeadArgs(k=k, in_channel=in_channel, add_linears=add_linears,
                                    u_shape=u_shape, z_align=z_align, pool_type=pool_type)

model = Model(task=Task.Regression, kronecker_args=kronecker_args, relu_args=relu_args,
              general_head_args=general_head_args, drop_out=drop_out, out_channel=1)

# sample random pc
point_cloud = torch.randn(1024, 3)

# inference
output = model(point_cloud)

print(output)
