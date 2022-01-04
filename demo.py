import torch
import torch.nn as nn

from weight_norm import weight_norm


module = nn.Linear(32, 64)

x = torch.randn(128, 32)

# Perform a forward pass with module without weight normalization.
# The output is not normalized.
out_1 = module(x)
print(out_1.mean(), out_1.std())

# Weight normalize module and attach initialization hooks.
# The initialization hooks will remove themselves again during the first forward pass.
module = weight_norm(module)

# The first forward pass with weight normalization uses the hooks to initialize parameters and then removes hooks.
# We only perform a single forward pass through each module (i.e. before initializing parameters) but we normalize the 
# (unnormalized) output after the forward pass and return this instead. Therefore, the output of the first forward
# pass is equal to the output of subsequent forward passes, if we input the same batch to the module (and the module is
# deterministic). To make sure we don't compute an erroneous gradient in any potential subsequent optimization, we
# normalize the output within a `with torch.no_grad()` and return it without gradients.
out_2 = module(x)
print(out_2.mean(), out_2.std())

# The second forward pass with weight normalization uses the reparamterized module with initialized parameters.
# The output output is normalized and also has gradients.
out_3 = module(x)
print(out_3.mean(), out_3.std())

# The output of the second forward pass is equal to the output of the third forward pass, if we input the same batch.
print(f"{torch.allclose(out_2, out_3, atol=1e-6)=}")
