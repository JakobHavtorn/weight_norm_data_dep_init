import torch
import torch.nn as nn

from weight_norm import weight_norm


module = nn.Linear(32, 64)

x = torch.randn(128, 32)

# Forward pass with module without weight normalization.
# Output is not normalized.
out_1 = module(x)
print(out_1.mean(), out_1.std())

# Weight normalize module and attach initialization hooks.
module = weight_norm(module)

# First forward pass with weight normalization initializes parameters.
# Output is normalized but has no gradients.
out_2 = module(x)
print(out_2.mean(), out_2.std())

# Second forward pass with weight normalization initializes parameters.
# Output is normalized and also has gradients.
out_3 = module(x)
print(out_3.mean(), out_3.std())
