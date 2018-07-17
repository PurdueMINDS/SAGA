import sys
import os
import torch

path = sys.argv[1]
assert os.path.isfile(path)
result = torch.load(path)

for idt in list(sorted(result.keys())):
    if idt == 'global': continue
    both_hazard, hazard_but_not, safe_but_not, both_safe = result[idt]

    print('{}: \033[34;1m{:4d}\033[0m \033[31;1m{:4d}\033[0m \033[31;1m{:4d}\033[0m \033[32;1m{:4d}\033[0m'.format(
            idt, both_hazard, hazard_but_not, safe_but_not, both_safe))
