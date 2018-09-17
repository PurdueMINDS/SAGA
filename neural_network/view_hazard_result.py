#   Copyright 2018 Jianfei Gao, Leonardo Teixeira, Bruno Ribeiro.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
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
