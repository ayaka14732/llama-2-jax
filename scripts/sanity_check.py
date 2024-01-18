import os

def check_file(filename):
    with open(filename, encoding='utf-8') as f:
        content = f.read()
        if filename not in ('scripts/sanity_check.py', 'lib/array_utils/__init__.py'):
            assert 'np.asarray' not in content, filename
        if filename not in ('scripts/sanity_check.py',):
            assert 'import torch.nn as nn' not in content, filename  # Use `import torch.nn as tnn` instead
        if filename not in ('scripts/sanity_check.py',):
            assert 'import jax.numpy as np' not in content, filename  # Use `import jax.numpy as jnp` instead

def check_directory(directory):
    for foldername, subfolders, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.py'):
                check_file(os.path.join(foldername, filename))

if __name__ == "__main__":
    check_directory('lib')
    check_directory('tests')
    check_directory('scripts')

