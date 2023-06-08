import ast
import os
import sys

def join_filenames(filenames: list[str]):
    """
    This function takes a list of filenames and returns a string with the filenames joined by ', ' 
    and the last one by ' and '. Each filename is enclosed in backticks.

    ```
    >>> join_filenames(['abc'])
    '`abc`'
    >>> join_filenames(['abc', 'def'])
    '`abc` and `def`'
    >>> join_filenames(['abc', 'def', 'ghi'])
    '`abc`, `def` and `ghi`'
    ```
    """
    if len(filenames) == 0:
        return ''
    if len(filenames) == 1:
        return f'`{filenames[0]}`'
    return ', '.join(f'`{name}`' for name in filenames[:-1]) + f' and `{filenames[-1]}`'

class NumPyAsArrayVisitor(ast.NodeVisitor):
    def visit_Import(self, node):
        for alias in node.names:
            if alias.name == 'numpy' or alias.name == 'numpy as np':
                self.generic_visit(node)

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name) and node.value.id == 'np' and node.attr == 'asarray':
            allowlist = ('lib/array_utils/__init__.py',)
            if not any(self.filepath.endswith(f) for f in allowlist):
                print(f'Error: `np.asarray` is only allowed in {join_filenames(allowlist)}, but it is used in `{self.filepath}` line {node.lineno}', file=sys.stderr)

def check_file(file):
    with open(file, 'r') as source:
        tree = ast.parse(source.read())
        visitor = NumPyAsArrayVisitor()
        visitor.filepath = file
        visitor.visit(tree)

def check_directory(directory):
    for foldername, subfolders, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.py'):
                check_file(os.path.join(foldername, filename))

if __name__ == "__main__":
    check_directory('lib')
    check_directory('tests')
    check_directory('scripts')
