import os

def check_file(filename):
    with open(filename, encoding='utf-8') as f:
        content = f.read()
        if filename not in ('lib/array_utils/__init__.py', 'scripts/sanity_check.py'):
            assert 'np.asarray' not in content, filename

def check_directory(directory):
    for foldername, subfolders, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.py'):
                check_file(os.path.join(foldername, filename))

if __name__ == "__main__":
    check_directory('lib')
    check_directory('tests')
    check_directory('scripts')
