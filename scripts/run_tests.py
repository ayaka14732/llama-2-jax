import os
from glob import iglob
import subprocess

def run_test(test_file: str) -> int:
    python_interpreter = os.path.join('venv', 'bin', 'python')
    result = subprocess.run([python_interpreter, test_file])
    return result.returncode

if __name__ == '__main__':
    for test_file in iglob('tests/**/*.py', recursive=True):
        result = run_test(test_file)
        status = '✅ PASSED' if result == 0 else '❌ FAILED'
        print(f'{status} {test_file}')
