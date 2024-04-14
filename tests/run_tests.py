import unittest

# pytest tests/ --fulltrace -s
if __name__ == '__main__':
    loader = unittest.TestLoader()
    tests = loader.discover('tests')
    runner = unittest.TextTestRunner()
    runner.run(tests)