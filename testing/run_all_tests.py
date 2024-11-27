import unittest
import os
import sys


def load_tests_from_tests_folder():
    # Define the path to the 'tests' folder
    tests_dir = os.path.join(os.path.dirname(__file__), 'tests')

    # Discover all test cases in the 'tests' folder
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(tests_dir, pattern='test_*.py')

    # Filter out any base test cases (classes that should not be executed)
    filtered_tests = unittest.TestSuite()
    for suite_ in test_suite:
        for test in suite_:
            # Only add tests that are not from the base test classes
            if not hasattr(test, 'skipTest') and 'base_tests' not in str(test):
                filtered_tests.addTest(test)

    return filtered_tests


if __name__ == '__main__':
    # Load all valid tests from the 'tests' folder
    suite = load_tests_from_tests_folder()

    # Run the test suite
    runner = unittest.TextTestRunner()
    result = runner.run(suite)

    # Exit with an appropriate code depending on the result
    sys.exit(not result.wasSuccessful())
