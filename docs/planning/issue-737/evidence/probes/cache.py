import unittest,sys,json
suite=unittest.defaultTestLoader.loadTestsFromName('tests.test_model_cache')
result=unittest.TextTestRunner(verbosity=2).run(suite)
print(json.dumps({'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skips':len(result.skipped)}))
sys.exit(not result.wasSuccessful())
