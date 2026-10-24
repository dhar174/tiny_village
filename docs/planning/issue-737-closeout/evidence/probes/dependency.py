import importlib.util,json
import main
assert importlib.util.find_spec('faiss') is None
assert importlib.util.find_spec('pygame') is None
ok,missing=main.check_dependencies('minimal')
assert ok and not missing,(ok,missing)
print(json.dumps({'faiss_absent':True,'pygame_absent':True,'minimal_dependency_check':ok,'scope':'dependency check only; canonical core runner supplies startup/event/memory evidence'}))
