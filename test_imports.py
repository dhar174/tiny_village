import importlib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    successful_imports = []
    failed_imports = []
    version_info = {}

    try:
        with open('requirements.txt', 'r') as f:
            packages = [line.strip().split('==')[0] for line in f if (stripped := line.strip()) and not stripped.startswith('#')]
    except FileNotFoundError:
        logging.error("requirements.txt not found.")
        return

    for package_name in packages:
        try:
            module = importlib.import_module(package_name.replace('-', '_'))
            successful_imports.append(package_name)
            try:
                version = getattr(module, '__version__')
                version_info[package_name] = version
            except AttributeError:
                logging.warning(f"Package {package_name} does not have a __version__ attribute.")
        except ImportError:
            logging.error(f"Failed to import package: {package_name}")
            failed_imports.append(package_name)
        except Exception:
            logging.exception(f"An unexpected error occurred while importing {package_name}.")
            failed_imports.append(package_name)

    print("\n--- Import Test Summary ---")
    print(f"Successfully imported: {len(successful_imports)}/{len(packages)}")
    if successful_imports:
        print("Successful packages:")
        for pkg in successful_imports:
            if pkg in version_info:
                print(f"  - {pkg} (version: {version_info[pkg]})")
            else:
                print(f"  - {pkg} (version: N/A)")

    if failed_imports:
        print("\nFailed imports:")
        for pkg in failed_imports:
            print(f"  - {pkg}")
    print("--- End of Summary ---")
    return len(failed_imports)
if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
