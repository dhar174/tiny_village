import os
import tempfile
import stat
import shutil

cache_dir = os.path.join(tempfile.gettempdir(), "hf_test_cache")

# Clean up if exists to ensure fresh run
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)

print(f"Executing relevant part of tiny_memories.py...")

# Extracting the vulnerable code snippet
with open("tiny_memories.py", "r") as f:
    lines = f.readlines()
    # Lines 338 to 343 (0-indexed: 337 to 343)
    vulnerable_code = "".join(lines[337:343])

print("Code to execute:")
print(vulnerable_code)

exec(vulnerable_code)

if os.path.exists(cache_dir):
    mode = os.stat(cache_dir).st_mode
    permissions = stat.S_IMODE(mode)
    print(f"Directory {cache_dir} permissions: {oct(permissions)}")
    if permissions == 0o777:
        print("VULNERABILITY CONFIRMED: Directory is world-writable (0o777)")
    elif permissions == 0o700:
        print("FIX VERIFIED: Directory is owner-only (0o700)")
    else:
        print(f"Directory permissions are {oct(permissions)}")
else:
    print(f"Directory {cache_dir} was not created.")
