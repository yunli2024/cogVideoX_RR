import importlib
from pathlib import Path

package_dir = Path(__file__).parent

# import all cogkit to trigger register
for subdir1 in package_dir.iterdir():
    if not (subdir1.is_dir() and not subdir1.name.startswith("_")):
        continue

    for module_path in subdir1.glob("*.py"):
        module_name = module_path.stem
        full_module_name = f".{subdir1.name}.{module_name}"
        importlib.import_module(full_module_name, package=__name__)

    for subdir2 in subdir1.iterdir():
        if not (subdir2.is_dir() and not subdir2.name.startswith("_")):
            continue

        # CogView4 requires a newer Diffusers revision than the repository lock.
        if subdir1.name == "cogview" and subdir2.name == "cogview4":
            continue
        for module_path in subdir2.glob("*.py"):
            module_name = module_path.stem
            full_module_name = f".{subdir1.name}.{subdir2.name}.{module_name}"
            importlib.import_module(full_module_name, package=__name__)
