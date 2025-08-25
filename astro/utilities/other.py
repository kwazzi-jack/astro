def get_module_path(module_name: str, is_base: bool = False):
    """Get the module path relative to Astro."""
    idx = -2 if is_base else -1
    return ".".join(module_name.split(".")[:idx])
