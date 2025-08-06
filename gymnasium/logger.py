"""Set of functions for logging messages."""
from __future__ import annotations

import warnings

from gymnasium.utils import colorize
from gymnasium import error


WARN = 30
ERROR = 40

min_level = 30


# Ensure DeprecationWarning to be displayed (#2685, #3059)
warnings.filterwarnings("once", "", DeprecationWarning, module=r"^gymnasium\.")


def warn(
    msg: str,
    *args: object,
    category: type[Warning] | None = None,
    stacklevel: int = 1,
):
    """Raises a warning to the user if the min_level <= WARN.

    Args:
        msg: The message to warn the user
        *args: Additional information to warn the user
        category: The category of warning
        stacklevel: The stack level to raise to
    """
    if min_level <= WARN:
        # Avoid call to external colorize if avoidable (hot path); string format only if args
        s = msg % args if args else msg
        colored = f"\x1b[33mWARN: {s}\x1b[0m"
        warnings.warn(
            colored,
            category=category,
            stacklevel=stacklevel + 1,
        )


def deprecation(msg: str, *args: object):
    """Logs a deprecation warning to users."""
    warn(msg, *args, category=DeprecationWarning, stacklevel=2)


def error(msg: str, *args: object):
    """Logs an error message if min_level <= ERROR in red on the sys.stderr."""
    if min_level <= ERROR:
        warnings.warn(colorize(f"ERROR: {msg % args}", "red"), stacklevel=3)

def _optimized_check_version_exists(ns, name, version, version_cache):
    # Only invoked if env_spec is missing
    from gymnasium.envs.registration import (_check_name_exists, get_env_id,
                                             registry)
    env_full_id = get_env_id(ns, name, version)
    if env_full_id in registry:
        return

    _check_name_exists(ns, name)
    if version is None:
        return

    message = f"Environment version `v{version}` for environment `{get_env_id(ns, name, None)}` doesn't exist."
    # Use one registry scan instead of multiple
    env_specs = []
    default_spec = []
    found_versioned = []

    for env_spec in registry.values():
        if env_spec.namespace == ns and env_spec.name == name:
            env_specs.append(env_spec)
            if env_spec.version is None:
                default_spec.append(env_spec)
            else:
                found_versioned.append(env_spec)

    env_specs.sort(key=lambda env_spec: int(env_spec.version or -1))
    if default_spec:
        message += f" It provides the default version `{default_spec[0].id}`."
        if len(env_specs) == 1:
            raise error.DeprecatedEnv(message)

    if found_versioned:
        latest_spec = max(found_versioned, key=lambda env_spec: env_spec.version)
    else:
        latest_spec = None

    if latest_spec is not None and version > latest_spec.version:
        version_list_msg = ", ".join(f"`v{env_spec.version}`" for env_spec in env_specs)
        message += f" It provides versioned environments: [ {version_list_msg} ]."
        raise error.VersionNotFound(message)

    if latest_spec is not None and version < latest_spec.version:
        raise error.DeprecatedEnv(
            f"Environment version v{version} for `{get_env_id(ns, name, None)}` is deprecated. "
            f"Please use `{latest_spec.id}` instead."
        )
    # else: nothing else to raise
