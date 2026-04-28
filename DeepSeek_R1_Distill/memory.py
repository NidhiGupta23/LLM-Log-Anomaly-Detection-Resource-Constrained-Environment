import os
import psutil


def get_cpu_memory_mb() -> float:
    """RSS memory of the current process in megabytes."""
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2


def get_system_memory_percent() -> float:
    """System-wide RAM utilisation as a percentage."""
    return psutil.virtual_memory().percent

