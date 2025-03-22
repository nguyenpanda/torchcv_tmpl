import torch


def get_cpu_info():
    """Return a string with system CPU information, i.e. 'Apple M2'."""
    import cpuinfo  # pip install py-cpuinfo

    k = "brand_raw", "hardware_raw", "arch_string_raw"  # info keys sorted by preference (not all keys always available)
    info = cpuinfo.get_cpu_info()  # This function takes lots of time to complete
    string = info.get(k[0] if k[0] in info else k[1] if k[1] in info else k[2], "unknown")
    return string.replace("(R)", "").replace("CPU ", "").replace("@ ", "")


def auto_choice_device():
    return (
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else
        'cpu'
    )


__all__ = [
    'auto_choice_device',
    'get_cpu_info',
]
