"""
Utilities
---------
"""

def progress_bar(current: int, total: int, feedback: str = "", width=20) -> None:
    """Prints a progress bar given a current epoch, total epochs and feedback."""
    print("Running...")
    progress = "|" * int(current / total * width + 1)
    bar = "-" * (width - len(progress))
    print(f"\r[{progress}{bar}] Epoch: {current+1} of {total} -> {feedback}", end="\r")
    if current + 1 == total:
        print("\n")
        print(f"\U0001F600 Done!")