import sys

LINUX = sys.platform.startswith("linux")
MACOS = sys.platform.startswith("darwin")
WIN = sys.platform.startswith("win32") or sys.platform.startswith("cygwin")