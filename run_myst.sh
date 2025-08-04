commit b4e2e0e577faed5024e110df34e4dec8d75aaff3
Author: Richard Higgins <richard@relh.net>
Date:   Sun Aug 3 17:22:36 2025 -0700

    more ruff

diff --git a/run_myst.sh b/run_myst.sh
new file mode 100755
index 0000000..073c255
--- /dev/null
+++ b/run_myst.sh
@@ -0,0 +1,19 @@
+#!/bin/bash
+
+# Myst runner script - automatically sets up PYTHONPATH for VGGT
+
+# Get the directory where this script is located
+SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
+
+# Set up PYTHONPATH to include VGGT and virtual environment
+export PYTHONPATH="$SCRIPT_DIR/../vggt:$VIRTUAL_ENV/lib/python3.12/site-packages"
+
+# Check if VGGT directory exists
+if [ ! -d "$SCRIPT_DIR/../vggt" ]; then
+    echo "Warning: VGGT directory not found at $SCRIPT_DIR/../vggt"
+    echo "Please run ./setup_rtx5080_complete.sh first"
+    exit 1
+fi
+
+# Run the myst script with all arguments passed through
+python "$SCRIPT_DIR/run.py" "$@" 
\ No newline at end of file
