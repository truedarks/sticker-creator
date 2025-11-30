"""
Local environment setup - forces use of local packages only.
This module should be imported first in all project scripts.
"""

import sys
import os
from pathlib import Path

def setup_local_environment():
    """Setup local environment and remove global site-packages from sys.path."""
    PROJECT_ROOT = Path(__file__).parent.absolute()
    LOCAL_SITE_PACKAGES = PROJECT_ROOT / "lib" / "site-packages"
    
    # Create local site-packages directory if it doesn't exist
    LOCAL_SITE_PACKAGES.mkdir(parents=True, exist_ok=True)
    
    local_path_str = str(LOCAL_SITE_PACKAGES)
    local_path_resolved = LOCAL_SITE_PACKAGES.resolve()
    
    # Remove global site-packages from sys.path to force local-only imports
    # Keep all non-site-packages paths (standard Python library paths)
    # Remove only global site-packages directories
    paths_to_keep = []
    
    for path in sys.path:
        path_str = str(path)
        path_obj = Path(path_str)
        
        # Keep if it's not a site-packages directory (standard Python paths)
        if 'site-packages' not in path_str:
            paths_to_keep.append(path)
        else:
            # It's a site-packages directory - check if it's our local one
            try:
                if path_obj.resolve() == local_path_resolved:
                    # It's our local site-packages - keep it
                    paths_to_keep.append(path)
                # Otherwise it's a global site-packages - remove it (don't add to keep list)
            except (OSError, ValueError):
                # Can't resolve path, but check if it matches our local path string
                if path_str == local_path_str:
                    paths_to_keep.append(path)
                # Otherwise skip (it's likely a global site-packages)
    
    # Replace sys.path with filtered paths
    sys.path = paths_to_keep
    
    # Add local site-packages at the beginning if not already there
    if local_path_str not in sys.path:
        sys.path.insert(0, local_path_str)
    
    return LOCAL_SITE_PACKAGES

# Auto-setup when imported
LOCAL_SITE_PACKAGES = setup_local_environment()

