"""Single source of truth for the ColorCast version string.

Every consumer that needs the version should import from here or from
``colorcast`` (which re-exports it).  Structured formats that cannot
import Python (``CITATION.cff``, ``.zenodo.json``) carry their own
copy; update all of them in sync when cutting a release.
"""

__version__ = "2.6.0"
