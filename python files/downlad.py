import ssl
import urllib3

# Strip out the strict checking flag introduced in Python 3.13+
orig_create_context = urllib3.util.ssl_.create_urllib3_context

def loose_create_context(*args, **kwargs):
    ctx = orig_create_context(*args, **kwargs)
    # Remove the VERIFY_X509_STRICT flag (0x20)
    ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
    return ctx

urllib3.util.ssl_.create_urllib3_context = loose_create_context

# Now execute the spacy download natively
from spacy.cli import download
download("en_core_web_lg")
