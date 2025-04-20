import argparse
import collections
import configparser
import hashlib
from http.client import TOO_MANY_REQUESTS
import os
import re
import sys
import zlib

argparser = argparse.ArgumentParser(description="Content tracker")
argsubparsers = argparser.add_subparsers(title="Commands", dest="command")
argsubparsers.required = True
