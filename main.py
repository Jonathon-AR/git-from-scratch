import argparse
import collections
import configparser
import hashlib
import os
import re
import sys
import zlib
import datetime
from fnmatch import fnmatch

argparser = argparse.ArgumentParser(description="Content tracker")
argsubparsers = argparser.add_subparsers(title="Commands", dest="command")
argsubparsers.required = True


def main(argv=sys.argv[1:]):
    args = argparser.parse_args(argv)

    # Map command strings to functions
    commands = {
        "add": cmd_add,
        "cat-file": cmd_cat_file,
        "checkout": cmd_checkout,
        "commit": cmd_commit,
        "init": cmd_init,
        "log": cmd_log,
        "ls-tree": cmd_ls_tree,
        "rm": cmd_rm,
    }

    # Execute the corresponding command function or print error
    commands.get(args.command, lambda args: print(f"Unknown command: {args.command}"))(args)



class GitRepository(object):
    """Represents a Git repository."""

    worktree = None
    gitdir = None
    conf = None

    def __init__(self, path, force=False):
        self.worktree = path
        self.gitdir = os.path.join(path, ".git")

        # Check if .git directory exists unless forced
        if not (force or os.path.isdir(self.gitdir)):
            raise Exception("Not a Git repository %s" % path)

        # Read configuration file
        self.conf = configparser.ConfigParser()
        cf = repo_file(self, "config")

        if cf and os.path.exists(cf):
            self.conf.read([cf])
        elif not force:
            raise Exception("Configuration file missing")

        # Check repository format version unless forced
        if not force:
            vers = int(self.conf.get("core", "repositoryformatversion"))
            if vers != 0:
                raise Exception("Unsupported respository format version %s" % vers)


def repo_file(repo, *path, mkdir=False):
    """Get path to a file in the .git directory, optionally creating parent dirs."""
    # Check if parent directory exists
    if repo_dir(repo, *path[:-1], mkdir=mkdir):
        return repo_path(repo, *path)


def repo_path(repo, *path):
    """Get the full path from the repository root."""
    # Changed repo.path to repo.gitdir to be consistent with GitRepository structure
    return os.path.join(repo.gitdir, *path)


def repo_dir(repo, *path, mkdir=False):
    """Get path to a directory in the .git directory, optionally creating it."""
    path = repo_path(repo, *path)
    if os.path.exists(path):
        if os.path.isdir(path):
            return path
        else:
            raise Exception("Not a directory %s" % path)

    # Create directory if requested and doesn't exist
    if mkdir:
        os.makedirs(path)
        return path
    else:
        return None


def repo_create(path):
    """Create a new Git repository at the specified path."""

    repo = GitRepository(path, True) # Force creation

    # Validate or create the worktree directory
    if os.path.exists(repo.worktree):
        if not os.path.isdir(repo.worktree):
            raise Exception("%s is not a directory!" % repo.worktree)
        if os.listdir(repo.worktree):
            raise Exception("%s is not empty!" % repo.worktree)
    else:
        os.makedirs(repo.worktree)

    # Create essential .git subdirectories
    assert repo_dir(repo, "branches", mkdir=True)
    assert repo_dir(repo, "objects", mkdir=True)
    assert repo_dir(repo, "refs", "tags", mkdir=True)
    assert repo_dir(repo, "refs", "heads", mkdir=True)

    # Create .git/description file
    with open(repo_file(repo, "description"), "w") as f:
        f.write(
            "Unnamed repository; edit this file 'description' to name the repository.\n"
        )

    # Create .git/HEAD file pointing to the default branch
    with open(repo_file(repo, "HEAD"), "w") as f:
        f.write("ref: refs/heads/master\n")

    # Create .git/config file with default settings
    with open(repo_file(repo, "config"), "w") as f:
        config = repo_default_config()
        config.write(f)

    return repo


def repo_default_config():
    """Generate a default configuration for a new repository."""
    ret = configparser.ConfigParser()

    ret.add_section("core")
    ret.set("core", "repositoryformatversion", "0")
    ret.set("core", "filemode", "false") # Corresponds to `git config core.filemode`
    ret.set("core", "bare", "false") # Corresponds to `git config core.bare`

    return ret

# Parser for the 'init' command
argsp = argsubparsers.add_parser("init", help="Initialize a new, empty repository.")
argsp.add_argument(
    "path",
    metavar="directory",
    nargs="?", # Optional argument
    default=".", # Default to current directory
    help="Where to create the repository",
)


def cmd_init(args):
    """Handler for the 'init' command."""
    repo_create(args.path)


def repo_find(path=".", required=True):
    """Find the root of the repository containing the given path."""
    path = os.path.realpath(path) # Get absolute path

    # Check if .git directory exists in the current path
    if os.path.isdir(os.path.join(path, ".git")):
        return GitRepository(path)

    # Recursively check parent directories
    parent = os.path.realpath(os.path.join(path, ".."))

    if parent == path:
        # Base case: reached root directory
        if required:
            raise Exception("No git directory")
        else:
            return None
    # Recursive call to check parent
    return repo_find(parent, required)


class GitObject(object):
    """Base class for Git objects (blob, tree, commit, tag)."""
    repo = None

    def __init__(self, repo, data=None):
        self.repo = repo

        if data != None:
            self.deserialize(data)

    def serialize(self):
        """Serialize object data. Must be implemented by subclasses."""
        raise Exception("Unimplemented!")

    def deserialize(self, data):
        """Deserialize raw data into object. Must be implemented by subclasses."""
        raise Exception("Unimplemented!")


def object_read(repo, sha):
    """Read a Git object from the repository by its SHA-1 hash."""
    # Construct path to object file
    path = repo_file(repo, "objects", sha[0:2], sha[2:])

    with open(path, "rb") as f:
        raw = zlib.decompress(f.read()) # Decompress object data

        # Find header parts: format, size
        x = raw.find(b" ")
        fmt = raw[0:x]

        y = raw.find(b"\x00", x)
        size = int(raw[x+1:y].decode("ascii")) # Size is between space and null byte

        # Validate object size
        if size != len(raw) - y - 1:
            raise Exception("Malformed object {0}: bad length".format(sha))

        # Determine object type and instantiate corresponding class
        if fmt == b"commit":
            c = GitCommit
        elif fmt == b"tree":
            c = GitTree
        elif fmt == b"tag":
            c = GitTag
        elif fmt == b"blob":
            c = GitBlob
        else:
            raise Exception(
                "Unknown type {0} for object {1}".format(fmt.decode("ascii"), sha)
            )

        # Return initialized object with its data
        return c(repo, raw[y + 1 :])


def object_find(repo, name, fmt=None, follow=True):
    """Find an object's SHA-1 hash by name (short/long hash, tag, branch, HEAD)."""
    # This function is simplified in the original code, just returning the name.
    # A full implementation would resolve refs and short SHAs.
    # Keeping original behavior for now:
    return name
    # See object_resolve for a more complete resolution logic used later.


def object_write(obj, actually_write=True):
    """Write a Git object to the object store."""
    data = obj.serialize() # Get serialized object data
    # Construct object content with header
    result = obj.fmt + b" " + str(len(data)).encode() + b"\x00" + data
    # Calculate SHA-1 hash
    sha = hashlib.sha1(result).hexdigest()

    if actually_write:
        # Construct path for the object
        path = repo_file(obj.repo, "objects", sha[0:2], sha[2:], mkdir=actually_write) # Corrected path segment
        # Write compressed object data to file
        if path: # Ensure path was created/exists
             with open(path, "wb") as f:
                 f.write(zlib.compress(result))
    return sha


class GitBlob(GitObject):
    """Represents a Git blob object (file content)."""
    fmt = b"blob"

    def serialize(self):
        """Return raw blob data."""
        return self.blobdata

    def deserialize(self, data):
        """Store raw data."""
        self.blobdata = data

# Parser for the 'cat-file' command
argsp = argsubparsers.add_parser(
    "cat-file", help="Provide content of repository objects"
)
argsp.add_argument(
    "type",
    metavar="type",
    choices=["blob", "commit", "tag", "tree"],
    help="Specify the type",
)
argsp.add_argument("object", metavar="object", help="The object to display")


def cmd_cat_file(args):
    """Handler for the 'cat-file' command."""
    repo = repo_find()
    cat_file(repo, args.object, fmt=args.type.encode())


def cat_file(repo, obj, fmt=None):
    """Read and print the contents of a Git object."""
    obj = object_read(repo, object_find(repo, obj, fmt=fmt))
    # Print object's serialized content to standard output
    sys.stdout.buffer.write(obj.serialize())

# Parser for the 'hash-object' command (implicitly defined by arguments added to argsp)
# Note: The original code adds these arguments to the 'cat-file' parser (`argsp`),
# which seems incorrect. These likely belong to a separate 'hash-object' command parser.
# Assuming it's intended for a hash-object like functionality:
argsp_hash_object = argsubparsers.add_parser( # Create a separate parser for clarity
    "hash-object", help="Compute object ID and optionally creates a blob from a file"
)
argsp_hash_object.add_argument(
    "-t",
    metavar="type",
    dest="type",
    choices=["blob", "commit", "tag", "tree"],
    default="blob",
    help="Specify the type",
)
argsp_hash_object.add_argument(
    "-w",
    dest="write",
    action="store_true",
    help="Actually write the object into the database",
)
argsp_hash_object.add_argument("path", help="Read object from <file>")


def object_hash(fd, fmt, repo=None):
    """Calculate the SHA-1 hash of data from a file descriptor, optionally writing the object."""
    data = fd.read()

    # Create the appropriate GitObject based on format
    if fmt == b"commit":
        obj = GitCommit(repo, data)
    elif fmt == b"tree":
        obj = GitTree(repo, data)
    elif fmt == b"tag":
        obj = GitTag(repo, data)
    elif fmt == b"blob":
        obj = GitBlob(repo, data)
    else:
        raise Exception("Unknown type %s!" % fmt)

    # Write the object if repo is provided (implies -w flag was used)
    return object_write(obj, repo)


def kvlm_parse(raw, start=0, dct=None):
    """Parse key-value list with message format (used in commits and tags)."""
    if not dct:
        dct = collections.OrderedDict() # Use OrderedDict to preserve key order

    # Find separators for key, value, and newline
    spc = raw.find(b" ", start) # Space separates key from value
    nl = raw.find(b"\n", start) # Newline marks end of key-value pair or start of message

    # Base case: no more key-value pairs, the rest is the message
    if (spc < 0) or (nl < spc):
        assert nl == start
        dct[b""] = raw[start + 1 :]
        return dct

    # Extract key
    key = raw[start:spc]

    # Find the end of the value, handling continuation lines (starting with space)
    end = start
    while True:
        end = raw.find(b"\n", end + 1)
        if raw[end + 1] != ord(" "):
            break

    # Extract value and replace continuation line markers
    value = raw[spc + 1 : end].replace(b"\n ", b"\n")

    # Store key-value pair, handling multiple values for the same key (e.g., parent)
    if key in dct:
        if type(dct[key]) == list:
            dct[key].append(value)
        else:
            dct[key] = [dct[key], value] # Convert to list on second occurrence
    else:
        dct[key] = value

    # Recursively parse the rest of the data
    return kvlm_parse(raw, start=end + 1, dct=dct)


def kvlm_serialize(kvlm):
    """Serialize a key-value dictionary back into the kvlm format."""
    ret = b""

    # Iterate through keys in the dictionary
    for k in kvlm.keys():
        if k == b"": # Skip message body key
            continue
        val = kvlm[k]
        # Ensure value is a list for consistent processing
        if type(val) != list:
            val = [val]

        # Write each value for the key
        for v in val:
            # Add key, space, value (with newline replacements), and newline
            ret += k + b" " + (v.replace(b"\n", b"\n ")) + b"\n"

    # Append the message body (associated with the empty key) after a blank line
    ret += b"\n" + kvlm[b""]
    return ret


class GitCommit(GitObject):
    """Represents a Git commit object."""
    fmt = b"commit"

    def deserialize(self, data):
        """Parse commit data using kvlm format."""
        self.kvlm = kvlm_parse(data)

    def serialize(self):
        """Serialize commit data back into kvlm format."""
        return kvlm_serialize(self.kvlm)

# Parser for the 'log' command
argsp = argsubparsers.add_parser("log", help="Display history of a given commit.")
argsp.add_argument("commit", default="HEAD", nargs="?", help="Commit to start at.")


def cmd_log(args):
    """Handler for the 'log' command (outputs Graphviz format)."""
    repo = repo_find()

    print("digraph wyaglog{")
    print("  node [shape=rect]") # Improve node appearance
    log_graphviz(repo, object_find(repo, args.commit), set()) # Use the more robust object_find
    print("}")


def log_graphviz(repo, sha, seen):
    """Recursively generate Graphviz representation of commit history."""

    if sha in seen:
        return # Avoid cycles and redundant processing
    seen.add(sha)

    commit = object_read(repo, sha)
    # Basic commit info for label
    short_hash = sha[:7]
    message_summary = commit.kvlm.get(b"", b"").split(b'\n', 1)[0].decode('utf-8', 'replace')
    print(f'  c_{sha} [label="{short_hash}\\n{message_summary}"]') # Node definition with label

    # Check if commit object has 'parent' key
    if b"parent" not in commit.kvlm:
        # Base case: the initial commit.
        return

    parents = commit.kvlm[b"parent"]

    # Ensure parents is a list for iteration
    if type(parents) != list:
        parents = [parents]

    # Recursively process parent commits
    for p in parents:
        p_sha = p.decode("ascii")
        print("  c_{0} -> c_{1};".format(sha, p_sha)) # Edge from child to parent
        log_graphviz(repo, p_sha, seen)


class GitTreeLeaf(object):
    """Represents an entry (leaf) in a Git tree object."""
    def __init__(self, mode, path, sha):
        self.mode = mode # File mode (permissions and type)
        self.path = path # File or directory name
        self.sha = sha # SHA-1 hash of the object (blob or subtree)


def tree_parse_one(raw, start=0):
    """Parse a single entry from raw tree data."""
    # Find space separating mode and path
    x = raw.find(b" ", start)
    # Mode is typically 5 or 6 bytes (e.g., 40000 for tree, 100644 for blob)
    assert x - start == 5 or x - start == 6

    mode = raw[start:x]
    # Find null byte separating path and SHA
    y = raw.find(b"\x00", x)
    path = raw[x + 1 : y]

    # Read the 20-byte SHA-1 hash and convert to hex string
    sha_bytes = raw[y + 1 : y + 21]
    sha = sha_bytes.hex() # Use .hex() for direct conversion

    # Return the end position of this entry and the parsed GitTreeLeaf object
    return y + 21, GitTreeLeaf(mode, path, sha)


def tree_parse(raw):
    """Parse the entire raw data of a tree object."""
    pos = 0
    max_len = len(raw) # Renamed max to max_len
    ret = list()
    # Iterate through the raw data, parsing one entry at a time
    while pos < max_len:
        pos, data = tree_parse_one(raw, pos)
        ret.append(data)

    return ret


def tree_serialize(obj):
    """Serialize a GitTree object back into raw byte format."""
    # Sort items by path as Git does for canonical representation
    obj.items.sort(key=lambda i: i.path)

    ret = b""
    for i in obj.items:
        ret += i.mode
        ret += b" "
        ret += i.path
        ret += b"\x00"
        # Convert hex SHA string back to 20 bytes
        sha_bytes = bytes.fromhex(i.sha)
        ret += sha_bytes
    return ret


class GitTree(GitObject):
    """Represents a Git tree object (directory listing)."""
    fmt = b"tree"

    def deserialize(self, data):
        """Parse raw tree data into a list of GitTreeLeaf objects."""
        self.items = tree_parse(data)

    def serialize(self):
        """Serialize the list of tree entries back into raw byte format."""
        return tree_serialize(self)

# Parser for the 'ls-tree' command
argsp = argsubparsers.add_parser("ls-tree", help="Pretty-print a tree object.")
argsp.add_argument("object", help="The object (tree SHA or ref pointing to a tree) to show.")


def cmd_ls_tree(args):
    """Handler for the 'ls-tree' command."""
    repo = repo_find()
    # Find the tree object SHA
    obj_sha = object_find(repo, args.object, fmt=b"tree")
    obj = object_read(repo, obj_sha)

    # Print each entry in the tree
    for item in obj.items:
        # Read the object pointed to by the entry to determine its type
        target_obj = object_read(repo, item.sha) # Read the referenced object
        print(
            "{0} {1} {2}\t{3}".format(
                item.mode.decode("ascii"), # Mode (e.g., 100644, 040000)
                target_obj.fmt.decode("ascii"), # Type (blob, tree)
                item.sha, # SHA of the object
                item.path.decode("ascii"), # Path/name
            )
        )

# Parser for the 'checkout' command
argsp = argsubparsers.add_parser(
    "checkout", help="Checkout a commit or tree into the working directory." # Clarified help
)
argsp.add_argument("commit", help="The commit or tree to checkout.")
argsp.add_argument("path", help="The EMPTY directory to checkout into.") # Clarified target


def cmd_checkout(args):
    """Handler for the 'checkout' command."""
    repo = repo_find()

    # Find the object to check out
    obj_sha = object_find(repo, args.commit) # Find SHA first
    obj = object_read(repo, obj_sha)

    # If it's a commit, get the tree it points to
    if obj.fmt == b"commit":
        tree_sha = obj.kvlm[b"tree"].decode("ascii")
        obj = object_read(repo, tree_sha) # Now obj is the root tree
    elif obj.fmt != b"tree":
         raise Exception(f"Cannot checkout object of type {obj.fmt.decode('ascii')}")

    # Validate and prepare the target directory
    if os.path.exists(args.path):
        if not os.path.isdir(args.path):
            raise Exception("Target path {0} is not a directory!".format(args.path))
        if os.listdir(args.path):
            raise Exception("Target directory {0} is not empty!".format(args.path))
    else:
        os.makedirs(args.path)

    # Recursively checkout the tree contents
    tree_checkout(repo, obj, os.path.realpath(args.path)) # Use realpath for consistency


def tree_checkout(repo, tree, path):
    """Recursively write the contents of a tree object to the filesystem."""
    for item in tree.items:
        obj = object_read(repo, item.sha)
        dest = os.path.join(path, item.path.decode("utf8")) # Decode path for os.path.join

        if obj.fmt == b"tree":
            # Create subdirectory and recurse
            os.makedirs(dest, exist_ok=True) # Use exist_ok=True
            tree_checkout(repo, obj, dest)
        elif obj.fmt == b"blob":
            # Write blob content to file
            # Ensure parent directory exists before writing file
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as f:
                f.write(obj.blobdata)


def ref_resolve(repo, ref):
    """Resolve a Git reference (like HEAD or branch name) to a SHA-1 hash."""
    path_to_ref = repo_file(repo, ref) # Get path to ref file

    # Handle cases where ref file might not exist (e.g., new repo, detached HEAD)
    if not path_to_ref or not os.path.exists(path_to_ref):
        # Try common locations if initial ref path fails
        common_refs = [os.path.join("refs", "heads", ref), os.path.join("refs", "tags", ref)]
        for common_ref in common_refs:
            path_to_ref = repo_file(repo, common_ref)
            if path_to_ref and os.path.exists(path_to_ref):
                break # Found a potential ref file
        else: # If loop completes without finding a file
             # If it looks like a SHA already, return it (for detached HEAD)
             if re.match(r"^[0-9a-fA-F]{40}$", ref):
                 return ref
             raise Exception(f"Reference not found: {ref}")

    with open(path_to_ref, "r") as fp:
        data = fp.read().strip() # Read and strip whitespace/newline

    # Check if it's a symbolic ref (points to another ref)
    if data.startswith("ref: "):
        # Recursively resolve the target ref
        return ref_resolve(repo, data[5:])
    else:
        # It's a direct ref (contains a SHA)
        return data


class GitTag(GitCommit):
    """Represents a Git tag object (inherits from GitCommit for kvlm handling)."""
    # Although tags can be lightweight (just a ref) or annotated (an object),
    # this class represents annotated tags, which have a structure similar to commits.
    fmt = b'tag'

# Parser arguments for a 'tag' command (implicitly defined)
# These arguments seem intended for a `git tag` like command.
# Adding them to a new parser for clarity.
argsp_tag = argsubparsers.add_parser("tag", help="Create, list, delete or verify a tag object signed with GPG")
argsp_tag.add_argument("-a",
                   action="store_true",
                   dest="create_tag_object",
                   help="Make an unsigned, annotated tag object") # Clarified help
argsp_tag.add_argument("name",
                   nargs="?", # Optional: list tags if name is omitted
                   help="The new tag's name")
argsp_tag.add_argument("object",
                   default="HEAD",
                   nargs="?",
                   help="The object the new tag will point to (commit sha usually)")


def object_resolve(repo, name):
    """Resolve a name (SHA prefix, ref, etc.) to a list of potential full SHA-1 hashes."""
    candidates = list()
    hashRE = re.compile(r"^[0-9A-Fa-f]{4,40}$") # Regex for SHA-like strings (4 to 40 hex chars)

    if not name or not name.strip(): # Handle empty or whitespace-only names
        return None # Or perhaps raise an error? Returning None for now.

    name = name.strip() # Clean up name

    # Handle special case: HEAD
    if name == "HEAD":
        try:
            return [ref_resolve(repo, "HEAD")]
        except Exception:
            # HEAD might not exist yet (e.g., initial commit)
            # Or could be detached but ref_resolve failed.
            # Depending on desired behavior, might return empty list or re-raise.
            return [] # Return empty list if HEAD resolution fails

    # Check if the name looks like a hash
    if hashRE.match(name):
        name_lower = name.lower() # Use lowercase for comparisons
        if len(name_lower) == 40:
            # If it's a full 40-char hash, check if it exists
            obj_path = repo_file(repo, "objects", name_lower[:2], name_lower[2:])
            if obj_path and os.path.exists(obj_path):
                 return [name_lower]
            else:
                 return [] # Full hash provided but object doesn't exist

        # If it's a short hash (4 to 39 chars)
        prefix = name_lower[0:2]
        path = repo_dir(repo, "objects", prefix, mkdir=False) # Check if prefix directory exists
        if path:
            rem = name_lower[2:] # The rest of the short hash
            for f in os.listdir(path):
                if f.startswith(rem):
                    candidates.append(prefix + f) # Add full hash to candidates
        # Return list of candidates found for the short hash
        return candidates

    # If not HEAD or a hash, try resolving as a ref (branch or tag)
    try:
        # Try resolving directly first (e.g., user provided 'refs/heads/master')
        resolved_sha = ref_resolve(repo, name)
        if resolved_sha: candidates.append(resolved_sha)
    except Exception:
         # If direct resolution fails, try common locations
         refs_to_try = [
             f"refs/heads/{name}",
             f"refs/tags/{name}",
         ]
         for ref in refs_to_try:
             try:
                 resolved_sha = ref_resolve(repo, ref)
                 if resolved_sha: candidates.append(resolved_sha)
                 # Avoid adding duplicates if multiple refs point to the same SHA
                 candidates = list(set(candidates))
             except Exception:
                 continue # Ignore if a specific ref path doesn't resolve

    return candidates


# This is a more robust version of object_find, using object_resolve.
# It replaces the placeholder object_find defined earlier.
def object_find(repo, name, fmt=None, follow=True):
    """Resolve name to a single object SHA, optionally checking type and following tags."""
    shas = object_resolve(repo, name) # Get potential SHAs

    if not shas:
        raise Exception("No such reference {0}.".format(name))

    if len(shas) > 1:
        raise Exception(
            "Ambiguous reference {0}: Candidates are:\n - {1}.".format(name,  "\n - ".join(shas)))

    sha = shas[0] # Unique SHA found

    # If no format check is needed, return the SHA
    if not fmt:
        return sha

    # Loop to follow tags if necessary
    while True:
        obj = object_read(repo, sha)

        # Check if the object's format matches the expected format
        if obj.fmt == fmt:
            return sha

        # If we shouldn't follow tags, and it's not the right type, return None
        if not follow:
            return None

        # Follow annotated tags to the tagged object
        if obj.fmt == b'tag':
            # Ensure 'object' key exists in the tag's kvlm data
            if b'object' in obj.kvlm:
                 sha = obj.kvlm[b'object'].decode("ascii")
            else:
                 raise Exception(f"Malformed tag object {sha}: missing 'object' field.")
        # Follow commits to trees if tree format is requested
        elif obj.fmt == b'commit' and fmt == b'tree':
             # Ensure 'tree' key exists in the commit's kvlm data
             if b'tree' in obj.kvlm:
                 sha = obj.kvlm[b'tree'].decode("ascii")
             else:
                 raise Exception(f"Malformed commit object {sha}: missing 'tree' field.")
        else:
            # Object is not the desired type, and not a tag/commit we can follow
            return None

# Arguments for a 'rev-parse' like command (implicitly defined)
# Adding to a new parser for clarity.
argsp_rev_parse = argsubparsers.add_parser("rev-parse", help="Pick out and massage parameters")
argsp_rev_parse.add_argument("--wyag-type", # Using a custom prefix to avoid conflict
                   metavar="type",
                   dest="type",
                   choices=["blob", "commit", "tag", "tree"],
                   default=None,
                   help="Specify the expected type")
argsp_rev_parse.add_argument("name",
                   help="The object name to parse")


class GitIndexEntry (object):
    """Represents a single entry in the Git index (staging area)."""
    def __init__(self, ctime=None, mtime=None, dev=None, ino=None,
                 mode_type=None, mode_perms=None, uid=None, gid=None,
                 fsize=None, sha=None, flag_assume_valid=None,
                 flag_stage=None, name=None):
        # Metadata from stat: creation time (seconds, nanoseconds)
        self.ctime = ctime
        # Metadata from stat: modification time (seconds, nanoseconds)
        self.mtime = mtime
        # Metadata from stat: device ID
        self.dev = dev
        # Metadata from stat: inode number
        self.ino = ino
        # File mode: type (regular, symlink, gitlink) encoded as bits
        self.mode_type = mode_type
        # File mode: permissions (e.g., 644, 755) as integer
        self.mode_perms = mode_perms
        # Metadata from stat: user ID of owner
        self.uid = uid
        # Metadata from stat: group ID of owner
        self.gid = gid
        # Metadata from stat: file size in bytes
        self.fsize = fsize
        # SHA-1 hash of the blob object representing the file content
        self.sha = sha
        # Flag: indicates if the file is assumed to be unchanged (optimization)
        self.flag_assume_valid = flag_assume_valid
        # Flag: indicates merge stage (0 for normal, 1-3 during merge conflicts)
        self.flag_stage = flag_stage
        # Full path of the file relative to the worktree root
        self.name = name

class GitIndex (object):
    """Represents the Git index file (staging area)."""
    version = None # Index format version (usually 2)
    entries = [] # List of GitIndexEntry objects
    # Extensions and checksum data are part of v3/v4 index, not handled here.
    # ext = None
    # sha = None

    def __init__(self, version=2, entries=None):
        if not entries:
            entries = list() # Initialize with empty list if none provided

        self.version = version
        self.entries = entries


def index_read(repo):
    """Read the index file from the repository."""
    index_file = repo_file(repo, "index")

    # Handle case where index doesn't exist (e.g., new repo)
    if not os.path.exists(index_file):
        return GitIndex() # Return an empty index object

    with open(index_file, 'rb') as f:
        raw = f.read()

    # Read header (12 bytes)
    header = raw[:12]
    signature = header[:4]
    assert signature == b"DIRC", "Invalid index signature" # "DIRC" for DirCache
    version = int.from_bytes(header[4:8], "big")
    assert version == 2, "wyag only supports index file version 2"
    count = int.from_bytes(header[8:12], "big") # Number of entries

    entries = list()
    content = raw[12:] # Start reading entry data after header
    idx = 0 # Current position in the content buffer
    for i in range(0, count):
        # Read metadata fields (timestamps, device, inode, mode, uid, gid, size)
        # Each field has a fixed size as per the index format specification.
        ctime_s =  int.from_bytes(content[idx: idx+4], "big")
        ctime_ns = int.from_bytes(content[idx+4: idx+8], "big")
        mtime_s = int.from_bytes(content[idx+8: idx+12], "big")
        mtime_ns = int.from_bytes(content[idx+12: idx+16], "big")
        dev = int.from_bytes(content[idx+16: idx+20], "big")
        ino = int.from_bytes(content[idx+20: idx+24], "big")

        # Mode (32 bits, but only 16 are used in v2)
        mode_bytes = content[idx+24: idx+28] # Read 4 bytes for mode
        mode = int.from_bytes(mode_bytes[0:4], "big") # Use all 4 bytes

        # The high 16 bits contain the object type (e.g., 0x8000 for regular file)
        # Correction: Mode is 16 bits in v2 format stored in bytes 26-28 (big endian)
        mode_v2 = int.from_bytes(content[idx+26: idx+28], "big")
        mode_type_v2 = mode_v2 >> 12 # Type is in the top 4 bits (1000=reg, 1010=symlink, 1110=gitlink)
        assert mode_type_v2 in [0b1000, 0b1010, 0b1110], f"Unknown mode type: {mode_type_v2:b}"
        # The low 9 bits contain standard Unix permissions.
        mode_perms_v2 = mode_v2 & 0o777 # Mask for permission bits

        uid = int.from_bytes(content[idx+28: idx+32], "big")
        gid = int.from_bytes(content[idx+32: idx+36], "big")
        fsize = int.from_bytes(content[idx+36: idx+40], "big")

        # Read SHA-1 hash (20 bytes) and format as hex string
        sha = content[idx+40: idx+60].hex()

        # Read flags (16 bits)
        flags = int.from_bytes(content[idx+60: idx+62], "big")
        # Parse flags: assume-valid, extended (must be 0 in v2), stage
        flag_assume_valid = (flags & 0b1000000000000000) != 0
        flag_extended = (flags & 0b0100000000000000) != 0
        assert not flag_extended, "Extended flag set in v2 index is not supported"
        flag_stage =  (flags & 0b0011000000000000) >> 12 # Extract stage bits

        # Name length is stored in the low 12 bits of the flags
        name_length = flags & 0b0000111111111111

        # Advance index past the fixed-size part of the entry (62 bytes)
        idx += 62

        # Read the entry name (variable length)
        if name_length < 0xFFF: # Normal case: length fits in 12 bits
            # Read name bytes and check for null terminator
            raw_name = content[idx : idx + name_length]
            # Ensure null terminator exists right after the name
            if idx + name_length >= len(content) or content[idx + name_length] != 0x00:
                 raise Exception("Index entry format error: Missing null terminator or invalid name length.")
            idx += name_length + 1 # Advance past name and null byte
        else: # Special case: name length is >= 0xFFF
            # Find the null terminator after the 0xFFF position
            null_idx = content.find(b'\x00', idx + 0xFFF)
            if null_idx == -1:
                 raise Exception("Index entry format error: Long name missing null terminator.")
            raw_name = content[idx: null_idx]
            idx = null_idx + 1 # Advance past name and null byte

        # Decode name from bytes (assuming UTF-8)
        name = raw_name.decode("utf8")


        # Simplified padding calculation: advance idx to next multiple of 8
        idx = 8 * math.ceil(idx / 8) # Ceiling division trick

        # Create GitIndexEntry object and add to list
        entries.append(GitIndexEntry(ctime=(ctime_s, ctime_ns),
                                     mtime=(mtime_s,  mtime_ns),
                                     dev=dev,
                                     ino=ino,
                                     mode_type=mode_type_v2, # Use corrected v2 mode type
                                     mode_perms=mode_perms_v2, # Use corrected v2 perms
                                     uid=uid,
                                     gid=gid,
                                     fsize=fsize,
                                     sha=sha,
                                     flag_assume_valid=flag_assume_valid,
                                     flag_stage=flag_stage,
                                     name=name))

    return GitIndex(version=version, entries=entries)

# Parser for 'ls-files' command
argsp = argsubparsers.add_parser("ls-files", help = "List files in the index (staging area).")
argsp.add_argument("--verbose", "-v", action="store_true", help="Show detailed information for each file.") # Added -v alias


def cmd_ls_files(args):
    """Handler for the 'ls-files' command."""
    repo = repo_find()
    index = index_read(repo)
    if args.verbose:
        print(f"Index file format v{index.version}, containing {len(index.entries)} entries.")

    for e in index.entries:
        print(e.name)
        if args.verbose:
            entry_type = { 0b1000: "regular file",
                           0b1010: "symlink",
                           0b1110: "git link" }[e.mode_type]
            print(f"  {entry_type} with perms: {e.mode_perms:o}")
            print(f"  on blob: {e.sha}")
            print(f"  created: {datetime.fromtimestamp(e.ctime[0])}.{e.ctime[1]}, modified: {datetime.fromtimestamp(e.mtime[0])}.{e.mtime[1]}")
            print(f"  device: {e.dev}, inode: {e.ino}")
            print(f"  user: {pwd.getpwuid(e.uid).pw_name} ({e.uid})  group: {grp.getgrgid(e.gid).gr_name} ({e.gid})")
            print(f"  flags: stage={e.flag_stage} assume_valid={e.flag_assume_valid}")

argsp = argsubparsers.add_parser("check-ignore", help = "Check path(s) against ignore rules.")
argsp.add_argument("path", nargs="+", help="Paths to check relative to repository root.")


def cmd_check_ignore(args):
    """Handler for the 'check-ignore' command."""
    repo = repo_find()
    rules = gitignore_read(repo) # Load ignore rules
    for path in args.path:
        if check_ignore(rules, path):
            print(path) # Print the original path provided by the user


def gitignore_parse1(raw):
    """Parse a single line from a .gitignore file."""
    raw = raw.strip() # Remove leading/trailing whitespace

    if not raw or raw[0] == "#":
        return None
    elif raw[0] == "!":
        return (raw[1:], False)
    elif raw[0] == "\\":
        return (raw[1:], True)
    else:
        return (raw, True)

def gitignore_parse(lines):
    """Parse multiple lines into a list of (pattern, ignore_flag) tuples."""
    ret = list()
    for line in lines:
        parsed = gitignore_parse1(line)
        if parsed:
            ret.append(parsed)
    # Reverse the list so later patterns take precedence (as per Git behavior)
    return list(reversed(ret))


class GitIgnore(object):
    """Stores parsed ignore rules from various sources."""
    # Rules that apply repository-wide (e.g., .git/info/exclude, global gitignore)
    absolute = None
    # Rules specific to directories (from .gitignore files found in the repo)
    # Key: directory path relative to repo root (POSIX style), Value: list of rules
    scoped = None

    def __init__(self, absolute, scoped):
        self.absolute = absolute
        self.scoped = scoped

def gitignore_read(repo):
    """Read ignore rules from all relevant sources (.git/info/exclude, global, .gitignore files)."""
    rules = GitIgnore()

    # 1. Read repo-specific excludes: .git/info/exclude
    exclude_file = os.path.join(repo.gitdir, "info", "exclude")
    if os.path.exists(exclude_file):
        with open(exclude_file, "r") as f:
            rules.absolute.append(gitignore_parse(f.readlines()))

    # 2. Read global gitignore (if configured and exists)
    # This part is simplified; a full implementation would check git config core.excludesfile
    # Using common locations as a fallback:
    config_home = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
    global_files = [
        os.path.join(config_home, "git", "ignore"), # XDG standard location
        os.path.expanduser("~/.gitignore") # Common user global ignore
    ]
    for global_file in global_files:
        if os.path.exists(global_file):
            try:
                with open(global_file, "r") as f:
                    rules.absolute.append(gitignore_parse(f.readlines()))
            except IOError as e:
                 print(f"Warning: Could not read global ignore file {global_file}: {e}", file=sys.stderr)


    # 3. Read .gitignore files from the index (committed .gitignore files)
    index = index_read(repo)
    gitignore_entries = {} # Store sha and path for reading later
    for entry in index.entries:
        # Check for .gitignore files at any level
        if entry.name == ".gitignore" or entry.name.endswith("/.gitignore"):
            # Store SHA to read content later, avoid reading duplicates if path is same
            gitignore_entries[entry.name] = entry.sha

    # Read content of found .gitignore files
    for name, sha in gitignore_entries.items():
        try:
            contents = object_read(repo, sha)
            if contents.fmt == b'blob':
                lines = contents.blobdata.decode("utf8").splitlines()
                # Directory containing this .gitignore, relative to repo root (POSIX path)
                dir_name = os.path.dirname(name).replace(os.sep, '/')
                # Store parsed rules, associated with their directory
                rules.scoped[dir_name] = gitignore_parse(lines)
            else:
                 print(f"Warning: Entry for .gitignore '{name}' is not a blob (type: {contents.fmt.decode()}). Skipping.", file=sys.stderr)
        except Exception as e:
             print(f"Warning: Could not read or parse .gitignore object {sha} for path '{name}': {e}", file=sys.stderr)

    return rules


def check_ignore1(ruleset, path):
    """Check a path against a single list of parsed rules (pattern, ignore_flag)."""
    # path is assumed to be POSIX style, relative to the location of the ruleset
    # (e.g., relative to repo root for absolute rules, relative to dir for scoped rules)
    match = None # Store the result of the last matching pattern

    # Iterate through rules (already reversed, so later rules have higher precedence)
    for (pattern, ignore_flag) in ruleset:
        # fnmatch handles basic globbing (*, ?, [])
        # Need to handle Git-specific patterns: **, trailing /, dir/ pattern
        # Simplified matching using fnmatch for now:

        # Handle directory-only patterns (ending with /)
        is_dir_pattern = pattern.endswith('/')
        if is_dir_pattern:
            pattern = pattern.rstrip('/') # Remove trailing slash for matching
            # This pattern should only match directories. We don't have that info here easily.
            # Approximation: If path doesn't look like a dir, skip dir pattern? Risky.
            # Git checks if the path *is* a directory. We'll ignore this check for simplicity.

        # Handle matching relative to the ruleset's location
        # For scoped rules, path should be relative to the directory containing .gitignore
        # For absolute rules, path is relative to repo root.
        # The `path` argument passed here needs to be adjusted accordingly before calling.

        # Basic fnmatch check
        if fnmatch(path, pattern):
            match = ignore_flag # Update match result
            continue # Continue checking subsequent rules for overrides

        # Handle patterns without '/' - match in any directory (if not absolute path pattern)
        # fnmatch already handles this if pattern has no '/'

        # Handle '**/pattern' - match in any subdirectory
        if pattern.startswith('**/'):
             base_pattern = pattern[3:]
             # Check if path ends with the base pattern after some directory components
             if fnmatch(os.path.basename(path), base_pattern) or \
                any(fnmatch(part, base_pattern) for part in path.split('/')): # Simplified check
                 match = ignore_flag
                 continue

        # Handle 'dir/pattern' - match relative to the ruleset location
        # fnmatch handles this correctly if the `path` passed is relative to the ruleset dir.

    return match # Return the last match found (or None if no match)


def check_ignore_scoped(scoped_rules, path):
    """Check path against .gitignore rules found in parent directories."""
    # path is POSIX style, relative to repo root
    result = None
    current_dir = os.path.dirname(path) # Get directory containing the path

    while True:
        # Check if rules exist for the current directory level
        if current_dir in scoped_rules:
            # Calculate path relative to the directory where .gitignore resides
            relative_path = os.path.relpath(path, current_dir if current_dir else '.')
            relative_path = relative_path.replace(os.sep, '/') # Ensure POSIX path

            # Check against rules for this level
            level_result = check_ignore1(scoped_rules[current_dir], relative_path)
            if level_result is not None:
                result = level_result # Update result if a match was found

        # Move to parent directory
        if current_dir == "": # Reached repo root
            break
        parent_dir = os.path.dirname(current_dir)
        # Handle edge case where dirname of root is root itself
        if parent_dir == current_dir:
             break
        current_dir = parent_dir

    return result # Return the most specific match found (or None)


def check_ignore_absolute(absolute_rulesets, path):
    """Check path against global/absolute ignore rules."""
    # path is POSIX style, relative to repo root
    result = None
    # Iterate through absolute rulesets (e.g., from info/exclude, global file)
    for ruleset in absolute_rulesets:
        ruleset_result = check_ignore1(ruleset, path)
        if ruleset_result is not None:
            result = ruleset_result # Update result if match found

    # Default to not ignored if no absolute rule matched
    return result if result is not None else False


def check_ignore(rules, path):
    """Determine if a path is ignored based on all loaded rules."""
    # path is assumed to be POSIX style, relative to the repository root.
    # os.path.isabs check removed as path should always be relative here.

    # 1. Check scoped rules (.gitignore files) from deepest to shallowest.
    #    A match here (ignore or un-ignore) takes precedence over shallower .gitignores
    #    and absolute rules, unless overridden by a deeper rule.
    scoped_result = check_ignore_scoped(rules.scoped, path)

    # If a scoped rule determined the status (ignored=True or unignored=False), return it.
    if scoped_result is not None:
        return scoped_result

    # 2. If no scoped rule matched, check absolute rules (info/exclude, global).
    absolute_result = check_ignore_absolute(rules.absolute, path)

    # Return the result from absolute rules (which defaults to False if no match).
    return absolute_result

# Parser for 'status' command
argsp = argsubparsers.add_parser("status", help = "Show the working tree status.")


def cmd_status(_): # Argument 'args' is unused, replaced with '_'
    """Handler for the 'status' command."""
    repo = repo_find()
    index = index_read(repo) # Read the current index

    # Print status sections
    cmd_status_branch(repo) # Current branch / HEAD state
    print("\nChanges to be committed:") # Staged changes (HEAD vs Index)
    cmd_status_head_index(repo, index)
    print("\nChanges not staged for commit:") # Unstaged changes (Index vs Worktree)
    cmd_status_index_worktree(repo, index)
    print("\nUntracked files:") # Files in worktree not in index and not ignored
    cmd_status_untracked(repo, index) # Separated untracked logic


def branch_get_active(repo):
    """Get the name of the currently active branch, or False if HEAD is detached."""
    try:
        head_ref_path = repo_file(repo, "HEAD")
        if not head_ref_path or not os.path.exists(head_ref_path):
             return False # Should not happen in a valid repo, but handle defensively

        with open(head_ref_path, "r") as f:
            head = f.read().strip() # Read HEAD content

        # Check if HEAD is a symbolic ref to a branch
        if head.startswith("ref: refs/heads/"):
            return head[16:] # Extract branch name
        else:
            # HEAD contains a SHA, so it's detached
            return False
    except Exception as e:
         print(f"Error reading HEAD: {e}", file=sys.stderr)
         return False # Return False on error


def cmd_status_branch(repo):
    """Print the current branch or detached HEAD status."""
    branch = branch_get_active(repo)
    if branch:
        print(f"On branch {branch}.")
    else:
        print(f"HEAD detached at {object_find(repo, 'HEAD')}")

def tree_to_dict(repo, ref, prefix=""):
    """Convert a tree object (recursively) into a dictionary {path: sha}."""
    ret = dict()
    tree_sha = object_find(repo, ref, fmt=b"tree")
    tree = object_read(repo, tree_sha)

    for leaf in tree.items:
        full_path = os.path.join(prefix, leaf.path)

            # Check if the leaf represents a subtree (directory)
            # Mode '040000' indicates a tree
        is_subtree = leaf.mode.startswith(b'04')

        if is_subtree:
            # Recursively process the subtree
            ret.update(tree_to_dict(repo, leaf.sha, full_path))
        else:
            # Store the blob SHA for the file path
            ret[full_path] = leaf.sha
    return ret


def cmd_status_head_index(repo, index):
    """Compare HEAD tree with the index to find staged changes."""
    print("Changes to be committed:")
    head_tree_dict = tree_to_dict(repo, "HEAD") # Get {path: sha} for HEAD
    index_dict = {entry.name.replace(os.sep, '/'): entry.sha for entry in index.entries} # {path: sha} for index

    staged_files = set(index_dict.keys())
    head_files = set(head_tree_dict.keys())

    # Added files: in index but not in HEAD
    added = staged_files - head_files
    for path in sorted(list(added)):
        print(f"  (use \"git rm --cached <file>...\" to unstage)") # Git's hint
        print(f"\tnew file:   {path}")

    # Deleted files: in HEAD but not in index
    deleted = head_files - staged_files
    for path in sorted(list(deleted)):
        print(f"  (use \"git add <file>...\" to update what will be committed)") # Git's hint
        print(f"\tdeleted:    {path}")

    # Modified files: in both, but SHAs differ
    modified = set()
    for path in head_files.intersection(staged_files):
        if head_tree_dict[path] != index_dict[path]:
            modified.add(path)
    for path in sorted(list(modified)):
        print(f"  (use \"git add <file>...\" to update what will be committed)") # Git's hint
        print(f"\tmodified:   {path}")

    if not added and not deleted and not modified:
        print("  (no changes added to commit)")


def cmd_status_index_worktree(repo, index):
    print("Changes not staged for commit:")

    ignore = gitignore_read(repo)

    gitdir_prefix = repo.gitdir + os.path.sep

    all_files = list()

    # We begin by walking the filesystem
    for (root, _, files) in os.walk(repo.worktree, True):
        if root==repo.gitdir or root.startswith(gitdir_prefix):
            continue

        for f in files:
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, repo.worktree)
            all_files.append(rel_path)


    for entry in index.entries:
        full_path = os.path.join(repo.worktree, entry.name)

        # That file *name* is in the index

        if not os.path.exists(full_path):
            print("  deleted: ", entry.name)
        else:
            stat = os.stat(full_path)

            # Compare metadata
            ctime_ns = entry.ctime[0] * 10**9 + entry.ctime[1]
            mtime_ns = entry.mtime[0] * 10**9 + entry.mtime[1]
            if (stat.st_ctime_ns != ctime_ns) or (stat.st_mtime_ns != mtime_ns):
                # If different, deep compare.
                # @FIXME This *will* crash on symlinks to dir.
                with open(full_path, "rb") as fd:
                    new_sha = object_hash(fd, b"blob", None)
                    # If the hashes are the same, the files are actually the same.
                    same = entry.sha == new_sha

                    if not same:
                        print("  modified:", entry.name)

        if entry.name in all_files:
            all_files.remove(entry.name)

    print()
    print("Untracked files:")

    for f in all_files:
        if not check_ignore(ignore, f):
            print(" ", f)

def index_write(repo, index):
    """Write the GitIndex object back to the index file."""
    index_file = repo_file(repo, "index")

    # Create parent directory if it doesn't exist (shouldn't be needed normally)
    os.makedirs(os.path.dirname(index_file), exist_ok=True)

    # Sort entries before writing for canonical format
    index.entries.sort(key=lambda e: (e.name, e.flag_stage))

    with open(index_file, "wb") as f:
        # Write Header
        f.write(b"DIRC") # Signature
        f.write(index.version.to_bytes(4, "big")) # Version (e.g., 2)
        f.write(len(index.entries).to_bytes(4, "big")) # Number of entries

        # Write Entries
        packed_entries = b""
        for e in index.entries:
            entry_data = b""
            # Timestamps (ctime, mtime) - 8 bytes each
            entry_data += e.ctime[0].to_bytes(4, "big") + e.ctime[1].to_bytes(4, "big")
            entry_data += e.mtime[0].to_bytes(4, "big") + e.mtime[1].to_bytes(4, "big")
            # Device and Inode - 4 bytes each
            entry_data += e.dev.to_bytes(4, "big")
            entry_data += e.ino.to_bytes(4, "big")
            # Mode (Type + Permissions) - 4 bytes (only 2 used in v2)
            # Reconstruct the 16-bit mode value for v2
            mode_v2 = (e.mode_type << 12) | e.mode_perms
            entry_data += (0).to_bytes(2, "big") # Padding for 32-bit mode field
            entry_data += mode_v2.to_bytes(2, "big") # Write 16-bit mode
            # UID and GID - 4 bytes each
            entry_data += e.uid.to_bytes(4, "big")
            entry_data += e.gid.to_bytes(4, "big")
            # File Size - 4 bytes
            entry_data += e.fsize.to_bytes(4, "big")
            # SHA-1 Hash - 20 bytes
            entry_data += bytes.fromhex(e.sha)
            # Flags (Assume-valid, Stage) + Name Length - 2 bytes
            flag_assume_valid_bit = 0x8000 if e.flag_assume_valid else 0
            stage_bits = (e.flag_stage & 0x3) << 12 # Stage is 2 bits (0-3) shifted
            name_bytes = e.name.encode("utf8")
            name_length = len(name_bytes)
            # Clamp name length if it exceeds max value for 12 bits (0xFFF)
            flags_name_len = min(name_length, 0xFFF)
            # Combine flags and name length
            flags_combined = flag_assume_valid_bit | stage_bits | flags_name_len
            entry_data += flags_combined.to_bytes(2, "big")

            # Name (variable length) + Null Terminator
            entry_data += name_bytes
            entry_data += b"\x00"

            # Padding: Ensure total entry length is multiple of 8 bytes
            entry_len_unpadded = 62 + len(name_bytes) + 1
            pad_len = (8 - (entry_len_unpadded % 8)) % 8
            entry_data += b"\x00" * pad_len

            packed_entries += entry_data

        # Write all packed entries
        f.write(packed_entries)

        # Calculate and write checksum
        # Checksum is SHA-1 of header + all packed entries
        content_to_checksum = f.getvalue() # Get bytes written so far
        checksum = hashlib.sha1(content_to_checksum).digest() # Calculate SHA-1 hash (digest gives bytes)
        f.write(checksum) # Append 20-byte checksum


# Parser for 'rm' command
argsp = argsubparsers.add_parser("rm", help="Remove files from the working tree and the index.")
argsp.add_argument("path", nargs="+", help="Files to remove.")
# Add --cached option like git rm
argsp.add_argument("--cached", action="store_true", help="Only remove from the index, keep worktree file.")
# Add -f/--force option (though not fully implemented below)
argsp.add_argument("-f", "--force", action="store_true", help="Override checks (not fully implemented).")


def cmd_rm(args):
    """Handler for the 'rm' command."""
    repo = repo_find()
    # Pass the 'cached' flag to the rm function
    rm(repo, args.path, delete=not args.cached, force=args.force)


def rm(repo, paths, delete=True, skip_missing=False, force=False):
    """Remove files from the index and optionally the working tree."""
    index = index_read(repo) # Read current index
    worktree_root = repo.worktree + os.sep # Ensure trailing separator

    # Normalize provided paths and check they are within the worktree
    paths_to_remove = set() # Store relative POSIX paths
    for path in paths:
        abspath = os.path.abspath(path)
        if not abspath.startswith(worktree_root):
            # Allow removing files outside worktree if forced? Git doesn't usually.
            # Sticking to worktree-only for now.
            raise Exception(f"Cannot remove path outside worktree: {path}")
        relpath = os.path.relpath(abspath, repo.worktree)
        posix_path = relpath.replace(os.sep, '/')
        paths_to_remove.add(posix_path)

    kept_entries = [] # Entries to keep in the new index
    removed_from_index = set() # Track which paths were actually found and removed
    files_to_delete = [] # Store full paths of files to delete from worktree

    # Iterate through index entries
    for entry in index.entries:
        entry_posix_path = entry.name.replace(os.sep, '/')
        if entry_posix_path in paths_to_remove:
            # Found an entry matching a path to remove
            removed_from_index.add(entry_posix_path)
            if delete: # If worktree deletion is requested
                full_path = os.path.join(repo.worktree, entry.name)
                files_to_delete.append(full_path)
            # Don't add this entry to kept_entries
        else:
            # Keep this entry
            kept_entries.append(entry)

    # Check if any requested paths were not found in the index
    not_found = paths_to_remove - removed_from_index
    if not_found and not skip_missing and not force:
        # Git error message: fatal: pathspec '...' did not match any files
        raise Exception(f"pathspec '{', '.join(sorted(list(not_found)))}' did not match any files")

    # Physically delete files from the working tree if requested
    if delete:
        deleted_count = 0
        errors = []
        for file_path in files_to_delete:
            try:
                if os.path.exists(file_path) or os.path.islink(file_path): # Check existence before unlinking
                     os.unlink(file_path)
                     deleted_count += 1
            except OSError as e:
                 # Collect errors instead of failing immediately
                 errors.append(f"Could not remove '{file_path}': {e}")
            # Git also handles directory removal if pathspec matches dir, more complex.

        if errors:
             # Report errors after attempting all deletions
             raise Exception("Errors occurred during file removal:\n" + "\n".join(errors))


    # Update the index with the kept entries and write it back
    if removed_from_index: # Only write if changes were made
         index.entries = kept_entries
         index_write(repo, index)
    # else: No matching entries found, index remains unchanged.


# Parser for 'add' command
argsp = argsubparsers.add_parser("add", help = "Add file contents to the index.")
argsp.add_argument("path", nargs="+", help="Files or directories to add.")


def cmd_add(args):
    """Handler for the 'add' command."""
    repo = repo_find()
    add(repo, args.path)


def add(repo, paths, delete=True, skip_missing=False):
    """Add file contents to the index (staging area)."""
    # This function reuses args from 'rm', which isn't quite right for 'add'.
    # 'delete' and 'skip_missing' are not typically relevant for 'add'.
    # Refactoring to focus on 'add' logic.

    repo = repo_find() # Ensure we have the repo object
    index = index_read(repo) # Read the current index
    index_entries = {e.name: e for e in index.entries} # Map name to entry for quick lookup/update

    worktree_root = repo.worktree + os.sep
    ignore_rules = gitignore_read(repo) # Load ignore rules

    files_to_add = set() # Collect relative POSIX paths of files to potentially add/update

    # Expand provided paths (could be files or directories)
    for path_spec in paths:
        abspath_spec = os.path.abspath(path_spec)

        # Check if path exists
        if not os.path.exists(abspath_spec):
             raise Exception(f"pathspec '{path_spec}' did not match any files")

        # Check if path is within worktree
        if not abspath_spec.startswith(worktree_root) and not os.path.commonpath([abspath_spec, worktree_root]) == worktree_root:
             raise Exception(f"path '{path_spec}' is outside repository")

        # If it's a directory, walk it
        if os.path.isdir(abspath_spec):
            for root, dirs, files in os.walk(abspath_spec, topdown=True):
                 # Prune ignored directories similar to status command
                 abs_root = os.path.abspath(root)
                 if abs_root.startswith(repo.gitdir): # Skip .git dir contents
                      dirs[:] = []
                      continue

                 dirs_to_remove = []
                 for d in dirs:
                     dir_path_full = os.path.join(root, d)
                     dir_path_rel = os.path.relpath(dir_path_full, repo.worktree)
                     posix_dir_path = dir_path_rel.replace(os.sep, '/') + '/'
                     if check_ignore(ignore_rules, posix_dir_path):
                         dirs_to_remove.append(d)
                 for d in dirs_to_remove:
                     dirs.remove(d)

                 # Process files in the directory
                 for f in files:
                     file_path_full = os.path.join(root, f)
                     file_path_rel = os.path.relpath(file_path_full, repo.worktree)
                     posix_file_path = file_path_rel.replace(os.sep, '/')
                     # Add file if not ignored
                     if not check_ignore(ignore_rules, posix_file_path):
                         files_to_add.add(posix_file_path)

        # If it's a file
        elif os.path.isfile(abspath_spec):
            relpath = os.path.relpath(abspath_spec, repo.worktree)
            posix_path = relpath.replace(os.sep, '/')
            # Add file if not ignored
            if not check_ignore(ignore_rules, posix_path):
                files_to_add.add(posix_path)
        # else: Could be symlink, block device etc. - Git handles these, we might ignore or error.


    # Process the collected files
    updated = False
    new_entries_list = list(index.entries) # Start with existing entries

    # Remove existing entries that match files_to_add to handle updates correctly
    current_entries_dict = {e.name: e for e in new_entries_list}
    new_entries_list = [e for e in new_entries_list if e.name not in files_to_add]


    for relpath in sorted(list(files_to_add)): # Process in sorted order
        abspath = os.path.join(repo.worktree, relpath.replace('/', os.sep))

        # Skip files that don't exist or are not files (e.g., symlinks we don't handle)
        if not os.path.isfile(abspath):
             # print(f"Warning: Skipping non-file path during add: {relpath}", file=sys.stderr)
             continue

        try:
            # Hash the file content to get blob SHA
            with open(abspath, "rb") as fd:
                # Pass repo to object_hash to write the blob object
                sha = object_hash(fd, b"blob", repo)

            # Get file metadata using stat
            stat_info = os.stat(abspath)

            # Check if file is already in index and unchanged
            # This check is implicitly handled by removing and re-adding below,
            # but an explicit check could optimize by avoiding hashing if metadata matches.
            # existing_entry = current_entries_dict.get(relpath)
            # if existing_entry and existing_entry.sha == sha and ... (check metadata):
            #     continue # Skip if unchanged

            # Create new index entry
            ctime_s = int(stat_info.st_ctime)
            ctime_ns = stat_info.st_ctime_ns % 10**9
            mtime_s = int(stat_info.st_mtime)
            mtime_ns = stat_info.st_mtime_ns % 10**9

            # Determine file mode (permissions) - use fixed 644/755 based on executable bit?
            # Git derives this from stat, respecting core.filemode config.
            # Simple approach: use 644 for non-executable, 755 for executable.
            is_executable = os.access(abspath, os.X_OK)
            mode_perms = 0o755 if is_executable else 0o644
            # Mode type is regular file (0b1000)
            mode_type = 0b1000

            entry = GitIndexEntry(
                ctime=(ctime_s, ctime_ns), mtime=(mtime_s, mtime_ns),
                dev=stat_info.st_dev, ino=stat_info.st_ino,
                mode_type=mode_type, # Regular file
                mode_perms=mode_perms, # Permissions
                uid=stat_info.st_uid, gid=stat_info.st_gid,
                fsize=stat_info.st_size, sha=sha,
                flag_assume_valid=False, # Reset assume-valid flag on add
                flag_stage=0, # Normal stage (0)
                name=relpath # Store relative path
            )

            # Add the new/updated entry
            new_entries_list.append(entry)
            updated = True

        except Exception as e:
            print(f"Error adding file '{relpath}': {e}", file=sys.stderr)
            # Decide whether to continue or raise the exception

    # If any files were added or updated, write the new index
    if updated:
        index.entries = new_entries_list
        index_write(repo, index)


# Parser for 'commit' command
argsp = argsubparsers.add_parser("commit", help="Record changes to the repository.")
argsp.add_argument("-m", "--message", # Added long form --message
                   metavar="message",
                   dest="message",
                   required=True, # Make message mandatory for this simple implementation
                   help="Commit message.")


def gitconfig_read():
    """Read Git configuration from standard global locations."""
    # Prioritize repo-local config first
    # repo = repo_find(required=False) # Find repo if possible
    # config_files = []
    # if repo and repo.conf:
    #      # How to get the file path from ConfigParser object? Need to read it initially.
    #      local_config_path = repo_file(repo, "config")
    #      if local_config_path and os.path.exists(local_config_path):
    #          config_files.append(local_config_path)

    # Standard global locations
    xdg_config_home = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
    global_files = [
        os.path.join(xdg_config_home, "git/config"),
        os.path.expanduser("~/.gitconfig")
    ]
    # config_files.extend(global_files) # Add global files after local

    config = configparser.ConfigParser()
    # Read files - later files override earlier ones if keys conflict
    config.read(global_files) # Simplified: only read global for now
    return config


def gitconfig_user_get(config=None):
    """Get user name and email from Git configuration."""
    if config is None:
        config = gitconfig_read() # Read config if not provided

    # Check if 'user' section and required keys exist
    if "user" in config:
        name = config["user"].get("name")
        email = config["user"].get("email")
        if name and email:
            return f"{name} <{email}>"

    # Fallback or error if user info is missing
    # Git prompts or uses system info. We'll raise an error.
    raise Exception("User identity (name and email) not configured. "
                    "Please set them using:\n"
                    "  git config --global user.name \"Your Name\"\n"
                    "  git config --global user.email \"you@example.com\"")


def tree_from_index(repo, index):
    """Build a Git tree object hierarchy from the index entries."""
    # Ensure index entries are sorted, crucial for deterministic tree structure
    index.entries.sort(key=lambda e: e.name)

    tree_data = {} # { 'path/to/dir': [GitIndexEntry or (name, sha, mode) for subtree] }

    # Populate tree_data with entries, creating parent directory lists as needed
    for entry in index.entries:
        path = entry.name
        dirname = os.path.dirname(path)

        # Ensure all parent directory levels exist in tree_data
        current_dir = dirname
        while current_dir != "" and current_dir not in tree_data:
            tree_data[current_dir] = []
            parent_dir = os.path.dirname(current_dir)
            # Handle potential infinite loop if dirname('/') is '/'
            if parent_dir == current_dir: break
            current_dir = parent_dir
        if "" not in tree_data: # Ensure root directory entry exists
             tree_data[""] = []

        # Add the entry to its immediate parent directory's list
        tree_data[dirname].append(entry)


    # Build trees bottom-up (from longest path to shortest)
    # Sort directory paths by length descending to process children before parents
    sorted_paths = sorted(tree_data.keys(), key=len, reverse=True)

    tree_shas = {} # Store computed SHA for each directory path

    for path in sorted_paths:
        tree = GitTree(repo=repo) # Create a new tree object for this directory
        tree.items = [] # Initialize items list

        # Process entries (files or subtrees) in this directory
        for item in tree_data[path]:
            if isinstance(item, GitIndexEntry): # It's a file entry from the index
                # Mode needs to be bytes (e.g., b'100644')
                mode_str = f"{item.mode_type:o}{item.mode_perms:o}"
                mode_bytes = mode_str.encode('ascii')
                # Basename is the file/dir name within the current directory
                basename = os.path.basename(item.name)
                leaf = GitTreeLeaf(mode=mode_bytes, path=basename.encode('utf8'), sha=item.sha)
                tree.items.append(leaf)
            elif isinstance(item, tuple): # It's a subtree tuple (name, sha, mode) we created
                (name, sha, mode) = item
                leaf = GitTreeLeaf(mode=mode.encode('ascii'), path=name.encode('utf8'), sha=sha)
                tree.items.append(leaf)
            else:
                 raise TypeError(f"Unexpected item type in tree_data: {type(item)}")


        # Write the tree object to the object store and get its SHA
        # Pass repo object to object_write
        sha = object_write(tree, actually_write=True)
        tree_shas[path] = sha # Store the SHA for this directory path

        # Add this tree to its parent directory's list in tree_data
        parent_path = os.path.dirname(path)
        if parent_path != path: # Avoid adding root to itself
             basename = os.path.basename(path)
             # Mode for a directory is always '040000'
             subtree_tuple = (basename, sha, "040000")
             if parent_path in tree_data:
                 tree_data[parent_path].append(subtree_tuple)
             # else: Parent should exist due to initial population loop


    # The SHA of the root tree ("") is the final result
    root_sha = tree_shas.get("")
    if not root_sha:
         # Handle empty index case - create and write an empty tree object
         empty_tree = GitTree(repo=repo)
         empty_tree.items = []
         root_sha = object_write(empty_tree, actually_write=True)

    return root_sha


def commit_create(repo, tree_sha, parent_sha, author, committer, message):
    """Create a commit object and write it to the object store."""
    commit = GitCommit(repo=repo) # Pass repo to GitObject constructor
    commit.kvlm = collections.OrderedDict() # Use OrderedDict for canonical order

    commit.kvlm[b"tree"] = tree_sha.encode("ascii")
    if parent_sha:
        # Handle multiple parents (for merge commits) if parent_sha is a list/tuple
        if isinstance(parent_sha, (list, tuple)):
             commit.kvlm[b"parent"] = [p.encode("ascii") for p in parent_sha]
        else:
             commit.kvlm[b"parent"] = parent_sha.encode("ascii")

    # Get current timestamp with timezone info
    timestamp = datetime.datetime.now(datetime.timezone.utc).astimezone() # Use local timezone
    # Format timestamp and timezone offset (e.g., 1618228800 +0100)
    ts_seconds = int(timestamp.timestamp())
    offset_seconds = int(timestamp.utcoffset().total_seconds())
    offset_hours = abs(offset_seconds) // 3600
    offset_minutes = (abs(offset_seconds) % 3600) // 60
    tz_sign = "+" if offset_seconds >= 0 else "-"
    tz_str = f"{tz_sign}{offset_hours:02}{offset_minutes:02}"
    timestamp_str = f"{ts_seconds} {tz_str}"

    # Format author and committer lines
    commit.kvlm[b"author"] = f"{author} {timestamp_str}".encode("utf8")
    commit.kvlm[b"committer"] = f"{committer} {timestamp_str}".encode("utf8") # Use separate committer

    # Add commit message (ensure trailing newline)
    message = message.strip() + "\n"
    commit.kvlm[b""] = message.encode("utf8") # Message body uses empty key

    # Write the commit object and return its SHA
    return object_write(commit, actually_write=True)


def cmd_commit(args):
    """Handler for the 'commit' command."""
    repo = repo_find()
    index = index_read(repo)

    # Check if index is empty
    if not index.entries:
        print("Nothing to commit (working tree clean?) - Index is empty.") # More informative message
        return # Exit if nothing to commit

    # Create tree object from index
    tree_sha = tree_from_index(repo, index)

    # Determine parent commit SHA (HEAD)
    parent_sha = None
    try:
        parent_sha = object_find(repo, "HEAD", fmt=b'commit', follow=False) # Find commit pointed by HEAD
    except Exception:
        # No parent if HEAD doesn't exist or doesn't point to a commit (initial commit)
        print("Initial commit") # Log message for initial commit

    # Get author and committer info from config
    try:
        author = gitconfig_user_get() # Reads config and gets formatted "Name <email>"
        committer = author # Use same for committer in this simple case
    except Exception as e:
        print(f"Error getting user identity: {e}", file=sys.stderr)
        return # Abort commit if user info is missing

    # Create the commit object
    commit_sha = commit_create(repo, tree_sha, parent_sha, author, committer, args.message)

    # Update the current branch HEAD (or main HEAD if detached)
    head_path = repo_file(repo, "HEAD")
    active_branch_ref = None
    if head_path and os.path.exists(head_path):
         with open(head_path, "r") as f:
             head_content = f.read().strip()
         if head_content.startswith("ref: "):
             active_branch_ref = head_content[5:] # Get the ref path (e.g., refs/heads/master)

    if active_branch_ref:
        # Update the branch file
        branch_file_path = repo_file(repo, active_branch_ref)
        if branch_file_path:
             os.makedirs(os.path.dirname(branch_file_path), exist_ok=True) # Ensure parent dirs exist
             with open(branch_file_path, "w") as fd:
                 fd.write(commit_sha + "\n")
             print(f"[{os.path.basename(active_branch_ref)} {commit_sha[:7]}] {args.message.splitlines()[0]}")
        else:
             print(f"Error: Could not find path for branch ref {active_branch_ref}", file=sys.stderr)
    else:
        # Update HEAD directly if detached
        with open(head_path, "w") as fd:
            fd.write(commit_sha + "\n") # Write SHA directly to HEAD
        print(f"[Detached HEAD {commit_sha[:7]}] {args.message.splitlines()[0]}")