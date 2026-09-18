#!/usr/bin/env python3

"""Parallelizing compiler launcher for HIP.

This script parallelizes HIP compilations by splitting device and host compilations
out into individual processes. Compile times are potentially improved by splitting
the work into smaller tasks. It is mainly effective when the object files of a
project have very different compile times, and compile for many different HIP
architectures at the same time.

To use the script, simply prepend it to your compile command. For example:

  $ phc.py hipcc -c -o test.o test.hip --offload-arch=gfx1201 --offload-arch=gfx942

Note that parallel compilation is only activated when (1) an object file is compiled
(by passing -c), and (2) multiple explicit --offload-arch= options are passed to the
compiler. The result is byte-for-byte what the compiler driver itself would have
produced. Whenever the split cannot be done faithfully, the original command is run
unchanged rather than failing the build.

A GNU Make jobserver is used when one is available, but it is not required: without
one, phc limits itself to the -jN it can see in MAKEFLAGS, or to one job per
architecture. Set PHC_JOBSERVER=0 to ignore a jobserver, and PHC_DEBUG=1 to have phc
report what it decided to do.

Additionally, this script is designed to work under a GNU Make Jobserver, such as the
one launched by GNU Make. Ninja >= 1.13 also integrates with a GNU Make Jobserver, but
it does not launch its own server. In order to use the script with CMake, use a custom
jobserver, for example jobserver_pool.py from the Ninja misc scripts[0]. The easiest way
to use the script with CMake is to pass it as CMake HIP compiler launcher, using
-DCMAKE_{CXX,HIP}_COMPILER_LAUNCHER=/path/to/phc.py.

tl;dr:
  $ cmake [...] \
    -GNinja \
    -DCMAKE_CXX_COMPILER_LAUNCHER=/path/to/phc.py \
    -DCMAKE_HIP_COMPILER_LAUNCHER=/path/to/phc.py
  $ /path/to/ninja/misc/jobserver_pool.py ninja

[0]: https://github.com/ninja-build/ninja/blob/656412538b6fc102b809a61e0efce422e5a20534/misc/jobserver_pool.py
"""

import sys
import subprocess
import os
import asyncio
import tempfile
import re
import time
import hashlib
import threading
import signal
import ctypes
import errno
import shutil
import stat
import select
from queue import Queue

"""
Set PHC_DEBUG=1 to make phc explain itself on stderr: which clang-offload-bundler
and which -targets list it read back from the driver, whether it found a usable
jobserver and how many jobs it will run at once, and why it decided to fall back
to a plain serial compile. Useful when a build produces an unexpected object.
"""
DEBUG = os.environ.get("PHC_DEBUG", "0") not in ("", "0")

"""
The jobserver client for this process, or None if there is no usable jobserver.
Filled in by __main__ before the event loop starts; see JobserverClient.detect().
"""
JOBSERVER = None

def debug(message):
    """Print a diagnostic if PHC_DEBUG is set."""
    if DEBUG:
        print(f"phc: {message}", file=sys.stderr)

class FallbackToSerial(Exception):
    """
    Raised when phc cannot be confident that reassembling the compilation by hand
    would produce exactly what the compiler driver itself would have produced:
    no usable clang-offload-bundler, an unexpected -### output, a -targets list
    that does not match our architectures, a failing bundler run, ...

    Failing the build is never the right answer in those cases, because the
    original command is always a valid way to produce the object -- only slower.
    main() catches this and runs the unmodified command.
    """

"""
None, or an open handle to a file to write ninja-style logs. These can be post-processed
using Ninjatracing[1] to turn them into a perfetto trace. Keep in mind that the core
from ninjatracing assignment is not perfect. You have to manually prepend `# ninja log v7`
in order for Ninjatracing to accept the file.

Tracing can be enabled using PHC_NINJA_TRACE=path.

[1]: https://github.com/nico/ninjatracing
"""
TRACE_FILE = None

def trace(start, end, filename):
    """
    Append an entry to the ninja trace. `start` and `end` can be obtained using
    time.time(), `filename` should be something that identifies this trace element.
    Ninjatracing and Perfetto interpret this as a file name, but it does not necessarily
    need to be a (valid) path.
    """
    start = int(start * 1000)
    end = int(end * 1000)
    hash = hashlib.md5(filename.encode("utf-8")).hexdigest()[:16]
    if TRACE_FILE is not None:
        # Fake ninja (v7) log
        TRACE_FILE.write(f"{start}\t{end}\t{end}\t{filename}\t{hash}\n")
        TRACE_FILE.flush()

class JobserverClient:
    """
    A quick-and-dirty async GNU Make Jobserver client implementation. The jobserver config
    is parsed from the environment in the constructor. Note that this implementation only
    supports unix-authentication to keep it simple. Use a context to properly open and
    close the backing file descriptors, if there are any.
    """

    def __init__(self, named_pipe_path=None, read_fd=None, write_fd=None):
        """
        Initialize the Jobserver Client. Use JobserverClient.detect() rather than
        constructing one directly from the environment: detection can legitimately
        fail (no jobserver at all, or one whose descriptors we were not given) and
        that is not an error, it just means phc limits itself locally.
        """
        self.named_pipe_path = named_pipe_path
        self.named_pipe = named_pipe_path is not None
        self.read_fd = read_fd
        self.write_fd = write_fd

        # Set when the program is shutting down, to get the thread that is waiting
        # for a token to stop waiting. See acquire_jobserver_token().
        self.cancelled = False

        # Tokens we have taken and not yet given back. Kept so that they can all be
        # returned from a signal handler: a token that is not returned is lost for
        # the rest of the build, and the symptom is not this compile failing but some
        # later target hanging forever waiting for a slot that no longer exists.
        self.held = []

        # We have to use raw libc read for reading the fd socket (os.read doesn't work).
        # see JobserverClient.acquire for more info.
        self.libc = ctypes.CDLL('libc.so.6', use_errno=True)
        self.libc.read.argtypes = (ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t)
        self.libc.read.restype = ctypes.c_ssize_t

        # A buffer to read into, from C.
        self.read_buf = ctypes.create_string_buffer(1)

    @staticmethod
    def detect():
        """
        Build a client from the environment, or return None if there is no jobserver
        we can safely talk to. PHC_JOBSERVER=0 forces the latter.

        IMPORTANT: this has to run before asyncio (or anything else) opens a file
        descriptor of its own. See the comment on the descriptor pair below.
        """
        if os.environ.get("PHC_JOBSERVER", "1") in ("", "0"):
            debug("jobserver: disabled by PHC_JOBSERVER=0")
            return None

        # Sccache only sets CARGO_MAKEFLAGS and removes the original MAKEFLAGS, so
        # try to parse them both.
        # An empty but present MAKEFLAGS falls through to CARGO_MAKEFLAGS: sccache
        # sets the latter and clears the former, and `os.environ.get("MAKEFLAGS", ...)`
        # would return that empty string rather than the default.
        makeflags = os.environ.get("MAKEFLAGS") or os.environ.get("CARGO_MAKEFLAGS", "")

        # ninja/misc/jobserver_pool.py starts a jobserver with a named pipe, but
        # sccache starts a jobserver with an anonymous pipe. So we have to parse
        # styles of environment variable. This is also why there are two separate
        # filedescriptor members.

        m = re.search(r"--jobserver-auth=fifo:([^\s]+)", makeflags)
        if m is not None:
            path = m.group(1)
            try:
                if not stat.S_ISFIFO(os.stat(path).st_mode):
                    debug(f"jobserver: '{path}' is not a fifo, ignoring it")
                    return None
            except OSError as e:
                debug(f"jobserver: cannot stat '{path}': {e}")
                return None
            debug(f"jobserver: fifo {path}")
            return JobserverClient(named_pipe_path=path)

        m = re.search(r"--jobserver-auth=(\d+),(\d+)\b", makeflags)
        if m is None:
            m = re.search(r"--jobserver-fds=(\d+),(\d+)\b", makeflags)
        if m is not None:
            read_fd = int(m.group(1))
            write_fd = int(m.group(2))

            # GNU Make advertises --jobserver-auth=R,W in MAKEFLAGS for every recipe,
            # but it only *passes the descriptors* to recipes it considers recursive
            # (those prefixed with '+' or mentioning $(MAKE)). For an ordinary recipe
            # they are closed, and a client that believes MAKEFLAGS blocks forever on
            # a read from a descriptor that is not there -- this is what makes other
            # jobserver clients hang under `make -j4` with a plain compile rule.
            #
            # So check that they really are open, and really are pipes. Both halves
            # matter. Descriptors 3 and 4 are the first two free numbers, so if make
            # did not pass them the next thing that opens a file descriptor will be
            # handed exactly those numbers: asyncio, for instance, creates a socket
            # pair for its own wakeup and would land right there. Reading a byte out
            # of the event loop's wakeup pipe would be a far more confusing failure
            # than a hang. A socket is therefore rejected (make uses pipe(2)), and
            # detection runs before asyncio starts (see __main__).
            for fd, what in ((read_fd, "read"), (write_fd, "write")):
                try:
                    mode = os.fstat(fd).st_mode
                except OSError as e:
                    debug(f"jobserver: {what} descriptor {fd} was not inherited ({e}); "
                          "the recipe is probably not marked '+'")
                    return None
                if not stat.S_ISFIFO(mode):
                    debug(f"jobserver: {what} descriptor {fd} is not a pipe, ignoring it")
                    return None

            debug(f"jobserver: descriptors {read_fd},{write_fd}")
            return JobserverClient(read_fd=read_fd, write_fd=write_fd)

        debug(f"jobserver: no --jobserver-auth in MAKEFLAGS '{makeflags}'")
        return None

    def __enter__(self):
        """
        Open the fifo file descriptors if required.

        If the jobserver passes a named pipe, we have to explicitly open it and close
        it. If the jobserver passes anonymous pipe fd's, theyre already opened, and we
        don't have to do anything here.
        """
        if self.named_pipe:
            # Open read-write rather than once for reading and once for writing: a
            # fifo opened O_RDONLY blocks until some other process opens the writing
            # end, which would hang here before any work has even started.
            self.read_fd = os.open(self.named_pipe_path, os.O_RDWR)
            self.write_fd = self.read_fd

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Give back anything we still hold and close the fifo file descriptors if we
        opened any before.
        """
        self.release_all()
        if self.named_pipe:
            os.close(self.read_fd)
            self.read_fd = self.write_fd = None

    def acquire(self, timeout=0.05):
        """Try to take a job slot, waiting at most `timeout` seconds for one.

        Returns the token, as bytes, or None if no token became available in time. A
        token must be given back with JobserverClient.release() when the job it paid
        for is done. Run this on a separate thread so that work can continue on the
        main one.

        Two things make this more delicate than it looks.

        The wait is a select() rather than a blocking read, because the descriptor may
        already be non-blocking and there is nothing we may do about it: GNU Make 4.3
        sets O_NONBLOCK on its jobserver pipe, and the flag lives on the open file
        description that make and every other client share, so it can neither be
        relied upon nor changed. A read that returns EAGAIN therefore means "no token
        right now", not "something went wrong" -- treating it as an error is what used
        to make phc die inside a worker thread and hang the whole compilation. It can
        also happen after select() says the pipe is readable, when another client wins
        the race for the same byte.

        And the read itself has to go through libc rather than os.read: CPython
        retries a read that fails with EINTR, and there is no way out of that loop, so
        a blocking read could not be interrupted to shut the program down.
        """
        # select() is interrupted by a signal but CPython restarts it, so a cancelled
        # wait is noticed through self.cancelled at the next timeout instead.
        try:
            readable, _, _ = select.select([self.read_fd], [], [], timeout)
        except (OSError, ValueError):
            return None
        if not readable:
            return None

        res = self.libc.read(self.read_fd, self.read_buf, 1)
        if res < 0:
            err = ctypes.get_errno()
            if err in (errno.EAGAIN, errno.EWOULDBLOCK, errno.EINTR):
                # No token after all: somebody else got there first, or we were woken
                # up by a signal. Neither is an error.
                return None
            raise OSError(err, os.strerror(err))
        elif res != 1:
            # Its only possible that res=0 here since we request for 1 byte.
            # I think this can only happen if the pipe is prematurely closed, so just raise
            # a related OSError.
            raise OSError(errno.EPIPE, os.strerror(errno.EPIPE))

        # .raw is the one byte that was read, as bytes: a token is not necessarily a
        # printable character, so it has to be round-tripped verbatim.
        token = self.read_buf.raw
        self.held.append(token)
        return token

    def release(self, token):
        """
        Write a token back into the fifo. This operation is not asynchronous because it
        basically never blocks anyway: On Linux, a pipe should have 64kB of internal storage
        by default, and a job server should only require a couple of hundred tokens at most.

        Only tokens we are still holding are written back. Every token is the same one
        byte, so the list of held tokens is really a count, and returning one we no
        longer hold would not give a token back -- it would invent one, and let the
        rest of the build run more jobs than -j allows. That is not hypothetical: the
        signal handler below returns everything we hold, and the interpreter does not
        necessarily die afterwards, so the jobs that were running go on to return the
        very same tokens on their way out.
        """
        try:
            self.held.remove(token)
        except ValueError:
            return
        try:
            os.write(self.write_fd, token)
        except OSError as error:
            # Very unlikely -- a jobserver pipe holds at most a few hundred tokens and
            # we only ever write back what we took -- but letting this escape would
            # leave the queue waiting for a task that nobody is going to finish.
            print(f"phc: failed to return a jobserver token: {error}", file=sys.stderr)
            return
        debug(f"jobserver: returned a token ({len(self.held)} still held)")

    def release_all(self):
        """
        Return every token we still hold. Called when the compilation is over and from
        the signal handler installed by compile_parallel(), so that a build interrupted
        with ^C or killed does not quietly shrink the jobserver for everyone else.
        """
        while self.held:
            try:
                self.release(self.held[-1])
            except OSError:
                # Nothing useful to do here; the pipe is gone and so is the build.
                del self.held[-1]

class NullJobserverClient(JobserverClient):
    """
    Stand-in used when there is no jobserver to talk to: phc must still work when it
    is run straight from a shell, from a test script, under `make -j1` (GNU Make does
    not start a jobserver for a single job) or under a recipe that was not given the
    jobserver descriptors.

    Rather than special-casing every call site, it hands out a fixed number of tokens
    from a private pipe of its own. That makes it behave exactly like the real thing
    -- including blocking in libc read(2) until a token comes back, and being
    interruptible by the SIGUSR1 that jobserver_scheduler() uses to cancel a pending
    acquire -- while capping concurrency locally instead of globally.
    """

    def __init__(self, tokens):
        super().__init__()
        self.tokens = max(0, tokens)

    def __enter__(self):
        self.read_fd, self.write_fd = os.pipe()
        os.write(self.write_fd, b"+" * self.tokens)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.close(self.read_fd)
        os.close(self.write_fd)
        self.read_fd = self.write_fd = None

def make_job_limit():
    """
    The -jN that make was started with, or None if it did not name a number.

    Only relevant when there is no jobserver: GNU Make starts one for -j2 and up, so
    what is left here is mostly `-j1` (and `-j`, which means unlimited). Without this,
    `make -j1` would run every architecture at once -- the exact opposite of what was
    asked for.
    """
    makeflags = os.environ.get("MAKEFLAGS") or os.environ.get("CARGO_MAKEFLAGS", "")
    m = re.search(r"-j\s*(\d+)", makeflags)
    return int(m.group(1)) if m is not None else None

def open_jobserver(jobs):
    """
    Return the client to distribute `jobs` parallel compilations over.

    Concurrency is one implicit slot -- the one the caller already accounted for when
    it started phc -- plus one job per token, so a client that hands out `jobs - 1`
    tokens allows all of them to run at once.
    """
    client = JOBSERVER
    if client is not None:
        return client

    limit = make_job_limit()
    slots = jobs if limit is None else max(1, min(jobs, limit))
    debug(f"no usable jobserver: running at most {slots} of {jobs} job(s) at a time")
    return NullJobserverClient(slots - 1)

async def run(cmd):
    """
    Asynchronously run a command to completion. This is basically an async version of
    subprocess.run(cmd, check=True): stdout and stderr is buffered and then written to the
    parent process' stdout and stderr. If the process' exit code was not 0, then this
    function raises subprocess.CalledProcessError with the appropriate fields set.

    See https://docs.python.org/3/library/asyncio-subprocess.html.
    """
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout, stderr = await process.communicate()

    # Write the subcommand's output to the output if there was any.
    if stdout:
        sys.stdout.buffer.write(stdout)
    if stderr:
        sys.stderr.buffer.write(stderr)

    if process.returncode != 0:
        raise subprocess.CalledProcessError(
            returncode=process.returncode,
            cmd=" ".join(cmd),
            output=stdout,
            stderr=stderr,
        )

async def run_capture(cmd):
    """
    Run a command and return (returncode, stdout, stderr) as text, printing nothing.
    Used for the -### probe, whose output is phc's input rather than the user's.

    Never raises: a command that cannot even be started is reported like any other
    failure, and the caller falls back to something that works.
    """
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as error:
        return 1, "", str(error)

    stdout, stderr = await process.communicate()
    return (process.returncode,
            stdout.decode("utf-8", "replace"),
            stderr.decode("utf-8", "replace"))

async def main():
    """
    Main entry point for the compiler launcher. Arguments are passed via sys.argv.
    """

    # Extract arguments. Note: Skip the script name.
    cmd = sys.argv[1:]
    # Add in the compiler if we've passed it explicitly
    if compiler := os.environ.get('PHC_COMPILER'):
        cmd.insert(0, compiler)

    # print(cmd, file=sys.stderr)
    # print(os.environ)

    # Open a trace if requested via PHC_NINJA_TRACE.
    trace_path = os.environ.get("PHC_NINJA_TRACE", "")
    if trace_path != "":
        global TRACE_FILE
        TRACE_FILE = open(trace_path, "a")

    # sccache launches the compiler in preprocessor-only mode. This has some issues in HIP,
    # but we can fix them up here.
    if "-E" in cmd:
        await preprocess_only(cmd)
        return

    # Check whether we're actually being asked to compile something. Sometimes CMake or
    # other tools run the compiler to get information about the compiler, for example
    # the implicit include directories. Additionally, don't bother with commands that
    # don't produce an object file: Those require more effort and project that use CMake
    # split that out into a separate command anyway.
    if "-c" not in cmd:
        await run(cmd)
        return

    # If not an actual heterogenous compilation, then also quit.
    for flag in ["--cuda-device-only", "--offload-device-only", "--cuda-host-only", "--offload-host-only"]:
        if flag in cmd:
            await run(cmd)
            return

    # Fetch the list of GPU targets to compile for from the command line.
    offload_archs = [arg[len("--offload-arch="):] for arg in cmd if arg.startswith("--offload-arch=")]

    # When using ROCm-CMake's automatic target detection on a system with multiple GPUs of
    # the same type, a particular arch is sometimes passed twice. This doesn't really add
    # anything though, so just quickly #filter these out.
    offload_archs = list(set(offload_archs))

    # Nothing to parallelize anyway, so don't bother. This will also filter out any remaining
    # compilations that don't offload.
    if len(offload_archs) <= 1:
        await run(cmd)
        return

    # Everything below reassembles by hand what the compiler driver would have done
    # internally. Whenever that turns out not to be reproducible faithfully, run the
    # original command instead: it is slower, but it is never wrong, and a build is
    # not the place to be pedantic about it.
    spec = CompileSpec(cmd)
    try:
        await compile_parallel(cmd, offload_archs, spec)
    except FallbackToSerial as reason:
        debug(f"falling back to a plain serial compile: {reason}")
        await run(cmd)

class CompileSpec:
    """
    The handful of things phc has to know about a compile command.

    The command line is scanned once, left to right, because several of these are
    last-one-wins (a repeated -o) and several take a
    separate argument that must not be mistaken for a flag (-MF x.d).
    """

    def __init__(self, cmd):
        # The final object. On the whole-program path the host compilation writes it
        # directly and this is only used to name trace entries, but on the RDC path
        # phc writes it itself with the bundler, so there it has to be known.
        self.output = "a.out"
        self.have_output = False

        # Only used by the fallback in compile_parallel() for a toolchain whose -###
        # names no bundler; normally the compression flags are mirrored from the
        # driver's own bundler command line.
        self.offload_compress = False
        self.offload_compression_level = None

        # A compilation unit id supplied by the caller, which always wins over one
        # phc would pick itself.
        self.cuid = None

        # Dependency file generation, which is a property of the host compilation.
        self.depfile = False          # -MD/-MMD: a dependency file is wanted
        self.depfile_named = False    # -MF: the caller said where it goes
        self.depfile_target = False   # -MT/-MQ: the caller said what rule it is for

        it = iter(cmd)
        for arg in it:
            if arg == "-o":
                self.output = next(it, self.output)
                self.have_output = True
            elif arg.startswith("-o"):
                self.output = arg[2:]
                self.have_output = True
            elif arg == "--offload-compress":
                self.offload_compress = True
            elif arg.startswith("--offload-compression-level="):
                self.offload_compression_level = arg.split("=")[-1]
            elif arg.startswith("-cuid="):
                self.cuid = arg[len("-cuid="):]
            elif arg in ("-MD", "-MMD"):
                self.depfile = True
            elif arg == "-MF":
                self.depfile_named = True
                next(it, None)
            elif arg in ("-MT", "-MQ"):
                self.depfile_target = True
                next(it, None)

class DriverInfo:
    """
    What the compiler driver itself says it would do for a given command line.

    phc puts the pieces of a compilation back together by hand, so every detail that
    ends up in the final object is read back from the driver with -### (which prints
    the sub-commands it would run, without running them) rather than guessed:

    bundler       the clang-offload-bundler belonging to *this* toolchain. It cannot
                  be derived from the compiler's path: on the ROCm builds CMSSW uses,
                  hipcc lives in rocm-hip/bin while the bundler lives in
                  rocm-llvm/lib/llvm/bin, and the clang-offload-bundler that happens
                  to be in PATH belongs to a different LLVM altogether.
    targets       the canonical -targets list. Its spelling depends on the toolchain
                  *and* on the mode -- `hipv4-amdgcn-amd-amdhsa--gfx942:sramecc+`
                  with `host-x86_64-unknown-linux-gnu` for a whole-program compile,
                  but `hip-amdgcn-amd-amdhsa-unknown-gfx942:sramecc+` with
                  `host-x86_64-redhat-linux-gnu` for an -fgpu-rdc one -- and the host
                  part is the distribution's own triple, not a fixed string. Their
                  order is significant too, and it differs between the two modes.
    bundle_extra  whatever else the driver hands to its bundler.
    cuid          the compilation unit id the driver picked. Every cc1 invocation of
                  one translation unit has to share it, and our host-only and
                  device-only command lines would otherwise each hash a different one
                  out of their own (differing) arguments.

    One -### run costs about 50 ms and compiles nothing, which is negligible next to
    the multi-second compilation it makes correct.
    """

    def __init__(self):
        self.bundler = None
        self.targets = None
        self.bundle_extra = []
        self.cuid = None

# A single "quoted argument" of a -### command line, with \" and \\ escapes.
DRIVER_ARGUMENT = re.compile(r'"((?:[^"\\]|\\.)*)"')

async def probe_driver(cmd, extra=()):
    """
    Ask the driver what it would do for this command line and return a DriverInfo.
    Fields that could not be found are left as None; the caller decides whether it
    can live without them.

    `extra` holds the arguments phc is going to add to each of its sub-compiles, so
    that the probe answers for the command line that will actually be used -- and so
    that a driver which does not understand them produces no bundler here rather than
    an error later.
    """
    returncode, stdout, stderr = await run_capture(list(cmd) + list(extra) + ["-###"])

    info = DriverInfo()
    # -### goes to stderr, but be forgiving about where a wrapper puts it.
    for line in (stderr + "\n" + stdout).splitlines():
        # Undo the \" and \\ escaping the driver applies inside each argument.
        args = [re.sub(r"\\(.)", r"\1", m.group(1))
                for m in DRIVER_ARGUMENT.finditer(line)]
        if not args:
            continue

        if info.cuid is None:
            for arg in args:
                if arg.startswith("-cuid="):
                    info.cuid = arg[len("-cuid="):]
                    break

        if info.bundler is None and os.path.basename(args[0]).startswith("clang-offload-bundler"):
            if not os.path.isfile(args[0]):
                continue
            info.bundler = args[0]
            for arg in args[1:]:
                if arg.startswith("-targets="):
                    info.targets = arg[len("-targets="):].split(",")
                elif arg.startswith(("-input=", "-output=", "-type=")):
                    # Supplied by us, not copied from the driver.
                    pass
                else:
                    # Everything else the driver passes to its bundler is mirrored:
                    # -bundle-align=4096 for a fat binary, --compress and
                    # --compression-level= when they were asked for. Mirroring beats
                    # hardcoding because it differs per mode -- this ROCm passes
                    # -bundle-align only for the whole-program bundle -- and a flag
                    # the driver would not have passed produces an object the driver
                    # would never have produced.
                    info.bundle_extra.append(arg)

    if info.bundler is None:
        debug(f"the driver named no clang-offload-bundler in its -### output "
              f"(exit status {returncode})")
    return info

def legacy_bundler_path(cmd):
    """
    Look for a clang-offload-bundler next to the compiler. Only used when the -###
    output named none, e.g. on a toolchain that does not bundle at compile time.

    This deliberately does not search PATH. The clang-offload-bundler in PATH often
    belongs to a different LLVM than the one the driver uses -- in the CMSSW ROCm
    builds it is a separate llvm package entirely -- and bundling with the wrong
    version corrupts silently instead of failing.
    """
    compiler = cmd[0]
    if os.path.dirname(compiler) == "":
        compiler = shutil.which(compiler) or compiler
    clang_dir = os.path.dirname(os.path.realpath(compiler))

    for candidate in (
        os.path.join(clang_dir, "clang-offload-bundler"),
        os.path.join(clang_dir, "..", "llvm", "bin", "clang-offload-bundler"),
        os.path.join(clang_dir, "..", "lib", "llvm", "bin", "clang-offload-bundler"),
    ):
        if os.path.isfile(candidate):
            return candidate

    raise FallbackToSerial("could not find clang-offload-bundler")

def bundle_inputs(targets, device_inputs, host_input):
    """
    Return the -input= arguments in the order the driver's -targets list dictates.

    clang-offload-bundler matches inputs to targets by position, so this order is not
    cosmetic: the driver puts the host entry first in a whole-program fat binary and
    last in an RDC object.

    An entry starting with `host-` takes the host input. For any other the
    architecture is the text after the *last* '-': gfx names never contain a '-', but
    they do contain ':' and '+', as in hip-amdgcn-amd-amdhsa-unknown-gfx90a:sramecc+.
    """
    if len(targets) != len(device_inputs) + 1:
        raise FallbackToSerial(f"the driver listed {len(targets)} targets "
                               f"for {len(device_inputs)} architecture(s)")

    inputs = []
    matched = set()
    for target in targets:
        if target.startswith("host-"):
            inputs.append(f"-input={host_input}")
            continue
        arch = target.rsplit("-", 1)[-1]
        if arch not in device_inputs:
            raise FallbackToSerial(f"target '{target}' matches none of the "
                                   f"requested architectures")
        matched.add(arch)
        inputs.append(f"-input={device_inputs[arch]}")

    if len(matched) != len(device_inputs):
        raise FallbackToSerial("the driver's target list does not name every architecture")
    return inputs

async def run_bundler(bundle_cmd):
    """
    Run clang-offload-bundler. A failure here is not the user's fault and not worth
    failing a build over, so it turns into a plain serial compile.
    """
    try:
        await run(bundle_cmd)
    except subprocess.CalledProcessError as error:
        raise FallbackToSerial(f"clang-offload-bundler exited with {error.returncode}")

async def compile_parallel(cmd, offload_archs, spec):
    """
    Compile one translation unit as one device compilation per architecture plus a
    single host compilation, and reassemble the result.

    Raises FallbackToSerial if that cannot be done faithfully. Exits with status 1 if
    one of the sub-compilations reported a genuine compile error (already printed).
    """
    # When compiling different compilation units separately, each one needs a 'CUID'
    # passed to it to help identify which compilation unit an object is part of.
    # Usually this is passed by the clang driver, but since we are emulating the driver
    # we need to pass it ourselves.
    #
    # It has to be passed as a *driver* option. Passing it to cc1 with -Xclang, as this
    # script used to, does not work: the driver appends its own generated -cuid after
    # our arguments and the last one wins. The symptom is subtle -- the object still
    # compiles, links and runs -- but every architecture's bundle then carries a
    # different __hip_cuid_<hash>, where a plain compile gives all of them the same
    # one, and the hash changes on every rebuild because the driver derives it from the
    # command line, which for our sub-compiles names a fresh temporary directory.
    #
    # A caller-supplied -cuid= always wins. Otherwise derive one from the command line,
    # deterministically, so that recompiling an unchanged file reproduces the object
    # bit for bit (which is what makes the result cacheable and diffable).
    cuid = spec.cuid
    if cuid is None:
        seed = os.path.abspath(spec.output) + "\0" + "\0".join(cmd)
        cuid = hashlib.sha256(seed.encode("utf-8", "replace")).hexdigest()[:16]
        cuid_args = [f"-cuid={cuid}"]
    else:
        # Already on the command line; adding it again would only be noise.
        cuid_args = []

    # Ask the driver what it would do; see DriverInfo for why none of this is guessed.
    info = await probe_driver(cmd, cuid_args)
    bundler = info.bundler or legacy_bundler_path(cmd)

    if info.targets is not None:
        targets = info.targets
        bundle_extra = info.bundle_extra
    else:
        # The original hardcoded spellings, kept for toolchains whose -### does not
        # mention a bundler at all. They assume an x86_64 Linux host.
        debug("no -targets from the driver, using the built-in whole-program spellings")
        targets = ["host-x86_64-unknown-linux-"] + \
                  [f"hipv4-amdgcn-amd-amdhsa--{arch}" for arch in offload_archs]
        bundle_extra = ["-bundle-align=4096"]
        if spec.offload_compress:
            bundle_extra.append("-compress")
        if spec.offload_compression_level is not None:
            bundle_extra.append(f"-compression-level={spec.offload_compression_level}")

    if info.cuid is not None and info.cuid != cuid:
        # The driver did not take our -cuid=. Nothing is broken by this -- it is what
        # this script always did -- but the object will not be reproducible.
        debug(f"the driver kept its own cuid {info.cuid} instead of {cuid}")

    debug(f"bundler: {bundler}")
    debug(f"targets: {','.join(targets)}")
    debug(f"cuid:    {cuid}")
    debug(f"mode:    whole program, "
          f"{len(offload_archs)} architectures: {' '.join(offload_archs)}")

    with open_jobserver(len(offload_archs)) as jobclient:

        # Tokens must go back even if this process is interrupted. A token that is not
        # returned is gone for the rest of the build, and what then hangs is not this
        # compilation but some later target waiting for a slot that no longer exists.
        saved_handlers = {}

        def terminate(signum, _frame):
            jobclient.release_all()
            signal.signal(signum, saved_handlers.get(signum, signal.SIG_DFL))
            os.kill(os.getpid(), signum)

        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            try:
                saved_handlers[signum] = signal.signal(signum, terminate)
            except (ValueError, OSError):
                pass

        try:
            with tempfile.TemporaryDirectory(prefix="phc-") as dir:

                # We can't let the main thread idle because that might actually cause a deadlock
                # (if all main threads are idling they are wasting their implicit job slot token).
                # Therefore we are also going to process offload compilation tasks on the main
                # thread. The work is divided using a work-stealing method and two 'scheduler's.
                # `main_scheduler()` runs tasks on the main thread, while `jobserver_scheduler()`
                # tries to acquire job slots and launch new async jobs if so.
                # `tasks` is the queue of compilation jobs to finish. `error_event` is an asyncio
                # event used to indicate that any job failed. We'll check it later after syncing
                # with the queue.

                tasks = asyncio.Queue()
                error_event = asyncio.Event()

                # Where each architecture's device code lands. The RDC path puts bitcode
                # here and the whole-program path a code object, but neither the name nor
                # compile_device() has to care.
                device_inputs = {arch: os.path.join(dir, f"{arch}.out") for arch in offload_archs}

                # Create the compilation tasks.
                for arch in offload_archs:
                    tasks.put_nowait(compile_device(cmd, arch, device_inputs[arch], cuid_args,
                                                    spec.output, error_event))

                # Launch our schedulers.
                asyncio.create_task(main_scheduler(tasks))
                jobserver_scheduler_task = asyncio.create_task(jobserver_scheduler(tasks, jobclient))

                # Wait for all tasks to be done processing.
                # We don't need to wait for the schedulers to finish, because tasks.task_done() is only
                # called _after_ the object is properly compiled.
                await tasks.join()

                # Cancel the jobserver scheduler task: This is needed if the main thread completed the
                # last task and the job server is currently exhausted. In that case, there is currently
                # a background thread blocking on jobserver.acquire, and we have to cancel that to cleanly
                # exit this program. See `jobserver_scheduler()` for more details.
                jobserver_scheduler_task.cancel()

                # tasks.task_done() is called after any potential changes to error_event, so we can
                # check here if its set. (Note: this used to call .set(), which always returns
                # None, so a device compile error was only noticed later, as a confusing
                # failure of the bundler.)
                if error_event.is_set():
                    # Note: error is already printed.
                    sys.exit(1)

                start = time.time()

                # The remainder of the commands are all serially executed within the same jobserver
                # task, the main thread of this process.

                # Whole-program compilation. The device code is packed into a fat
                # binary which the host compilation then embeds, and the host
                # compilation writes the final object itself.
                #
                # Note: the bundle also needs an entry for the host target, even
                # though HIP does not use it, hence the -input=/dev/null that
                # bundle_inputs() places wherever the driver puts the host target.
                bundle = os.path.join(dir, "bundle.hipfb")
                bundle_cmd = [bundler, "-type=o"] + bundle_extra + \
                    ["-targets=" + ",".join(targets)] + \
                    bundle_inputs(targets, device_inputs, "/dev/null") + \
                    [f"-output={bundle}"]
                await run_bundler(bundle_cmd)

                bundle_end = time.time()
                trace(start, bundle_end, f"{spec.output}::bundle")

                # Compile the final executable.
                # Preprocess the host compilation command.
                host_cmd = []
                for arg in cmd:
                    # This time, we don't need to include any GPU targets to compile for, as we're only
                    # targeting the host. We can leave the MF/MD/MT and -o options in place this time,
                    # we actually want to emit the dependency info as well as put the object in the
                    # original output location.
                    if arg.startswith("--offload-arch="):
                        pass
                    # Also get rid of --offload-jobs, its not needed anymore.
                    elif arg.startswith("--offload-jobs="):
                        pass
                    # Skip any flags related to compression, we'll do that later.
                    elif arg == "--offload-compress":
                        pass
                    elif arg.startswith("--offload-compression-level"):
                        pass
                    # Pass on any other options.
                    else:
                        host_cmd.append(arg)

                # Only compile the host part of the input file, ignore any device code.
                host_cmd.append("--offload-host-only")
                # Ask clang to embed the offload bundle that we produced earlier. Note: this must be
                # passed to cc1 via -Xclang.
                host_cmd.append("-Xclang")
                host_cmd.append("-fcuda-include-gpubinary")
                host_cmd.append("-Xclang")
                host_cmd.append(bundle)
                # And the CUID, as a driver option (see above).
                host_cmd.extend(cuid_args)
                await run(host_cmd)

                host_end = time.time()
                trace(bundle_end, host_end, spec.output)
        finally:
            for signum, handler in saved_handlers.items():
                signal.signal(signum, handler)

async def main_scheduler(queue):
    """
    Scheduler for the main thread. This function pulls tasks from the queue and
    runs them until completion. Exits when the queue is empty.
    """
    while not queue.empty():
        try:
            task = queue.get_nowait()
        except asyncio.QueueEmpty:
            return

        try:
            await task
        finally:
            queue.task_done()

async def jobserver_scheduler(queue, jobclient):
    """
    Scheduler for running tasks on jobserver slots. The basic idea of this function
    is to pull jobs from the queue, wait for a token to be ready, and then start a
    new asyncio task to run it to completion.

    When the last item of the queue has been completed by `main_scheduler()` and we've
    already started waiting for a job slot, we have to cancel that read. Both because
    we no longer need the slot and to cleanly exit the program, using sys.exit(0) without
    ending the read causes the program to hang.

    The only way to interrupt a blocking read() is by sending a signal to the thread,
    which causes the read to fail with EINTR. In order to do that we also have to
    install a dummy signal handler, because otherwise the program would simply crash.
    """

    # Install the dummy signal handler
    signal.signal(signal.SIGUSR1, sigusr1_handler)

    while not queue.empty():
        # In order to send the kill signal to the thread that is currently blocking,
        # we have to know its thread ID. This queue is used to send the TID from the
        # worker thread back to this thread.
        tid_queue = Queue()

        try:
            # Wait for a token on an asyncio background thread.
            token = await asyncio.to_thread(acquire_jobserver_token, jobclient, tid_queue)
        except asyncio.CancelledError:
            # Tell the waiting thread to stop waiting, which it checks between polls,
            # and interrupt whatever it is blocked in so that it notices right away.
            jobclient.cancelled = True
            # Wait for the TID to get sent throug the queue. Its probably already there,
            # but this way we can be sure. There shouldnt be a problem with waiting for a
            # brief moment, it shouldn't be able to get stuck.
            tid = tid_queue.get()
            # Kill him, Anakin, kill him now.
            signal.pthread_kill(tid, signal.SIGUSR1)
            return

        if token is None:
            # No token, and none to be expected: the jobserver went away or we are
            # shutting down. Stop scheduling extra jobs -- the main thread is working
            # through the very same queue, so this costs parallelism and nothing else.
            debug("jobserver: no tokens available, continuing on the main thread only")
            return

        try:
            task = queue.get_nowait()
        except asyncio.QueueEmpty:
            # Release if the main_scheduler already started processing this item.
            jobclient.release(token)
            return

        # Complete the work in a different task so that we can
        # continue scheduling new work here.
        debug(f"jobserver: acquired a token ({len(jobclient.held)} held, "
              f"{len(jobclient.held) + 1} job(s) running)")
        asyncio.create_task(jobserver_worker(task, queue, jobclient, token))

def sigusr1_handler(_signum, _frame):
    """
    Dummy signal handler. We don't need it to actually do anything, its just
    here to prevent Linux from automatically killing the process because a
    signal handler is missing.
    """
    pass

def acquire_jobserver_token(jobclient, tid_queue):
    """Wait for a jobserver token and return it, or None if none can be had.

    This function should be run on a separate thread so that it can be cancelled
    using signal.pthread_kill(). The TID to kill is passed through the tid_queue
    to the calling thread.

    It returns None rather than failing: this thread only buys *extra* parallelism,
    and the main thread is working through the same queue. Anything raised or exited
    here would be swallowed by the asyncio runtime and would leave the queue's join()
    waiting for tasks nobody is going to run any more -- which is a hang, and a hang
    of the whole build rather than of this one compilation.
    """

    # threading.get_ident() should correspond with the pthread thread ID on posix
    # systems.
    tid_queue.put(threading.get_ident())

    while not jobclient.cancelled:
        try:
            token = jobclient.acquire()
        except OSError as e:
            if e.errno != errno.EINTR:
                print('failed to acquire jobserver token:', e, file=sys.stderr)
            return None
        if token is not None:
            return token

    return None

async def jobserver_worker(task, queue, jobclient, token):
    """
    This worker runs a coroutine to completion, using a token
    obtained from a job server. When the coroutine is completed,
    the token is released and the task is marked as done in the associated job
    queue.
    """
    try:
        await task
    finally:
        # task_done() has to happen whatever else goes wrong: the main thread is
        # blocked in queue.join() and would otherwise wait forever.
        try:
            jobclient.release(token)
        finally:
            queue.task_done()

async def compile_device(cmd, arch, output, cuid_args, host_output, error_event):
    """
    Asynchronously compile the device code of a source file for a particular architecure.

    Parameters:
    -----------
    cmd:
        The full compiler command for the compilation. Still contains the --offload_archs=
        options for the other architectures.
    arch:
        The architecture to compile this object for.
    output:
        The location to place the output for the compilation for this architecture.
    cuid_args:
        The arguments that pin the Compilation Unit ID, which must be the same for the host
        and the device compilation of one compilation unit. Empty if the caller already
        passed a -cuid= of its own.
    host_output:
        The host output corresponding to this compilation. This is mainly used for tracing.
    error_event:
        An asyncio event to set when the compilation yielded an error. When the event is
        set, the error message is already printed to stdout/stderr.
    """

    # Pre-process the compilation command to fix it up for device-only compilation.
    new_cmd = []
    it = iter(cmd)
    for arg in it:
        # Get rid of any --offload-arch= options, we'll fix them up later.
        if arg.startswith("--offload-arch="):
            pass
        # Also get rid of --offload-jobs: We only have one compilation now. If its still
        # passed, then clang will emit a warning.
        elif arg.startswith("--offload-jobs="):
            pass
        # Get rid of any dependency information flags: We don't want to regenerate
        # these files every time. Besides, clang gives a warning about not having used
        # these options if they are passed with a device-only compilation.
        elif arg == "-MD" or arg == "-MMD":
            pass
        # -MQ takes an argument just like -MT and -MF does; leaving it behind would
        # hand the device compilation a stray extra input file.
        elif arg == "-MT" or arg == "-MF" or arg == "-MQ":
             next(it)
        # Get rid of the original output file. We'll add the new one later.
        elif arg == "-o":
            next(it)
        elif arg.startswith("-o"):
            pass
        # Skip any flags related to compression, we'll do that later.
        elif arg == "--offload-compress":
            pass
        elif arg.startswith("--offload-compression-level"):
            pass
        # Pass on any other options.
        else:
            new_cmd.append(arg)

    # Now specify the new device architecture
    new_cmd.append(f"--offload-arch={arch}")
    # Only compile the device part of the source file.
    new_cmd.append("--offload-device-only")
    # Pass the CUID also. Note that this is a driver option, not -Xclang: see the
    # comment where it is computed in compile_parallel().
    new_cmd.extend(cuid_args)
    # Don't package the output in an offload bundle for us: We're going to manually put
    # all of the architectures together, this saves a few unbundling steps.
    new_cmd.append(f"--no-gpu-bundle-output")
    # And pass the new output.
    new_cmd.append("-o")
    new_cmd.append(output)

    start = time.time()

    try:
        # Run command. Take care to release any slots even if there was a compile error.
        await run(new_cmd)
    except subprocess.CalledProcessError:
        error_event.set()
    finally:
        end = time.time()
        trace(start, end, f"{host_output}::{arch}")

async def preprocess_only(cmd):
    """
    Run an (offload) compilation in preprocessor-only mode. This is a special path in
    PHC because there are some issues related to this in sccache[2] and clang[3], which
    we can easily fix up here for the time being.

    [2]: https://github.com/mozilla/sccache/issues/2762
    [3]: https://github.com/llvm/llvm-project/issues/207375
    """
    it = iter(cmd)
    new_cmd = []
    for arg in it:
        # Get rid of any compression flags. Since this is preprocessing-only, it shouldn't
        # affect the output (other than not producing compressed data on the stdout).
        if arg == "--offload-compress":
            pass
        # Don't compile with the new offload driver even if we're asked explicitly.
        elif arg == "--offload-new-driver":
            pass
        else:
            new_cmd.append(arg)

    # Don't compile with the new offload driver.
    new_cmd.append("--no-offload-new-driver")

    await run(new_cmd)

if __name__ == "__main__":
    # Look for the jobserver *before* the event loop starts. If make advertised
    # descriptors it did not actually pass, they are free numbers, and asyncio would
    # be handed exactly those for its own wakeup socket pair -- at which point they
    # would look inherited. See JobserverClient.detect().
    JOBSERVER = JobserverClient.detect()

    # Run the main function to completion using asyncio.
    asyncio.run(main())
