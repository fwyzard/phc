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
from queue import Queue

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

    # When compiling different compilation units separately, each one needs a 'CUID'
    # passed to it to help identify which compilation unit an object is part of.
    # Usually this is passed by the clang driver, but since we are emulating the driver
    # we need to pass it ourselves.
    cuid = os.urandom(8).hex()

    # Try to find the offload bundler. Its usually next to the compiler, but be sure
    # to resolve any symlinks first (for example if the compiler is /usr/bin/hipcc).
    clang_dir = os.path.dirname(os.path.realpath(cmd[0]))
    offload_bundler = os.path.join(clang_dir, "clang-offload-bundler")
    if not os.path.isfile(offload_bundler):
        offload_bundler = os.path.join(clang_dir, "..", "llvm", "bin", "clang-offload-bundler")
    if not os.path.isfile(offload_bundler):
        raise ValueError("could not find clang-offload-bundler")

    # Figure out the main output file. We'll mainly use this for logging trace info about
    # when a file was compiled.

    # Figure out some common things from the compile command:
    # - The main output file. We'll mainly use this for logging trace info about when a file
    #   was compiled.
    # - The offload compression level. Because we're packaging the offload bundle manually,
    #   we'll have to pass relevant parameters to that command.

    it = iter(cmd)
    host_output = "a.out"
    offload_compress = False
    offload_compression_level = None
    for arg in it:
        if arg == "-o":
           host_output = next(it)
        elif arg.startswith("-o"):
           host_output = arg[2:]
        elif arg == "--offload-compress":
            offload_compress = True
        elif arg.startswith('--offload-compression-level='):
            offload_compression_level = arg.split('=')[-1]

    with open_jobserver(len(offload_archs)) as jobclient:
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

            # Create the compilation tasks.
            for arch in offload_archs:
                output = os.path.join(dir, f"{arch}.out")
                tasks.put_nowait(compile_device(cmd, arch, output, cuid, host_output, error_event))

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
            # check here if its set.
            if error_event.is_set():
                # Note: error is already printed.
                sys.exit(1)

            start = time.time()

            # The remainder of the commands are all serially executed within the same jobserver
            # task, the main thread of this process.

            # Now manually combine the inputs into an offload bundle.
            # Note: We also have to pass the host target (x86_64 most likely) and corresponding
            # input (-input=/dev/null). This is not used by HIP internally, but the bundle still
            # needs the entry for some reason.
            bundle = os.path.join(dir, "bundle.hipfb")
            bundle_cmd = [offload_bundler, "-type=o", "-bundle-align=4096", f"-output={bundle}", "-targets=host-x86_64-unknown-linux-"]
            # Append the device targets for each architecture.
            bundle_cmd[-1] += ''.join([f",hipv4-amdgcn-amd-amdhsa--{arch}" for arch in offload_archs])
            bundle_cmd.append("-input=/dev/null")
            # And the device code objects for each architecture, in the same order.
            bundle_cmd.extend(["-input=" + os.path.join(dir, f"{arch}.out") for arch in offload_archs])

            # Add in the compression options, if available.
            if offload_compress:
                bundle_cmd.append("-compress")
            if offload_compression_level is not None:
                bundle_cmd.append(f"-compression-level={offload_compression_level}")

            await run(bundle_cmd)

            bundle_end = time.time()
            trace(start, bundle_end, f"{host_output}::bundle")

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
            # Also pass the CUID to cc1 with the same method.
            host_cmd.append("-Xclang")
            host_cmd.append(f"-cuid={cuid}")
            await run(host_cmd)

            host_end = time.time()
            trace(bundle_end, host_end, host_output)

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

async def compile_device(cmd, arch, output, cuid, host_output, error_event):
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
    cuid:
        The Compilation Unit ID. Must be the same for the host and device compilation of the
        same compilation unit.
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
        elif arg == "-MT" or arg == "-MQ" or arg == "-MF":
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
    # Pass the CUID also.
    new_cmd.append("-Xclang")
    new_cmd.append(f"-cuid={cuid}")
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
