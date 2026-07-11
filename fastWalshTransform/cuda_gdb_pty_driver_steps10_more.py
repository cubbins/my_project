#!/usr/bin/env python3
import os
import pty
import select
import signal
import subprocess
import sys
import time

PROMPT = "(cuda-gdb)"
READ_TIMEOUT = 20.0
STEP_COUNT = 40

class PtyGdbDriver:
    def __init__(self, argv, log_path="cuda_gdb_pty_steps10_ops.log"):
        self.argv = argv
        self.log_path = log_path
        self.master_fd = None
        self.slave_fd = None
        self.proc = None
        self.logf = None

    def start(self):
        self.master_fd, self.slave_fd = pty.openpty()
        self.logf = open(self.log_path, "w", encoding="utf-8", errors="replace")
        self.proc = subprocess.Popen(
            self.argv,
            stdin=self.slave_fd,
            stdout=self.slave_fd,
            stderr=self.slave_fd,
            text=False,
            close_fds=True,
            preexec_fn=os.setsid,
        )
        os.close(self.slave_fd)
        self.slave_fd = None

    def stop(self):
        try:
            if self.proc and self.proc.poll() is None:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
        except Exception:
            pass
        try:
            if self.master_fd is not None:
                os.close(self.master_fd)
                self.master_fd = None
        except Exception:
            pass
        try:
            if self.logf:
                self.logf.close()
                self.logf = None
        except Exception:
            pass

    def _read_some(self, timeout=0.25):
        out = b""
        rlist, _, _ = select.select([self.master_fd], [], [], timeout)
        if self.master_fd in rlist:
            try:
                chunk = os.read(self.master_fd, 65536)
                if chunk:
                    out += chunk
            except OSError:
                pass
        return out

    def read_until_prompt(self, timeout=READ_TIMEOUT):
        deadline = time.time() + timeout
        buf = b""
        while time.time() < deadline:
            chunk = self._read_some()
            if chunk:
                buf += chunk
                text = chunk.decode("utf-8", errors="replace")
                sys.stdout.write(text)
                sys.stdout.flush()
                self.logf.write(text)
                self.logf.flush()
                if PROMPT in buf.decode("utf-8", errors="replace"):
                    return buf.decode("utf-8", errors="replace")
            if self.proc.poll() is not None:
                break
        return buf.decode("utf-8", errors="replace")

    def send_line(self, line, expect_prompt=True, timeout=READ_TIMEOUT):
        print(f"\n[pty-driver] cmd: {line}")
        os.write(self.master_fd, (line + "\n").encode("utf-8"))
        if expect_prompt:
            return self.read_until_prompt(timeout)
        return ""

def main():
    if len(sys.argv) < 2:
        print("usage: python3 cuda_gdb_pty_driver_steps10_with_ops.py ./fastWalshTransform_debug")
        sys.exit(1)

    target = sys.argv[1]
    d = PtyGdbDriver(["cuda-gdb", target])

    try:
        d.start()
        d.read_until_prompt(timeout=30.0)

        for cmd in [
            "set pagination off",
            "set confirm off",
            "set cuda break_on_launch application",
            "set cuda launch_blocking on",
            "set cuda kernel_events application",
            "set cuda context_events on",
            "set cuda single_stepping_optimizations off",
            "set cuda step_divergent_lanes on",
        ]:
            d.send_line(cmd)

        d.send_line("run", expect_prompt=False)
        d.read_until_prompt(timeout=60.0)

        for cmd in [
            "cuda thread 1",
            "info cuda kernels",
            "where",
            "frame",
            "info frame",
            "info args",
            "info locals",
            "list",
            "info source",
            "info registers",
            "info registers pc",
            "x/5i $pc",
        ]:
            d.send_line(cmd)

        for i in range(STEP_COUNT):
            d.send_line("stepi")
            d.send_line("info registers")
            d.send_line("info registers pc")
            d.send_line("x/5i $pc")

        d.send_line("quit", expect_prompt=False)
        time.sleep(1.0)
        print("\nSaved log: cuda_gdb_pty_steps10_ops.log")

    finally:
        d.stop()

if __name__ == "__main__":
    main()