

#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
import socket

import textwrap



REPORT_FILE = "test_powershell_environment_report.json"

def print_title(title: str) -> None:
    print()
    print("=" * 80)
    print(title.center(80))
    print("=" * 80)

def print_item(name, value, width=24):
    print(f"{name:<{width}} : {value}")

def print_wrapped(text, indent=4):
    wrapped = textwrap.fill(
        text,
        width=78,
        subsequent_indent=" " * indent
    )
    print(wrapped)

def print_list(title, values):
    print(title)
    print("-" * len(title))

    if not values:
        print("    None")
        return

    for item in values:
        print(f"    • {item}")


def print_command(result):
    print(f"Command : {result['command']}")
    print()

    if result.get("stdout"):
        print("Output")
        print("------")
        print(result["stdout"])

    if result.get("stderr"):
        print()
        print("Errors")
        print("------")
        print(result["stderr"])

def print_connection_table(tests):

    print(f"{'Server':25} {'Status':12} Driver")
    print("-" * 70)

    for t in tests:
        print(
            f"{t['server'][:25]:25} "
            f"{t['status'][:12]:12} "
            f"{t['driver']}"
        )





def run(cmd: list[str], timeout: int = 20) -> dict:
    try:
        result = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return {
            "command": " ".join(cmd),
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except Exception as exc:
        return {
            "command": " ".join(cmd),
            "error": f"{type(exc).__name__}: {exc}",
        }


def powershell_available() -> bool:
    return shutil.which("pwsh") is not None


def powershell_install_hint() -> str:
    return """
sudo apt-get update
sudo apt-get install -y wget apt-transport-https software-properties-common

source /etc/os-release
wget -q https://packages.microsoft.com/config/ubuntu/$VERSION_ID/packages-microsoft-prod.deb

sudo dpkg -i packages-microsoft-prod.deb
rm packages-microsoft-prod.deb

sudo apt-get update
sudo apt-get install -y powershell
"""


def run_pwsh(script: str) -> dict:
    return run(["pwsh", "-NoProfile", "-Command", script], timeout=30)


def probe_sql_server() -> dict:
    sql = {
        "pyodbc_available": False,
        "odbc_drivers": [],
        "sqlcmd_available": shutil.which("sqlcmd") or shutil.which("/opt/mssql-tools18/bin/sqlcmd"),
        "common_ports": {},
        "connection_tests": [],
    }

    try:
        import pyodbc
        sql["pyodbc_available"] = True
        sql["odbc_drivers"] = pyodbc.drivers()
    except Exception as exc:
        sql["pyodbc_error"] = f"{type(exc).__name__}: {exc}"
        return sql

    for host, port in [
        ("localhost", 1433),
        ("127.0.0.1", 1433),
        ("localhost", 1434),
    ]:
        try:
            with socket.create_connection((host, port), timeout=2):
                sql["common_ports"][f"{host}:{port}"] = "open"
        except Exception as exc:
            sql["common_ports"][f"{host}:{port}"] = f"closed/unreachable: {type(exc).__name__}"

    drivers = [
        "ODBC Driver 18 for SQL Server",
        "ODBC Driver 17 for SQL Server",
    ]

    servers = [
        "localhost",
        "127.0.0.1",
        "localhost,1433",
        "127.0.0.1,1433",
    ]

    for driver in drivers:
        if driver not in sql["odbc_drivers"]:
            continue

        for server in servers:
            conn_str = (
                f"DRIVER={{{driver}}};"
                f"SERVER={server};"
                "DATABASE=master;"
                "Trusted_Connection=yes;"
                "TrustServerCertificate=yes;"
                "Encrypt=yes;"
                "Connection Timeout=3;"
            )

            test = {
                "driver": driver,
                "server": server,
                "database": "master",
                "auth": "Trusted_Connection",
            }

            try:
                conn = pyodbc.connect(conn_str, timeout=3)
                cur = conn.cursor()
                cur.execute("""
                    SELECT
                        @@SERVERNAME AS server_name,
                        DB_NAME() AS database_name,
                        @@VERSION AS version_text;
                """)
                row = cur.fetchone()
                test["status"] = "connected"
                test["server_name"] = row.server_name
                test["database_name"] = row.database_name
                test["version_text"] = row.version_text
                conn.close()
            except Exception as exc:
                test["status"] = "failed"
                test["error"] = f"{type(exc).__name__}: {exc}"

            sql["connection_tests"].append(test)

    return sql

def print_section(title: str, data) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(json.dumps(data, indent=2, ensure_ascii=False))


def main() -> None:
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "current_directory": str(Path.cwd()),
        "python": {
            "version": platform.python_version(),
            "executable": shutil.which("python3") or shutil.which("python"),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "node": platform.node(),
        },
        "environment_variables_subset": {
            key: os.environ.get(key)
            for key in [
                "USER",
                "HOME",
                "SHELL",
                "PATH",
                "LANG",
                "VIRTUAL_ENV",
                "CONDA_DEFAULT_ENV",
                "WSL_DISTRO_NAME",
            ]
        },
        "powershell": {
            "available": powershell_available(),
            "path": shutil.which("pwsh"),
        },
        "linux_security_environment": {},
        "network_environment": {},
        "filesystem_environment": {},
        "process_environment": {},
        "sql_server_environment": {},
    }

    if not powershell_available():
        report["powershell"]["install_command"] = powershell_install_hint()
        print("PowerShell is not installed.")
        print("Install it with:")
        print(powershell_install_hint())
    else:
        report["powershell"]["version"] = run_pwsh("$PSVersionTable | ConvertTo-Json -Depth 4")
        print_section("PowerShell", report["powershell"])

        report["linux_security_environment"] = {
            "whoami": run_pwsh("whoami"),
            "id": run_pwsh("id"),
            "sudo_access_check": run_pwsh(
                "sudo -n true 2>$null; "
                "if ($LASTEXITCODE -eq 0) { 'sudo without password: yes' } "
                "else { 'sudo without password: no or unavailable' }"
            ),
            "groups": run_pwsh("groups"),
            "umask": run_pwsh("umask"),
            "capabilities": run_pwsh("getcap -r /usr/bin /bin 2>$null | Select-Object -First 50"),
            "listening_processes": run_pwsh("ss -tulpn 2>$null | Select-Object -First 100"),
            "firewall_ufw": run_pwsh("sudo ufw status 2>$null"),
            "apparmor_status": run_pwsh("aa-status 2>$null"),
        }
        print_section("Linux Security Environment", report["linux_security_environment"])

        report["network_environment"] = {
            "hostname": run_pwsh("hostname"),
            "ip_addresses": run_pwsh("ip addr"),
            "routes": run_pwsh("ip route"),
            "dns": run_pwsh("cat /etc/resolv.conf"),
            "open_connections": run_pwsh("ss -tunap 2>$null | Select-Object -First 100"),
        }
        print_section("Network Environment", report["network_environment"])

        report["filesystem_environment"] = {
            "pwd": run_pwsh("Get-Location"),
            "directory_listing": run_pwsh(
                "Get-ChildItem -Force | "
                "Select-Object Mode,Length,LastWriteTime,Name | "
                "ConvertTo-Json -Depth 3"
            ),
            "mnt_c_listing": run_pwsh("ls -la /mnt/c"),
            "disk_usage": run_pwsh("df -h"),
            "mounts": run_pwsh("mount | Select-Object -First 100"),
        }
        print_section("Filesystem Environment", report["filesystem_environment"])

        report["process_environment"] = {
            "top_processes": run_pwsh("ps aux --sort=-%mem | head -30"),
            "system_info": run_pwsh("uname -a"),
            "os_release": run_pwsh("cat /etc/os-release"),
        }
        print_section("Process Environment", report["process_environment"])

        report["sql_server_environment"] = probe_sql_server()
        print_section("SQL Server Environment", report["sql_server_environment"])

    Path(REPORT_FILE).write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print_section("Complete Environment Report", report)
    print(f"\nEnvironment report written to: {REPORT_FILE}")

if __name__ == "__main__":
    main()


