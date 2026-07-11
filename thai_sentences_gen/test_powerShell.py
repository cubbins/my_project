#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import textwrap
from datetime import datetime
from pathlib import Path


REPORT_FILE = "test_powershell_environment_report.json"

###############################################################################
# SQL Server configuration
###############################################################################
SQL_TEST_SERVERS = [
    "localhost",
    "127.0.0.1",
    "localhost,1433",
    "127.0.0.1,1433",
    "10.0.0.20,63451",
]

SQL_DATABASE = os.environ.get("MSSQL_DATABASE", "master")
SQL_USERNAME = os.environ.get("MSSQL_USERNAME")
SQL_PASSWORD = os.environ.get("MSSQL_PASSWORD")

SQL_DATABASE = os.environ.get(
    "MSSQL_DATABASE",
    "master",
)

SQL_USERNAME = os.environ.get(
    "MSSQL_USERNAME"
)

SQL_PASSWORD = os.environ.get(
    "MSSQL_PASSWORD"
)

"""
# --- SQL Server connection ---
export MSSQL_SERVER="10.0.0.20,63451"
export MSSQL_DATABASE="mov"
export MSSQL_USERNAME="cubbins"
export MSSQL_PASSWORD="Verne123!"
"""

class ReportPrinter:
    def __init__(self, width: int = 80):
        self.width = width

    def title(self, text: str) -> None:
        print()
        print("=" * self.width)
        print(text.center(self.width))
        print("=" * self.width)

    def section(self, text: str) -> None:
        print()
        print("-" * self.width)
        print(text)
        print("-" * self.width)

    def item(self, name: str, value, label_width: int = 24) -> None:
        value = "" if value is None else str(value)
        prefix = f"{name:<{label_width}} : "

        wrapped = textwrap.wrap(
            value,
            width=max(20, self.width - len(prefix)),
            subsequent_indent=" " * len(prefix),
        )

        if not wrapped:
            print(prefix)
            return

        print(prefix + wrapped[0])
        for line in wrapped[1:]:
            print(line)

    def command(self, title: str, result: dict) -> None:
        self.section(title)
        self.item("Command", result.get("command", ""))

        if result.get("returncode") is not None:
            self.item("Return code", result.get("returncode"))

        if result.get("stdout"):
            print()
            print("Output:")
            print(textwrap.indent(result["stdout"], "    "))

        if result.get("stderr"):
            print()
            print("Errors:")
            print(textwrap.indent(result["stderr"], "    "))

        if result.get("error"):
            self.item("Error", result["error"])

    def list_items(self, title: str, values: list) -> None:
        self.section(title)

        if not values:
            print("    None")
            return

        for value in values:
            print(
                textwrap.fill(
                    f"• {value}",
                    width=self.width,
                    initial_indent="    ",
                    subsequent_indent="      ",
                )
            )

    def dict_items(self, title: str, values: dict) -> None:
        self.section(title)

        if not values:
            print("    None")
            return

        for key, value in values.items():
            self.item(str(key), value)

    def sql_server(self, sql: dict) -> None:
        self.title("SQL SERVER ENVIRONMENT")

        self.item("pyodbc available", sql.get("pyodbc_available"))
        self.item("sqlcmd available", sql.get("sqlcmd_available"))

        if sql.get("pyodbc_error"):
            self.item("pyodbc error", sql.get("pyodbc_error"))

        self.list_items("ODBC Drivers", sql.get("odbc_drivers", []))
        self.dict_items("Common SQL Ports", sql.get("common_ports", {}))
        self.connection_table(sql.get("connection_tests", []))

    def connection_table(self, tests: list) -> None:
        self.section("SQL Server Connection Tests")

        if not tests:
            print("    None")
            return

        print(f"{'Server':25} {'Status':12} Driver")
        print("-" * self.width)

        for test in tests:
            print(
                f"{test.get('server', '')[:25]:25} "
                f"{test.get('status', '')[:12]:12} "
                f"{test.get('driver', '')}"
            )

            if test.get("error"):
                print(
                    textwrap.fill(
                        test["error"],
                        width=self.width,
                        initial_indent="    Error: ",
                        subsequent_indent="           ",
                    )
                )

            if test.get("status") == "connected":
                self.item("Server name", test.get("server_name"))
                self.item("Current database", test.get("database_name"))
                self.item("Database count", test.get("database_count"))

                databases = test.get("databases", [])

                if databases:
                    print()
                    print("    Databases")
                    print("    " + "-" * 68)
                    print(f"    {'ID':>4} {'Name':30} {'State':12} Recovery")
                    print("    " + "-" * 68)

                    for db in databases:
                        print(
                            f"    {db.get('database_id', ''):>4} "
                            f"{db.get('name', '')[:30]:30} "
                            f"{db.get('state_desc', '')[:12]:12} "
                            f"{db.get('recovery_model_desc', '')}"
                        )
                else:
                    print("    Databases: None returned")

def connection_table(self, tests: list) -> None:
    self.section("SQL Server Connection Tests")

    if not tests:
        print("    None")
        return

    print(f"{'Server':25} {'Status':12} Driver")
    print("-" * self.width)

    for test in tests:
        print(
            f"{test.get('server', '')[:25]:25} "
            f"{test.get('status', '')[:12]:12} "
            f"{test.get('driver', '')}"
        )

        if test.get("error"):
            print(
                textwrap.fill(
                    test["error"],
                    width=self.width,
                    initial_indent="    Error: ",
                    subsequent_indent="           ",
                )
            )

        if test.get("status") == "connected":
            self.item("Server name", test.get("server_name"))
            self.item("Current database", test.get("database_name"))
            self.item("Database count", test.get("database_count"))

            databases = test.get("databases", [])

            if databases:
                print()
                print("    Databases")
                print("    " + "-" * 68)
                print(f"    {'ID':>4} {'Name':30} {'State':12} Recovery")
                print("    " + "-" * 68)

                for db in databases:
                    print(
                        f"    {db.get('database_id', ''):>4} "
                        f"{db.get('name', '')[:30]:30} "
                        f"{db.get('state_desc', '')[:12]:12} "
                        f"{db.get('recovery_model_desc', '')}"
                    )
            else:
                print("    Databases: None returned")

    def sql_server(self, sql: dict) -> None:
        self.title("SQL SERVER ENVIRONMENT")
        self.item("pyodbc available", sql.get("pyodbc_available"))
        self.item("sqlcmd available", sql.get("sqlcmd_available"))

        if sql.get("pyodbc_error"):
            self.item("pyodbc error", sql["pyodbc_error"])

        self.list_items("ODBC Drivers", sql.get("odbc_drivers", []))
        self.dict_items("Common SQL Ports", sql.get("common_ports", {}))
        self.connection_table(sql.get("connection_tests", []))


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
        "sqlcmd_available": shutil.which("sqlcmd")
        or shutil.which("/opt/mssql-tools18/bin/sqlcmd"),
        "common_ports": {},
        "connection_tests": [],
        "configuration": {
            "servers": SQL_TEST_SERVERS,
            "database": SQL_DATABASE,
            "username_supplied": bool(SQL_USERNAME),
            "password_supplied": bool(SQL_PASSWORD),
        },
    }

    try:
        import pyodbc

        sql["pyodbc_available"] = True
        sql["odbc_drivers"] = pyodbc.drivers()
    except Exception as exc:
        sql["pyodbc_error"] = f"{type(exc).__name__}: {exc}"
        return sql

    if SQL_USERNAME and SQL_PASSWORD:
        auth_part = f"UID={SQL_USERNAME};PWD={SQL_PASSWORD};"
        auth_name = "SQL Login"
    else:
        auth_part = "Trusted_Connection=yes;"
        auth_name = "Trusted_Connection"

    drivers_to_test = [
        "ODBC Driver 18 for SQL Server",
        "ODBC Driver 17 for SQL Server",
    ]

    installed_drivers = set(sql["odbc_drivers"])

    port_targets = [
        ("localhost", 1433),
        ("127.0.0.1", 1433),
        ("localhost", 1434),
    ]

    for server in SQL_TEST_SERVERS:
        if "," in server:
            host, port_text = server.rsplit(",", 1)
            if port_text.isdigit():
                port_targets.append((host, int(port_text)))

    seen_ports = set()

    for host, port in port_targets:
        key = f"{host}:{port}"

        if key in seen_ports:
            continue

        seen_ports.add(key)

        try:
            with socket.create_connection((host, port), timeout=2):
                sql["common_ports"][key] = "open"
        except Exception as exc:
            sql["common_ports"][key] = (
                f"closed/unreachable: {type(exc).__name__}"
            )

    for driver in drivers_to_test:
        if driver not in installed_drivers:
            continue

        for server in SQL_TEST_SERVERS:
            display_conn_str = (
                f"DRIVER={{{driver}}};"
                f"SERVER={server};"
                f"DATABASE={SQL_DATABASE};"
                f"AUTH={auth_name};"
                "Encrypt=no;"
                "TrustServerCertificate=yes;"
                "Connection Timeout=5;"
            )

            conn_str = (
                f"DRIVER={{{driver}}};"
                f"SERVER={server};"
                f"DATABASE={SQL_DATABASE};"
                f"{auth_part}"
                "Encrypt=no;"
                "TrustServerCertificate=yes;"
                "Connection Timeout=5;"
            )

            test = {
                "driver": driver,
                "server": server,
                "database": SQL_DATABASE,
                "auth": auth_name,
                "connection_string_without_password": display_conn_str,
            }

            try:
                conn = pyodbc.connect(conn_str, timeout=5)
                cur = conn.cursor()

                cur.execute(
                    """
                    SELECT
                        @@SERVERNAME AS server_name,
                        DB_NAME() AS database_name,
                        @@VERSION AS version_text;
                    """
                )

                row = cur.fetchone()

                test["status"] = "connected"
                test["server_name"] = row.server_name
                test["database_name"] = row.database_name
                test["version_text"] = row.version_text

                cur.execute(
                    """
                    SELECT
                        name,
                        database_id,
                        create_date,
                        state_desc,
                        recovery_model_desc
                    FROM sys.databases
                    ORDER BY database_id;
                    """
                )

                databases = []

                for db_row in cur.fetchall():
                    databases.append(
                        {
                            "name": db_row.name,
                            "database_id": db_row.database_id,
                            "create_date": str(db_row.create_date),
                            "state_desc": db_row.state_desc,
                            "recovery_model_desc": db_row.recovery_model_desc,
                        }
                    )

                test["databases"] = databases
                test["database_count"] = len(databases)

                conn.close()

            except Exception as exc:
                test["status"] = "failed"
                test["error"] = f"{type(exc).__name__}: {exc}"

            sql["connection_tests"].append(test)

    return sql

def main() -> None:
    printer = ReportPrinter(width=80)

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

    printer.title("POWERSHELL / UBUNTU ENVIRONMENT SURVEY")
    printer.item("Timestamp", report["timestamp"])
    printer.item("Current directory", report["current_directory"])

    printer.section("Python")
    printer.item("Version", report["python"]["version"])
    printer.item("Executable", report["python"]["executable"])

    printer.section("Platform")
    for key, value in report["platform"].items():
        printer.item(key, value)

    printer.dict_items(
        "Environment Variables",
        report["environment_variables_subset"],
    )

    if not powershell_available():
        report["powershell"]["install_command"] = powershell_install_hint()

        printer.title("POWERSHELL NOT INSTALLED")
        printer.item("Available", False)
        print()
        print(report["powershell"]["install_command"])

    else:
        report["powershell"]["version"] = run_pwsh(
            "$PSVersionTable | ConvertTo-Json -Depth 4"
        )

        printer.title("POWERSHELL ENVIRONMENT")
        printer.item("Available", report["powershell"]["available"])
        printer.item("Path", report["powershell"]["path"])
        printer.command("PowerShell Version", report["powershell"]["version"])

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
            "capabilities": run_pwsh(
                "getcap -r /usr/bin /bin 2>$null | Select-Object -First 50"
            ),
            "listening_processes": run_pwsh(
                "ss -tulpn 2>$null | Select-Object -First 100"
            ),
            "firewall_ufw": run_pwsh("sudo ufw status 2>$null"),
            "apparmor_status": run_pwsh("aa-status 2>$null"),
        }

        printer.title("LINUX SECURITY ENVIRONMENT")
        for name, result in report["linux_security_environment"].items():
            printer.command(name, result)

        report["network_environment"] = {
            "hostname": run_pwsh("hostname"),
            "ip_addresses": run_pwsh("ip addr"),
            "routes": run_pwsh("ip route"),
            "dns": run_pwsh("cat /etc/resolv.conf"),
            "open_connections": run_pwsh(
                "ss -tunap 2>$null | Select-Object -First 100"
            ),
        }

        printer.title("NETWORK ENVIRONMENT")
        for name, result in report["network_environment"].items():
            printer.command(name, result)

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

        printer.title("FILESYSTEM ENVIRONMENT")
        for name, result in report["filesystem_environment"].items():
            printer.command(name, result)

        report["process_environment"] = {
            "top_processes": run_pwsh("ps aux --sort=-%mem | head -30"),
            "system_info": run_pwsh("uname -a"),
            "os_release": run_pwsh("cat /etc/os-release"),
        }

        printer.title("PROCESS ENVIRONMENT")
        for name, result in report["process_environment"].items():
            printer.command(name, result)

        report["sql_server_environment"] = probe_sql_server()
        printer.sql_server(report["sql_server_environment"])

    Path(REPORT_FILE).write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    printer.title("REPORT SAVED")
    printer.item("JSON file", REPORT_FILE)


if __name__ == "__main__":
    main()