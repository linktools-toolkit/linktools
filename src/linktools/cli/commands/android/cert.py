#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author  : Hu Ji
@file    : cert.py 
@time    : 2023/07/14
@site    :  
@software: PyCharm 

              ,----------------,              ,---------,
         ,-----------------------,          ,"        ,"|
       ,"                      ,"|        ,"        ,"  |
      +-----------------------+  |      ,"        ,"    |
      |  .-----------------.  |  |     +---------+      |
      |  |                 |  |  |     | -==----'|      |
      |  | $ sudo rm -rf / |  |  |     |         |      |
      |  |                 |  |  |/----|`---=    |      |
      |  |                 |  |  |   ,/|==== ooo |      ;
      |  |                 |  |  |  // |(((( [33]|    ,"
      |  `-----------------'  |," .;'| |((((     |  ,"
      +-----------------------+  ;;  | |         |,"
         /_)______________(_/  //'   | +---------+
    ___________________________/___  `,
   /  oooooooooooooooo  .o.  oooo /,   `,"-----------
  / ==ooooooooooooooo==.o.  ooo= //   ,``--{)B     ,"
 /_==__==========__==_ooo__ooo=_/'   /___________,"
"""

import hashlib
import os
from argparse import ArgumentParser, Namespace
from datetime import datetime
from typing import Optional, Type, List, Union

import OpenSSL
from rich import get_console
from rich.table import Table

from linktools import utils
from linktools.cli import subcommand, subcommand_argument, AndroidCommand


class Command(AndroidCommand):
    """
    Display detailed X.509 certificate information for secure communication
    """

    @property
    def known_errors(self) -> List[Type[BaseException]]:
        return super().known_errors + [OpenSSL.crypto.Error]

    def init_arguments(self, parser: ArgumentParser) -> None:
        self.add_subcommands(parser)

    def run(self, args: Namespace) -> Optional[int]:
        subcommand = self.parse_subcommand(args)
        if not subcommand:
            return self.print_subcommands(args)
        return subcommand.run(args)

    @subcommand("info", help="display certificate information")
    @subcommand_argument("path", help="cert path")
    def on_info(self, path: str, test: int):
        def format_date(date: str):
            date = datetime.strptime(date, '%Y%m%d%H%M%SZ')
            return date.strftime('%Y-%m-%d %H:%M:%S')

        def format_hex(data: Union[int, bytes], length: int = None):
            result = f"{data:x}"
            if len(result) % 2 != 0:
                result = f"0{result}"
            if length is not None and len(result) < length:
                result = result.zfill(length)
            return f"0x{result}"

        def format_components(issuer: OpenSSL.SSL.X509Name):
            components = []
            for item in issuer.get_components():
                key = item[0].decode("utf-8")
                value = item[1].decode("utf-8")
                components.append(f"{key}={value}")
            return ", ".join(components)

        def dump_pubkey(cert: OpenSSL.SSL.X509):
            return OpenSSL.crypto.dump_publickey(OpenSSL.crypto.FILETYPE_PEM, cert.get_pubkey())

        def subject_name_hash_old(cert: OpenSSL.SSL.X509):
            subject_name = cert.get_subject().der()
            hash_obj = hashlib.md5(subject_name)
            hash_value = hash_obj.digest()
            hash_string = ''.join(f"{b:02x}" for b in reversed(hash_value[:4]))
            return f"0x{hash_string}"

        cert = OpenSSL.crypto.load_certificate(
            OpenSSL.crypto.FILETYPE_PEM,
            utils.read_file(os.path.expanduser(path), text=False)
        )

        issuer = cert.get_issuer()

        table = Table(show_lines=True)
        table.add_column("Key", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        table.add_row("Version", f"{cert.get_version() + 1} ({format_hex(cert.get_version())})")
        table.add_row("Serial Number", f"{format_hex(cert.get_serial_number())}")
        table.add_row("Subject Hash", f"{format_hex(cert.subject_name_hash(), 8)}")
        table.add_row("Subject Hash (Old)", f"{subject_name_hash_old(cert)}")
        table.add_row("Signature Algorithm", cert.get_signature_algorithm().decode("UTF-8"))
        table.add_row("Common Name", issuer.commonName)
        table.add_row("Not Before", format_date(cert.get_notBefore().decode("UTF-8")))
        table.add_row("Not After", format_date(cert.get_notAfter().decode("UTF-8")))
        table.add_row("Has Expired", "true" if cert.has_expired() else "false")
        table.add_row("Components", format_components(issuer))
        table.add_row("Pubkey Bits", str(cert.get_pubkey().bits()))
        table.add_row("Pubkey", dump_pubkey(cert).decode("utf-8"))

        console = get_console()
        console.print(table)

    @subcommand("install", help="start setting activity", pass_args=True)
    @subcommand_argument("path", help="cert path")
    def on_install(self, args: Namespace, path: str):
        device = args.device_picker.pick()
        dest = device.push_file(path, device.get_data_path("cert"), log_output=True)
        device.shell("am", "start", "--user", "0",
                     "-n", "com.android.certinstaller/.CertInstallerMain",
                     "-a", "android.intent.action.VIEW",
                     "-t", "application/x-x509-ca-cert",
                     "-d", "file://%s" % dest,
                     log_output=True)


command = Command()
if __name__ == "__main__":
    command.main()
