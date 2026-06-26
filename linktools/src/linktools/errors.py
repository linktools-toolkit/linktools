#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""linktools exception hierarchy."""


class Error(Exception):
    """Base exception for linktools-specific errors."""


class ModuleError(Error):
    """Raised when a linktools module cannot be loaded or used."""


class DownloadError(Error):
    """Base exception for download failures."""


class ExecError(Error):
    """Base exception for process execution failures."""


class DownloadHttpError(DownloadError):
    """Download error that carries an HTTP status code."""

    def __init__(self, code, e):
        super().__init__(e)
        self.code = code


class ConfigError(Error):
    """Raised when configuration data is invalid or unavailable."""


class ToolError(Error):
    """Base exception for tool discovery and execution failures."""


class ToolNotFound(ToolError):
    """Raised when a requested tool cannot be found."""


class ToolNotSupport(ToolError):
    """Raised when a tool is not supported in the current environment."""


class ToolExecError(ToolError):
    """Raised when a tool process exits with an execution error."""


class NoFreePortFoundError(Error):
    """Exception indicating that no free port could be found."""
