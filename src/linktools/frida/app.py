#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author    : HuJi <jihu.hj@alibaba-inc.com>
# Datetime  : 2021/12/20 3:41 下午
# User      : huji
# Product   : PyCharm
# Project   : link

import abc
import json
import logging
import os
import re
import threading
from typing import TYPE_CHECKING, Optional, Union, Dict, Collection, Callable, Any, List

import frida
from frida.core import Session, Script

from .script import FridaUserScript, FridaEvalCode, FridaScriptFile
from .server import FridaServer
from .. import utils, environ
from ..decorator import timeoutable
from ..metadata import __release__
from ..reactor import Reactor
from ..types import TimeoutType, Stoppable

if TYPE_CHECKING:
    import _frida

_logger = environ.get_logger("frida.app")


class FridaReactor(Reactor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cancellable = frida.Cancellable()

    def _work(self, fn: Callable[[], any]):
        with self.cancellable:
            fn()


class FridaSession(utils.get_derived_type(Session)):  # proxy for frida.core.Session

    __super__: Session

    def __init__(self, session: Session, name: str = None):
        super().__init__(session)
        self._name: str = name or ""
        self._scripts: [FridaScript] = []

    @property
    def pid(self) -> int:
        return self._impl.pid

    @property
    def name(self) -> str:
        return self._name

    @property
    def scripts(self) -> ["FridaScript"]:
        return self._scripts

    @property
    def script(self) -> Optional["FridaScript"]:
        return self._scripts[0] if self._scripts else None

    @property
    def is_detached(self) -> bool:
        if hasattr(self.__super__, "is_detached"):
            return self.__super__.is_detached
        return False

    def append(self, script: "FridaScript"):
        if script not in self._scripts:
            self._scripts.append(script)

    def pop(self) -> Optional["FridaScript"]:
        if self._scripts:
            return self._scripts.pop()
        return None

    def __repr__(self):
        return f"Session(pid={self.pid}, name={self.name})" \
            if self.name \
            else f"Session(pid={self.pid})"

    __str__ = __repr__


class FridaScript(utils.get_derived_type(Script)):  # proxy for frida.core.Script

    __super__: Script

    def __init__(self, session: FridaSession, script: Script):
        super().__init__(script)
        self._session: FridaSession = session
        self._session.append(self)

    @property
    def session(self) -> FridaSession:
        return self._session

    @property
    def exports_sync(self) -> "frida.core.ScriptExportsSync":
        if hasattr(self.__super__, "exports_sync"):
            return self.__super__.exports_sync
        return self.__super__.exports

    def __repr__(self):
        return f"Script(pid={self.session.pid}, name={self.session.name})" \
            if self.session.name \
            else f"Script(pid={self.session.pid})"

    __str__ = __repr__


class FridaDeviceHandler(metaclass=abc.ABCMeta):

    def on_spawn_added(self, spawn: "_frida.Spawn"):
        """
        spaw进程添加回调，默认resume所有spawn进程
        :param spawn: spawn进程信息
        """
        _logger.debug(f"{spawn} added")

    def on_spawn_removed(self, spawn: "_frida.Spawn"):
        """
        spaw进程移除回调，默认只打印log
        :param spawn: spawn进程信息
        """
        _logger.debug(f"{spawn} removed")

    def on_child_added(self, child: "_frida.Child"):
        """
        子进程添加回调，默认resume所有子进程
        :param child: 子进程信息
        """
        _logger.debug(f"{child} added")

    def on_child_removed(self, child: "_frida.Child"):
        """
        子进程移除回调，默认只打印log
        :param child: 子进程信息
        """
        _logger.debug(f"{child} removed")

    def on_output(self, pid: int, fd, data):
        _logger.debug(f"Output: pid={pid}, fd={fd}, data={data}")

    def on_device_lost(self):
        _logger.info("Device lost")


class FridaSessionHandler(metaclass=abc.ABCMeta):

    def on_session_attached(self, session: FridaSession):
        """
        会话建立连接回调函数，默认只打印log
        :param session: 附加的会话
        """
        _logger.info(f"{session} attached")

    def on_session_detached(self, session: FridaSession, reason: str, crash: "_frida.Crash"):
        """
        会话结束回调函数，默认只打印log
        :param session: 结束的会话
        :param reason: 结束原因
        :param crash: crash信息
        """
        _logger.info(f"{session} detached, reason={reason}")


class FridaScriptHandler(metaclass=abc.ABCMeta):
    class LogLevel:
        DEBUG = "debug"
        INFO = "info"
        WARNING = "warning"
        ERROR = "error"

    def on_script_message(self, script: FridaScript, message: Any, data: Any):
        """
        脚本消息回调函数，默认按照格式打印
        :param script: frida的脚本
        :param message: frida server发送的数据
        :param data: frida server发送的data
        """

        if utils.get_item(message, "type") == "send":
            payload = utils.get_item(message, "payload")
            if payload and isinstance(payload, dict):

                # 单独解析Emitter发出来的消息
                events: Optional[List[Any]] = payload.pop("$events", None)
                if events:
                    for event in events:
                        # 如果消息类型是log，那就直接调on_log
                        log = event.pop("log", None)
                        if log is not None:
                            level = log.get("level") or self.LogLevel.DEBUG
                            message = log.get("message")
                            self.on_script_log(script, level, message, data)

                        # 如果只是普通消息，则调用on_event
                        msg = event.pop("msg", None)
                        if msg is not None:
                            try:
                                self.on_script_event(script, msg, data)
                            except Exception as e:
                                _logger.error(f"on_script_event error", exc_info=e)

                        # 如果是error消息，则调用on_log展示stack
                        error = event.pop("error", None)
                        if error is not None:
                            stack = utils.get_item(error, "stack")
                            self.on_script_log(script, self.LogLevel.ERROR, stack if stack else error, data)

                    return

            # 其他类型调用on_script_send方法解析
            if payload or data:
                self.on_script_send(script, payload, data)
                return

        elif utils.get_item(message, "type") == "error":
            stack = utils.get_item(message, "stack")
            self.on_script_log(script, self.LogLevel.ERROR, stack if stack else message, data)
            return

        else:
            self.on_script_log(script, self.LogLevel.WARNING, message, data)
            return

    def on_script_log(self, script: FridaScript, level: str, message: Any, data: Any):
        """
        脚本打印日志回调
        :param script: frida的脚本
        :param level: 日志级别
        :param message: 日志内容
        :param data: 事件数据
        """
        log_fn = _logger.debug
        if level == self.LogLevel.INFO:
            log_fn = _logger.info
        if level == self.LogLevel.WARNING:
            log_fn = _logger.warning
        if level == self.LogLevel.ERROR:
            log_fn = _logger.error

        if not utils.is_empty(message):
            log_fn(message)

    def on_script_event(self, script: FridaScript, event: Any, data: Any):
        """
        脚本发送事件回调
        :param script: frida的脚本
        :param event: 事件消息
        :param data: 事件带回来的数据
        """
        message = f"{script} event: {os.linesep}" \
                  f"{json.dumps(event, indent=2, ensure_ascii=False)}"
        self.on_script_log(script, self.LogLevel.INFO, message, None)

    def on_script_send(self, script: FridaScript, payload: Any, data: Any):
        """
        脚本调用send是收到的回调，例send({trace: "xxx"}, null)
        :param script: frida的脚本
        :param payload: 上述例子的{trace: "xxx"}
        :param data: 上述例子的null
        """
        message = f"{script} send: {os.linesep}" \
                  f"{payload}"
        self.on_script_log(script, self.LogLevel.INFO, message, data)

    def on_script_destroyed(self, script: FridaScript):
        """
        脚本结束回调函数，默认只打印log
        :param script: frida的脚本
        """
        self.on_script_log(script, self.LogLevel.DEBUG, f"{script} destroyed", None)


class FridaFileHandler(metaclass=abc.ABCMeta):

    def on_file_change(self, file: FridaScriptFile):
        """
        脚本文件改变回调，默认重新加载脚本
        :param file: 脚本文件路径
        """
        _logger.debug(f"{file} changed")


class FridaEventCounter:

    def __init__(self):
        self._map = {}
        self._lock = threading.RLock()

    def increase(self, group: "Group"):
        with self._lock:
            keys = group.values
            if keys not in self._map:
                self._map[keys] = 0
            self._map[keys] = self._map[keys] + 1
            return self._map[keys]

    class Group:

        def __init__(self, accept_empty: bool = False):
            self._accept_empty = accept_empty
            self._names = []
            self._values = []

        def add(self, **kwargs):
            for k, v in kwargs.items():
                if self._accept_empty or v is not None:
                    self._names.append(k)
                    self._values.append(v)
            return self

        @property
        def names(self):
            return tuple(self._names)

        @property
        def values(self):
            return tuple(self._values)

        def __repr__(self):
            return f"Group({', '.join(self._names)})"


class FridaManager:

    def __init__(self, reactor: FridaReactor):
        self._reactor = reactor
        self._cancel_handlers: "Dict[str, Callable[[], Any]]" = {}

        self._lock = threading.RLock()
        self._sessions: "Dict[int, FridaSession]" = {}

    @property
    def sessions(self) -> Dict[int, FridaSession]:
        with self._lock:
            sessions = {}
            for pid in list(self._sessions.keys()):
                session = self._sessions.get(pid)
                if not session.is_detached:
                    sessions[pid] = session
                    continue
                self._sessions.pop(pid)
            return sessions

    def get_session(self, pid: int) -> Optional[FridaSession]:
        with self._lock:
            session = self._sessions.get(pid)
            if session is not None:
                if not session.is_detached:
                    return session
                self._sessions.pop(pid)
            return None

    def set_session(self, session: FridaSession):
        with self._lock:
            self._sessions[session.pid] = session

    def add_device_handler(self, device: frida.core.Device, handler: FridaDeviceHandler):
        self._call_cancel_handler(device)

        cb_spawn_added = lambda spawn: threading.Thread(target=handler.on_spawn_added, args=(spawn,)).start()
        cb_spawn_removed = lambda spawn: self._reactor.schedule(lambda: handler.on_spawn_removed(spawn))
        cb_child_added = lambda child: self._reactor.schedule(lambda: handler.on_child_added(child))
        cb_child_removed = lambda child: self._reactor.schedule(lambda: handler.on_child_removed(child))
        cb_output = lambda pid, fd, data: self._reactor.schedule(lambda: handler.on_output(pid, fd, data))
        cb_lost = lambda: self._reactor.schedule(lambda: handler.on_device_lost())

        device.on("spawn-added", cb_spawn_added)
        device.on("spawn-removed", cb_spawn_removed)
        device.on("child-added", cb_child_added)
        device.on("child-removed", cb_child_removed)
        device.on("output", cb_output)
        # device.on('process-crashed', cb_process_crashed)
        # device.on('uninjected', cb_uninjected)
        device.on("lost", cb_lost)

        def cancel():
            utils.ignore_error(device.off, args=("spawn-added", cb_spawn_added))
            utils.ignore_error(device.off, args=("spawn-removed", cb_spawn_removed))
            utils.ignore_error(device.off, args=("child-added", cb_child_added))
            utils.ignore_error(device.off, args=("child-removed", cb_child_removed))
            utils.ignore_error(device.off, args=("output", cb_output))
            # utils.ignore_error(device.off, args=("process-crashed", cb_process_crashed))
            # utils.ignore_error(device.off, args=("uninjected", cb_uninjected))
            utils.ignore_error(device.off, args=("lost", cb_lost))

        self._register_cancel_handler(device, cancel)

    def remove_device_handler(self, device: frida.core.Device):
        self._call_cancel_handler(device)

    def add_session_handler(self, session: FridaSession, handler: FridaSessionHandler):
        self._call_cancel_handler(session)

        def on_detached(reason, crash):
            self._reactor.schedule(lambda: self._call_cancel_handler(session))
            with self._lock:
                self._sessions.pop(session.pid, None)
            self._reactor.schedule(lambda: handler.on_session_detached(session, reason, crash))

        session.on("detached", on_detached)

        def cancel():
            utils.ignore_error(session.off, args=("detached", on_detached))

        self._register_cancel_handler(session, cancel)

    def remove_session_handler(self, session: FridaSession):
        self._call_cancel_handler(session)

    def add_script_handler(self, script: FridaScript, handler: FridaScriptHandler):
        self._call_cancel_handler(script)

        def on_message(msg, data):
            return handler.on_script_message(script, msg, data)

        def on_destroyed():
            self._reactor.schedule(lambda: self._call_cancel_handler(script))
            return handler.on_script_destroyed(script)

        script.on("message", on_message)
        script.on("destroyed", on_destroyed)

        def cancel():
            utils.ignore_error(script.off, args=("message", on_message))
            utils.ignore_error(script.off, args=("destroyed", on_destroyed))

        self._register_cancel_handler(script, cancel)

    def remove_script_handler(self, script: FridaScript):
        self._call_cancel_handler(script)

    def add_file_handler(self, files: [FridaScriptFile], handler: FridaFileHandler):
        self._call_cancel_handler(files)

        last_change_id = 0
        monitors: Dict[str, frida.FileMonitor] = {}

        def make_monitor(file):
            _logger.debug(f"Monitor file: {file.path}")
            monitor = frida.FileMonitor(str(file.path))
            monitor.on("change", lambda changed_file, other_file, event_type: on_change_handler(event_type, file))
            monitor.enable()
            return monitor

        def on_change_handler(event_type, changed_file):
            nonlocal last_change_id
            if event_type == "changes-done-hint":
                _logger.debug(f"Monitor event: {event_type}, file: {changed_file}")
                last_change_id += 1
                change_id = last_change_id
                changed_file.clear()
                self._reactor.schedule(lambda: on_change_schedule(change_id, changed_file), delay=0.5)

        def on_change_schedule(change_id, changed_file):
            nonlocal last_change_id
            if change_id == last_change_id:
                handler.on_file_change(changed_file)

        for file in files:
            if file.path not in monitors:
                monitors[file.path] = make_monitor(file)

        def cancel():
            for monitor in monitors.values():
                monitor.disable()

        self._register_cancel_handler(files, cancel)

    def remove_file_handler(self, files: [FridaScriptFile]):
        self._call_cancel_handler(files)

    def _register_cancel_handler(self, key: Any, handler: Callable[[], Any]):
        self._cancel_handlers[self._make_key(key)] = handler

    def _call_cancel_handler(self, key: Any):
        handler = self._cancel_handlers.pop(self._make_key(key), None)
        if handler:
            handler()

    @classmethod
    def _make_key(cls, key: Any):
        if isinstance(key, (list, tuple, set)):
            key = ",".join([str(hash(i)) for i in key])
        return key


class FridaApplication(Stoppable, FridaDeviceHandler, FridaSessionHandler, FridaScriptHandler, FridaFileHandler):
    """
    ----------------------------------------------------------------------

    eg.
        #!/usr/bin/env python3
        # -*- coding: utf-8 -*-

        from linktools.frida import FridaApplication, FridaEvalCode
        from linktools.frida.android import AndroidFridaServer


        jscode = \"\"\"
        Java.perform(function () {
            JavaHelper.hookMethods(
                "java.util.HashMap", "put", JavaHelper.getEventImpl({stack: false, args: true})
            );
        });
        \"\"\"

        if __name__ == "__main__":

            with AndroidFridaServer() as server:

                app = FridaApplication(
                    server,
                    target_identifiers="com.topjohnwu.magisk",
                    user_scripts=(FridaEvalCode(jscode),),
                    enable_spawn_gating=True
                )

                app.inject_all()
                app.run()

    ----------------------------------------------------------------------
    """

    def __init__(
            self,
            device: Union[frida.core.Device, "FridaServer"],
            target_identifiers: Union[str, Collection[str]] = None,
            user_parameters: Dict[str, any] = None,
            user_scripts: Union[FridaUserScript, Collection[FridaUserScript]] = None,
            enable_spawn_gating: bool = False,
            enable_child_gating: bool = False,
            eternalize: str = False,
    ):
        self._device = device

        # 初始化运行环境
        self._last_error = None
        self._stop_request = threading.Event()
        self._reactor = FridaReactor(on_stop=self._on_stop, on_error=self._on_error)
        self._manager = FridaManager(self._reactor)

        # 初始化内置脚本
        script_path = environ.get_asset_path("frida.min.js")
        if not __release__ or environ.debug or not os.path.exists(script_path):
            script_path = environ.get_asset_path("frida.js")
        self._internal_script = FridaScriptFile(script_path)

        # 初始化需要注入进程的匹配规则
        if isinstance(target_identifiers, str):
            self._target_identifiers = [re.compile(target_identifiers)]
        elif isinstance(target_identifiers, Collection):
            self._target_identifiers = [re.compile(i) for i in target_identifiers]
        else:
            self._target_identifiers: [re.Pattern] = []

        # 初始化用户传递的参数
        self._user_parameters = user_parameters or {}

        # 初始化所有需要注入的代码片段/脚本文件/远程脚本文件
        if isinstance(user_scripts, FridaUserScript):
            self._user_scripts = [user_scripts]
        elif isinstance(user_scripts, Collection):
            self._user_scripts = user_scripts
        else:
            self._user_scripts: [FridaUserScript] = []

        # 初始化所有需要监听的脚本文件
        self._user_files = []
        for user_script in self._user_scripts:
            if isinstance(user_script, FridaScriptFile):
                self._user_files.append(user_script)
        if environ.debug:
            self._user_files.append(self._internal_script)

        # 保存其余变量
        self._enable_spawn_gating = enable_spawn_gating
        self._enable_child_gating = enable_child_gating
        self._eternalize = eternalize

        self._event_counter = FridaEventCounter()

    @property
    def device(self) -> frida.core.Device:
        return self._device

    def _init(self):
        _logger.debug(f"FridaApplication init")

        for user_script in self._user_scripts:
            user_script.load()

        self._stop_request.clear()

        self._manager.add_file_handler(self._user_files, self)
        self._manager.add_device_handler(self.device, self)

        if self._enable_spawn_gating:
            try:
                self.device.enable_spawn_gating()
            except frida.NotSupportedError:
                _logger.warning(f"Enable child gating is not supported, ignore")
        else:
            try:
                self.device.disable_spawn_gating()
            except frida.NotSupportedError:
                pass

    def _deinit(self):
        _logger.debug(f"FridaApplication deinit")

        self._manager.remove_device_handler(self.device)
        self._manager.remove_file_handler(self._user_files)

    @property
    def is_running(self) -> bool:
        return self._reactor.is_running()

    def start(self):
        assert not self.is_running
        try:
            self._init()
            self._reactor.start()
        except:
            self.stop()
            raise

    @timeoutable
    def run(self, timeout: TimeoutType = None):
        assert not self.is_running
        try:
            self._init()
            self._reactor.start()
            utils.wait_event(self._stop_request, timeout)
        finally:
            self.stop()

    @timeoutable
    def wait(self, timeout: TimeoutType = None) -> bool:
        return utils.wait_event(self._stop_request, timeout)

    def stop(self):
        self._reactor.signal_stop()
        if not self._reactor.wait(5):
            _logger.warning("Worker did not finish normally")
        self._deinit()

    def signal_stop(self):
        self._stop_request.set()
        self._reactor.signal_stop()

    def __enter__(self):
        assert not self.is_running
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def schedule(self, fn: Callable[[], any], delay: float = None):
        self._reactor.schedule(fn, delay)

    def load_script(self, process_id: int, process_name: str = None, resume: bool = True):
        """
        加载脚本，注入到指定进程
        :param process_id: 进程id
        :param process_name: 进程名称
        :param resume: 注入后是否需要resume进程
        """
        self._reactor.schedule(lambda: self._load_script(process_id, process_name, resume))

    def spawn(self, program: str, resume: bool = True, **kwargs) -> int:
        """
        启动进程，若进程匹配target_identifiers则注入脚本
        """
        pid = self.device.spawn(program, **kwargs)
        for identifier in self._target_identifiers:
            if identifier.search(program):
                self.load_script(pid, program, resume=resume)
                break
        else:
            if resume:
                self._reactor.schedule(lambda: utils.ignore_error(self.device.resume, args=(pid,)))
        return pid

    def inject_all(self, resume: bool = True) -> [int]:
        """
        根据target_identifiers注入所有匹配的进程
        :return: 注入的进程pid
        """

        # 匹配所有app的进程id
        app_pids = set()
        app_list = self.device.enumerate_applications()
        for identifier in self._target_identifiers:
            for app in app_list:
                if app.pid > 0:
                    if identifier.search(app.identifier):
                        app_pids.add(app.pid)

        # 匹配所有进程
        processes = set()
        process_list = self.device.enumerate_processes()
        for identifier in self._target_identifiers:
            for process in process_list:
                if process.pid > 0:
                    if process.pid in app_pids or identifier.search(process.name):
                        processes.add(process)

        pids = list()
        if len(processes) > 0:
            # 进程存在，直接注入
            for process in processes:
                self.load_script(process.pid, process.name, resume=resume)
                pids.append(process.pid)

        return pids

    @property
    def sessions(self) -> Dict[int, FridaSession]:
        """
        所有已注入的session
        """
        return self._manager.sessions

    def attach_session(self, pid: int):
        """
        附加指定进程
        :param pid: 进程id
        """
        self._reactor.schedule(lambda: self._attach_session(pid))

    def detach_session(self, pid: int):
        """
        分离指定进程
        :param pid: 进程id
        """
        session = self._manager.get_session(pid)
        if session is not None:
            self._reactor.schedule(lambda: self._detach_session(session))

    def _load_script_files(self):

        script_files = []

        # 保持脚本log输出级别同步
        if _logger.isEnabledFor(logging.DEBUG):
            script_files.append(FridaEvalCode("Log.setLevel(Log.DEBUG);"))
        elif _logger.isEnabledFor(logging.INFO):
            script_files.append(FridaEvalCode("Log.setLevel(Log.INFO);"))
        elif _logger.isEnabledFor(logging.WARNING):
            script_files.append(FridaEvalCode("Log.setLevel(Log.WARNING);"))
        elif _logger.isEnabledFor(logging.ERROR):
            script_files.append(FridaEvalCode("Log.setLevel(Log.ERROR);"))

        for user_script in self._user_scripts:
            script_files.append(user_script)

        return [o.as_dict() for o in script_files]

    def _load_script(self, process_id: int, process_name: str, resume: bool):
        _logger.debug(f"Attempt to load script: pid={process_id}, resume={resume}")

        session = self._attach_session(process_id, process_name)
        self._unload_script(session)

        kwargs = {}
        if utils.parse_version(frida.__version__) < (14,):
            kwargs["runtime"] = "v8"
        script = FridaScript(
            session,
            session.create_script(self._internal_script.source, **kwargs)
        )
        self._manager.add_script_handler(script, self)

        try:
            script.load()
            script.exports_sync.load_scripts(
                self._load_script_files(),
                self._user_parameters
            )
        finally:
            if resume:
                utils.ignore_error(self.device.resume, args=(process_id,))

        self._reactor.schedule(lambda: self.on_script_loaded(script))

    def _attach_session(self, process_id: int, process_name: str = None):
        session = self._manager.get_session(process_id)
        if session:
            return session

        _logger.debug(f"Attempt to attach process: pid={process_id}")

        if not process_name:
            for process in self.device.enumerate_processes():
                if process.pid == process_id:
                    process_name = process.name
                    break
            else:
                raise frida.ProcessNotFoundError(f"unable to find process with pid '{process_id}'")

        session = self.device.attach(process_id)
        session = FridaSession(session, process_name)
        self._manager.set_session(session)

        if self._enable_child_gating:
            try:
                session.enable_child_gating()
            except frida.NotSupportedError:
                _logger.warning(f"Enable child gating is not supported, ignore")
        else:
            try:
                session.disable_child_gating()
            except frida.NotSupportedError:
                pass

        self._manager.add_session_handler(session, self)
        self._reactor.schedule(lambda: self.on_session_attached(session))

        return session

    def _detach_session(self, session: FridaSession):
        if session is not None:
            _logger.debug(f"{session} detach")
            utils.ignore_error(session.detach)

    def _unload_script(self, session: FridaSession):
        if not session:
            return
        while True:
            script = session.pop()
            if not script:
                break
            _logger.debug(f"{script} unload")
            utils.ignore_error(script.unload)

    def _eternalize_script(self, session: FridaSession):
        if not session:
            return
        while True:
            script = session.pop()
            if not script:
                break
            _logger.debug(f"{script} eternalize")
            utils.ignore_error(script.eternalize)

    def _on_stop(self):
        process_script = self._unload_script
        if self._eternalize:
            process_script = self._eternalize_script

        for session in self.sessions.values():
            process_script(session)
            self._detach_session(session)

        self.on_stop()

    def on_stop(self):
        _logger.debug("Application stopped")

    def _on_error(self, exc, traceback):
        self._last_error = exc
        self.on_error(exc, traceback)

    def on_error(self, exc, traceback):
        if isinstance(exc, (KeyboardInterrupt, frida.TransportError, frida.ServerNotRunningError)):
            _logger.error(f"{traceback if environ.debug else exc}")
            self.stop()
        elif isinstance(exc, (frida.core.RPCException,)):
            _logger.error(f"{exc}")
        else:
            _logger.error(f"{traceback if environ.debug else exc}")

    def raise_on_error(self):
        if self._last_error is not None:
            raise self._last_error

    def on_device_lost(self):
        _logger.info("Device lost")
        self.stop()

    def on_file_change(self, file: FridaScriptFile):
        """
        脚本文件改变回调，默认重新加载脚本
        :param file: 脚本文件路径
        """
        _logger.debug(f"{file} changed")
        for session in self.sessions.values():
            self.load_script(session.pid)

    def _resume(self, process_id: int, process_name: str):
        try:
            # 可能ios设备可能会出错: https://github.com/frida/frida-core/issues/462
            self.device.resume(process_id)
        except Exception as e:
            _logger.warning(f"Resume process `{process_name}` with pid {process_id} failed: {e}")

    def on_spawn_added(self, spawn: "_frida.Spawn"):
        """
        spaw进程添加回调，默认resume所有spawn进程
        :param spawn: spawn进程信息
        """
        _logger.debug(f"{spawn} added")

        if spawn and spawn.identifier:
            for identifier in self._target_identifiers:
                if identifier.search(spawn.identifier):
                    self.load_script(spawn.pid, spawn.identifier, resume=True)
                    return
        self._reactor.schedule(lambda: self._resume(spawn.pid, spawn.identifier))

    def on_child_added(self, child: "_frida.Child"):
        """
        子进程添加回调，默认resume所有子进程
        :param child: 子进程信息
        """
        _logger.debug(f"{child} added")

        if child and child.identifier:
            for identifier in self._target_identifiers:
                if identifier.search(child.identifier):
                    self.load_script(child.pid, child.identifier, resume=True)
                    return
        self._reactor.schedule(lambda: self._resume(child.pid, child.identifier))

    def on_script_loaded(self, script: FridaScript):
        """
        脚本加载回调，默认只打印log
        :param script: frida的脚本
        """
        _logger.debug(f"{script} loaded")

    def on_script_event(self, script: FridaScript, event: Any, data: Any):
        """
        脚本发送事件回调
        :param script: frida的脚本
        :param event: 事件消息
        :param data: 事件数据
        """
        pid_count = self._event_counter.increase(
            FridaEventCounter.Group(accept_empty=False).add(
                pid=script.session.pid,
            )
        )
        pid_method_count = self._event_counter.increase(
            FridaEventCounter.Group(accept_empty=False).add(
                pid=script.session.pid,
                method=utils.get_item(event, "method_name"),
            )
        )
        # 日志展示当前进程当前方法一共出现的次数和具体详情
        _logger.info(
            f"{script} event pid@index={pid_count} pid+method@index={pid_method_count} {os.linesep}"
            f"{json.dumps(event, indent=2, ensure_ascii=False)}",
        )

    def on_script_send(self, script: FridaScript, payload: Any, data: Any):
        """
        脚本调用send是收到的回调，例send({trace: "xxx"}, null)
        :param script: frida的脚本
        :param payload: 上述例子的{trace: "xxx"}
        :param data: 上述例子的null
        """
        _logger.info(f"{script} send, payload={payload}")
