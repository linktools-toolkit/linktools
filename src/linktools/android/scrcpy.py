#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author  : Hu Ji
@file    : scrcpy.py
@time    : 2024/11/22 12:03 
@site    : https://github.com/ice-black-tea
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
import json
import random
import select
import socket
import struct
import subprocess
import threading
import time
from collections.abc import Callable
from typing import List, Dict, Any, Optional

from .adb import AdbDevice, AdbError
from .. import environ, utils
from ..decorator import cached_classproperty
from ..types import Stoppable

logger = environ.get_logger("android.scrcpy")


class ScrcpyError(AdbError):
    pass


class ScrcpyStopError(ScrcpyError):
    pass


class ScrcpyControlMessage:
    TYPE_INJECT_KEYCODE = 0
    TYPE_INJECT_TEXT = 1
    TYPE_INJECT_TOUCH_EVENT = 2
    TYPE_INJECT_SCROLL_EVENT = 3
    TYPE_BACK_OR_SCREEN_ON = 4
    TYPE_EXPAND_NOTIFICATION_PANEL = 5
    TYPE_EXPAND_SETTINGS_PANEL = 6
    TYPE_COLLAPSE_PANELS = 7
    TYPE_GET_CLIPBOARD = 8
    TYPE_SET_CLIPBOARD = 9
    TYPE_SET_DISPLAY_POWER = 10
    TYPE_ROTATE_DEVICE = 11
    TYPE_UHID_CREATE = 12
    TYPE_UHID_INPUT = 13
    TYPE_UHID_DESTROY = 14
    TYPE_OPEN_HARD_KEYBOARD_SETTINGS = 15
    TYPE_START_APP = 16
    TYPE_RESET_VIDEO = 17

    COPY_KEY_NONE = 0
    COPY_KEY_COPY = 1
    COPY_KEY_CUT = 2


class ScrcpyKeyEvent:
    ACTION_DOWN = 0
    ACTION_UP = 1
    ACTION_MOVE = 2

    KEYCODE_UNKNOWN = 0
    KEYCODE_SOFT_LEFT = 1
    KEYCODE_SOFT_RIGHT = 2
    KEYCODE_HOME = 3
    KEYCODE_BACK = 4
    KEYCODE_CALL = 5
    KEYCODE_ENDCALL = 6
    KEYCODE_0 = 7
    KEYCODE_1 = 8
    KEYCODE_2 = 9
    KEYCODE_3 = 10
    KEYCODE_4 = 11
    KEYCODE_5 = 12
    KEYCODE_6 = 13
    KEYCODE_7 = 14
    KEYCODE_8 = 15
    KEYCODE_9 = 16
    KEYCODE_STAR = 17
    KEYCODE_POUND = 18
    KEYCODE_DPAD_UP = 19
    KEYCODE_DPAD_DOWN = 20
    KEYCODE_DPAD_LEFT = 21
    KEYCODE_DPAD_RIGHT = 22
    KEYCODE_DPAD_CENTER = 23
    KEYCODE_VOLUME_UP = 24
    KEYCODE_VOLUME_DOWN = 25
    KEYCODE_POWER = 26
    KEYCODE_CAMERA = 27
    KEYCODE_CLEAR = 28
    KEYCODE_A = 29
    KEYCODE_B = 30
    KEYCODE_C = 31
    KEYCODE_D = 32
    KEYCODE_E = 33
    KEYCODE_F = 34
    KEYCODE_G = 35
    KEYCODE_H = 36
    KEYCODE_I = 37
    KEYCODE_J = 38
    KEYCODE_K = 39
    KEYCODE_L = 40
    KEYCODE_M = 41
    KEYCODE_N = 42
    KEYCODE_O = 43
    KEYCODE_P = 44
    KEYCODE_Q = 45
    KEYCODE_R = 46
    KEYCODE_S = 47
    KEYCODE_T = 48
    KEYCODE_U = 49
    KEYCODE_V = 50
    KEYCODE_W = 51
    KEYCODE_X = 52
    KEYCODE_Y = 53
    KEYCODE_Z = 54
    KEYCODE_COMMA = 55
    KEYCODE_PERIOD = 56
    KEYCODE_ALT_LEFT = 57
    KEYCODE_ALT_RIGHT = 58
    KEYCODE_SHIFT_LEFT = 59
    KEYCODE_SHIFT_RIGHT = 60
    KEYCODE_TAB = 61
    KEYCODE_SPACE = 62
    KEYCODE_SYM = 63
    KEYCODE_EXPLORER = 64
    KEYCODE_ENVELOPE = 65
    KEYCODE_ENTER = 66
    KEYCODE_DEL = 67
    KEYCODE_GRAVE = 68
    KEYCODE_MINUS = 69
    KEYCODE_EQUALS = 70
    KEYCODE_LEFT_BRACKET = 71
    KEYCODE_RIGHT_BRACKET = 72
    KEYCODE_BACKSLASH = 73
    KEYCODE_SEMICOLON = 74
    KEYCODE_APOSTROPHE = 75
    KEYCODE_SLASH = 76
    KEYCODE_AT = 77
    KEYCODE_NUM = 78
    KEYCODE_HEADSETHOOK = 79
    KEYCODE_FOCUS = 80
    KEYCODE_PLUS = 81
    KEYCODE_MENU = 82
    KEYCODE_NOTIFICATION = 83
    KEYCODE_SEARCH = 84
    KEYCODE_MEDIA_PLAY_PAUSE = 85
    KEYCODE_MEDIA_STOP = 86
    KEYCODE_MEDIA_NEXT = 87
    KEYCODE_MEDIA_PREVIOUS = 88
    KEYCODE_MEDIA_REWIND = 89
    KEYCODE_MEDIA_FAST_FORWARD = 90
    KEYCODE_MUTE = 91
    KEYCODE_PAGE_UP = 92
    KEYCODE_PAGE_DOWN = 93
    KEYCODE_PICTSYMBOLS = 94
    KEYCODE_SWITCH_CHARSET = 95
    KEYCODE_BUTTON_A = 96
    KEYCODE_BUTTON_B = 97
    KEYCODE_BUTTON_C = 98
    KEYCODE_BUTTON_X = 99
    KEYCODE_BUTTON_Y = 100
    KEYCODE_BUTTON_Z = 101
    KEYCODE_BUTTON_L1 = 102
    KEYCODE_BUTTON_R1 = 103
    KEYCODE_BUTTON_L2 = 104
    KEYCODE_BUTTON_R2 = 105
    KEYCODE_BUTTON_THUMBL = 106
    KEYCODE_BUTTON_THUMBR = 107
    KEYCODE_BUTTON_START = 108
    KEYCODE_BUTTON_SELECT = 109
    KEYCODE_BUTTON_MODE = 110
    KEYCODE_ESCAPE = 111
    KEYCODE_FORWARD_DEL = 112
    KEYCODE_CTRL_LEFT = 113
    KEYCODE_CTRL_RIGHT = 114
    KEYCODE_CAPS_LOCK = 115
    KEYCODE_SCROLL_LOCK = 116
    KEYCODE_META_LEFT = 117
    KEYCODE_META_RIGHT = 118
    KEYCODE_FUNCTION = 119
    KEYCODE_SYSRQ = 120
    KEYCODE_BREAK = 121
    KEYCODE_MOVE_HOME = 122
    KEYCODE_MOVE_END = 123
    KEYCODE_INSERT = 124
    KEYCODE_FORWARD = 125
    KEYCODE_MEDIA_PLAY = 126
    KEYCODE_MEDIA_PAUSE = 127
    KEYCODE_MEDIA_CLOSE = 128
    KEYCODE_MEDIA_EJECT = 129
    KEYCODE_MEDIA_RECORD = 130
    KEYCODE_F1 = 131
    KEYCODE_F2 = 132
    KEYCODE_F3 = 133
    KEYCODE_F4 = 134
    KEYCODE_F5 = 135
    KEYCODE_F6 = 136
    KEYCODE_F7 = 137
    KEYCODE_F8 = 138
    KEYCODE_F9 = 139
    KEYCODE_F10 = 140
    KEYCODE_F11 = 141
    KEYCODE_F12 = 142
    KEYCODE_NUM_LOCK = 143
    KEYCODE_NUMPAD_0 = 144
    KEYCODE_NUMPAD_1 = 145
    KEYCODE_NUMPAD_2 = 146
    KEYCODE_NUMPAD_3 = 147
    KEYCODE_NUMPAD_4 = 148
    KEYCODE_NUMPAD_5 = 149
    KEYCODE_NUMPAD_6 = 150
    KEYCODE_NUMPAD_7 = 151
    KEYCODE_NUMPAD_8 = 152
    KEYCODE_NUMPAD_9 = 153
    KEYCODE_NUMPAD_DIVIDE = 154
    KEYCODE_NUMPAD_MULTIPLY = 155
    KEYCODE_NUMPAD_SUBTRACT = 156
    KEYCODE_NUMPAD_ADD = 157
    KEYCODE_NUMPAD_DOT = 158
    KEYCODE_NUMPAD_COMMA = 159
    KEYCODE_NUMPAD_ENTER = 160
    KEYCODE_NUMPAD_EQUALS = 161
    KEYCODE_NUMPAD_LEFT_PAREN = 162
    KEYCODE_NUMPAD_RIGHT_PAREN = 163
    KEYCODE_VOLUME_MUTE = 164
    KEYCODE_INFO = 165
    KEYCODE_CHANNEL_UP = 166
    KEYCODE_CHANNEL_DOWN = 167
    KEYCODE_ZOOM_IN = 168
    KEYCODE_ZOOM_OUT = 169
    KEYCODE_TV = 170
    KEYCODE_WINDOW = 171
    KEYCODE_GUIDE = 172
    KEYCODE_DVR = 173
    KEYCODE_BOOKMARK = 174
    KEYCODE_CAPTIONS = 175
    KEYCODE_SETTINGS = 176
    KEYCODE_TV_POWER = 177
    KEYCODE_TV_INPUT = 178
    KEYCODE_STB_POWER = 179
    KEYCODE_STB_INPUT = 180
    KEYCODE_AVR_POWER = 181
    KEYCODE_AVR_INPUT = 182
    KEYCODE_PROG_RED = 183
    KEYCODE_PROG_GREEN = 184
    KEYCODE_PROG_YELLOW = 185
    KEYCODE_PROG_BLUE = 186
    KEYCODE_APP_SWITCH = 187
    KEYCODE_BUTTON_1 = 188
    KEYCODE_BUTTON_2 = 189
    KEYCODE_BUTTON_3 = 190
    KEYCODE_BUTTON_4 = 191
    KEYCODE_BUTTON_5 = 192
    KEYCODE_BUTTON_6 = 193
    KEYCODE_BUTTON_7 = 194
    KEYCODE_BUTTON_8 = 195
    KEYCODE_BUTTON_9 = 196
    KEYCODE_BUTTON_10 = 197
    KEYCODE_BUTTON_11 = 198
    KEYCODE_BUTTON_12 = 199
    KEYCODE_BUTTON_13 = 200
    KEYCODE_BUTTON_14 = 201
    KEYCODE_BUTTON_15 = 202
    KEYCODE_BUTTON_16 = 203
    KEYCODE_LANGUAGE_SWITCH = 204
    KEYCODE_MANNER_MODE = 205
    KEYCODE_3D_MODE = 206
    KEYCODE_CONTACTS = 207
    KEYCODE_CALENDAR = 208
    KEYCODE_MUSIC = 209
    KEYCODE_CALCULATOR = 210
    KEYCODE_ZENKAKU_HANKAKU = 211
    KEYCODE_EISU = 212
    KEYCODE_MUHENKAN = 213
    KEYCODE_HENKAN = 214
    KEYCODE_KATAKANA_HIRAGANA = 215
    KEYCODE_YEN = 216
    KEYCODE_RO = 217
    KEYCODE_KANA = 218
    KEYCODE_ASSIST = 219
    KEYCODE_BRIGHTNESS_DOWN = 220
    KEYCODE_BRIGHTNESS_UP = 221
    KEYCODE_MEDIA_AUDIO_TRACK = 222
    KEYCODE_SLEEP = 223
    KEYCODE_WAKEUP = 224
    KEYCODE_PAIRING = 225
    KEYCODE_MEDIA_TOP_MENU = 226
    KEYCODE_11 = 227
    KEYCODE_12 = 228
    KEYCODE_LAST_CHANNEL = 229
    KEYCODE_TV_DATA_SERVICE = 230
    KEYCODE_VOICE_ASSIST = 231
    KEYCODE_TV_RADIO_SERVICE = 232
    KEYCODE_TV_TELETEXT = 233
    KEYCODE_TV_NUMBER_ENTRY = 234
    KEYCODE_TV_TERRESTRIAL_ANALOG = 235
    KEYCODE_TV_TERRESTRIAL_DIGITAL = 236
    KEYCODE_TV_SATELLITE = 237
    KEYCODE_TV_SATELLITE_BS = 238
    KEYCODE_TV_SATELLITE_CS = 239
    KEYCODE_TV_SATELLITE_SERVICE = 240
    KEYCODE_TV_NETWORK = 241,
    KEYCODE_TV_ANTENNA_CABLE = 242
    KEYCODE_TV_INPUT_HDMI_1 = 243
    KEYCODE_TV_INPUT_HDMI_2 = 244
    KEYCODE_TV_INPUT_HDMI_3 = 245
    KEYCODE_TV_INPUT_HDMI_4 = 246
    KEYCODE_TV_INPUT_COMPOSITE_1 = 247
    KEYCODE_TV_INPUT_COMPOSITE_2 = 248
    KEYCODE_TV_INPUT_COMPONENT_1 = 249
    KEYCODE_TV_INPUT_COMPONENT_2 = 250
    KEYCODE_TV_INPUT_VGA_1 = 251
    KEYCODE_TV_AUDIO_DESCRIPTION = 252
    KEYCODE_TV_AUDIO_DESCRIPTION_MIX_UP = 253
    KEYCODE_TV_AUDIO_DESCRIPTION_MIX_DOWN = 254
    KEYCODE_TV_ZOOM_MODE = 255
    KEYCODE_TV_CONTENTS_MENU = 256
    KEYCODE_TV_MEDIA_CONTEXT_MENU = 257
    KEYCODE_TV_TIMER_PROGRAMMING = 258
    KEYCODE_HELP = 259
    KEYCODE_NAVIGATE_PREVIOUS = 260
    KEYCODE_NAVIGATE_NEXT = 261
    KEYCODE_NAVIGATE_IN = 262
    KEYCODE_NAVIGATE_OUT = 263
    KEYCODE_STEM_PRIMARY = 264
    KEYCODE_STEM_1 = 265
    KEYCODE_STEM_2 = 266
    KEYCODE_STEM_3 = 267
    KEYCODE_DPAD_UP_LEFT = 268
    KEYCODE_DPAD_DOWN_LEFT = 269
    KEYCODE_DPAD_UP_RIGHT = 270
    KEYCODE_DPAD_DOWN_RIGHT = 271
    KEYCODE_MEDIA_SKIP_FORWARD = 272
    KEYCODE_MEDIA_SKIP_BACKWARD = 273
    KEYCODE_MEDIA_STEP_FORWARD = 274
    KEYCODE_MEDIA_STEP_BACKWARD = 275
    KEYCODE_SOFT_SLEEP = 276
    KEYCODE_CUT = 277
    KEYCODE_COPY = 278
    KEYCODE_PASTE = 279
    KEYCODE_SYSTEM_NAVIGATION_UP = 280
    KEYCODE_SYSTEM_NAVIGATION_DOWN = 281
    KEYCODE_SYSTEM_NAVIGATION_LEFT = 282
    KEYCODE_SYSTEM_NAVIGATION_RIGHT = 283
    KEYCODE_ALL_APPS = 284


class ScrcpyMotionEvent:
    ACTION_DOWN = 0
    ACTION_UP = 1
    ACTION_MOVE = 2
    ACTION_CANCEL = 3
    ACTION_OUTSIDE = 4
    ACTION_POINTER_DOWN = 5
    ACTION_POINTER_UP = 6
    ACTION_HOVER_MOVE = 7
    ACTION_SCROLL = 8
    ACTION_HOVER_ENTER = 9
    ACTION_HOVER_EXIT = 10
    ACTION_BUTTON_PRESS = 11
    ACTION_BUTTON_RELEASE = 12


class ScrcpySurfaceControl:
    POWER_MODE_OFF = 0
    POWER_MODE_NORMAL = 2


class ScrcpyServer(Stoppable):

    def __init__(self, device: AdbDevice = None, version: str = None):
        self._device = device or AdbDevice()
        self._version = version
        self._process: Optional[utils.Process] = None

    @cached_classproperty
    def _server_info(self) -> "List[Dict[str, str]]":
        server_path = environ.get_asset_path("android-tools.json")
        server_data = json.loads(utils.read_file(server_path, text=True))
        return server_data["SCRCPY_SERVER"]

    def start(self, *args: Any):

        def start():
            server_info = dict(self._server_info)
            server_version = server_info["version"] = self._version or server_info["version"]
            server_name = server_info["name"].format(**server_info)
            server_url = server_info["url"].format(**server_info)
            server_path = environ.get_data_path("android", "scrcpy", server_name, create_parent=True)
            if not server_path.exists():
                url_file = environ.get_url_file(server_url)
                url_file.save(server_path.parent, server_path.name)

            remote_path = self._device.push_file(
                server_path,
                self._device.get_data_path("scrcpy"),
                server_name,
                skip_exist=True,
            )
            self._process = self._device.popen(
                "shell",
                utils.list2cmdline([
                    f"CLASSPATH={remote_path}", "app_process", "/", "com.genymobile.scrcpy.Server", server_version,
                    *[str(arg) for arg in args]
                ]),
                text=True,
                bufsize=1,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            for out, err in self._process.fetch(timeout=1):
                if err:
                    logger.warning(err)

            if self._process.poll() is not None:
                raise ScrcpyError(f"{self._device} scrcpy server start failed")

            return self

        return self._stop_on_error(start)

    def stop(self):
        if self._process:
            logger.debug(f"{self._device} stop scrcpy server")
            utils.ignore_error(self._process.recursive_kill)
            try:
                self._process.wait(1)
            except subprocess.TimeoutExpired:
                utils.ignore_error(self._process.kill)
                logger.warning(f"{self._device} scrcpy server stop timeout")
            self._process = None


class ScrcpySession(Stoppable):

    def __init__(self, device: AdbDevice = None, version: str = None):
        self._device = device or AdbDevice()
        self._version = version
        self._thread = threading.Thread()
        self._lock = threading.RLock()
        self._listeners: Dict[str, "List[Callable[..., Any]]"] = dict()
        self._video_socket = None
        self._audio_socket = None
        self._control_socket = None
        self._control_lock = threading.RLock()

    @property
    def is_running(self) -> bool:
        thread = self._thread
        return thread and thread.is_alive()

    def start(
            self, video: bool = True, audio: bool = True, control: bool = True,
            video_codec: str = "h264", video_source: str = "display", video_bit_rate: int = 8000000,
            max_size: int = 1366, max_fps: int = 30,
    ) -> "ScrcpySession":

        if not video and not audio and not control:
            raise RuntimeError("video, audio, and control are not allowed to be false at the same time")

        def worker_thread():
            server = forward = None
            scid = str(random.randint(0, 5)) + "".join([hex(random.randint(1, 15))[-1] for _ in range(7)])
            try:
                server = ScrcpyServer(device=self._device, version=self._version)
                server.start(
                    "tunnel_forward=true",
                    "cleanup=false",
                    f"scid={scid}",
                    f"video={'true' if video else 'false'}",
                    f"audio={'true' if audio else 'false'}",
                    f"control={'true' if control else 'false'}",
                    f"max_size={max_size}",
                    f"max_fps={max_fps}",
                    f"video_codec={video_codec}",
                    f"video_source={video_source}",
                    f"video_bit_rate={video_bit_rate}",
                    "log_level=warn",
                    "send_dummy_byte=true",
                    "send_device_meta=true",
                    "send_codec_meta=true",
                    "send_frame_meta=true",
                )
                forward = self._device.forward("tcp:0", f"localabstract:scrcpy_{scid}")

                video_socket = audio_socket = control_socket = None
                if video:
                    video_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    video_socket.connect(("localhost", forward.local_port))
                if audio:
                    audio_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    audio_socket.connect(("localhost", forward.local_port))
                if control:
                    control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    control_socket.connect(("localhost", forward.local_port))

                with self._lock:
                    self._video_socket = video_socket
                    self._audio_socket = audio_socket
                    self._control_socket = control_socket

                first_sock = utils.coalesce(self._video_socket, self._audio_socket, self._control_socket)

                # receive dummy byte
                dummy_byte = first_sock.recv(1)
                if dummy_byte != b'\x00':
                    raise ScrcpyError("Did not receive Dummy Byte!")
                logger.debug(f"{self._device} receive scrcpy dummy byte: {dummy_byte}")

                # receive device meta
                device_meta = first_sock.recv(64).decode(errors="ignore").rstrip('\x00')
                if not device_meta:
                    raise ScrcpyError("Did not receive Device Name!")
                logger.debug(f"{self._device} receive scrcpy device meta: {device_meta}")

                if video_socket is not None:
                    # receive video codec metadata
                    codec_id = video_socket.recv(4).decode(errors="ignore")
                    logger.debug(f"{self._device} receive scrcpy video codec id: {codec_id}")
                    width, height = struct.unpack(">II", video_socket.recv(8))
                    logger.debug(f"{self._device} receive scrcpy video width: {width}, height: {height}")

                if audio_socket is not None:
                    # receive audio codec metadata
                    codec_id = audio_socket.recv(4).decode(errors="ignore")
                    logger.debug(f"{self._device} receive scrcpy audio codec id: {codec_id}")

                with self._lock:
                    listeners = self._listeners.get("init", list())
                for listener in listeners:
                    try:
                        listener()
                    except ScrcpyStopError:
                        raise
                    except Exception as e:
                        logger.error(f"{self._device} scrcpy init listener error: {e}")

                try:
                    self._recv_media_packets(video_socket, audio_socket, control_socket)
                except:
                    pass

            except Exception as e:
                logger.error(f"{self._device} scrcpy session error: {e}")

            finally:
                self.stop()
                with self._lock:
                    self._video_socket = None
                    self._audio_socket = None
                    self._control_socket = None

                if forward is not None:
                    forward.stop()
                if server is not None:
                    server.stop()

                with self._lock:
                    if self._thread:
                        self._thread = None

                with self._lock:
                    listeners = self._listeners.get("stop", list())
                for listener in listeners:
                    try:
                        listener()
                    except Exception as e:
                        logger.error(f"{self._device} scrcpy stop listener error: {e}")

        with self._lock:
            if not self.is_running:
                self._thread = threading.Thread(target=worker_thread)
                self._thread.start()

        return self

    def _recv_media_packets(self, video_socket, audio_socket, control_socket):

        def recv_data(sock):
            try:
                return sock.recv(0x100000)
            except OSError as e:
                logger.debug(f"{self._device} receive scrcpy data error: {e}")
                return None

        def split_packets(buffer, data):
            # 处理分包和粘包的问题
            packets = list()
            buffer = buffer + data if buffer is not None else data

            index = 0
            header_len = 12
            while len(buffer) >= index + header_len:
                header = buffer[index: index + header_len]
                pts, packet_size = struct.unpack(">QI", header)
                if len(buffer) < index + header_len + packet_size:
                    break
                packet = bytes(buffer[index + header_len:index + header_len + packet_size])
                packets.append(packet)
                index += header_len + packet_size

            return buffer[index:], packets

        def notify_listeners(event, packets):
            with self._lock:
                listeners = self._listeners.get(event, empty)
            for package in packets:
                for listener in listeners:
                    try:
                        listener(package)
                    except ScrcpyStopError:
                        raise
                    except Exception as e:
                        logger.error(f"{self._device} scrcpy {event} listener error: {e}")

        rlist = []
        xlist = []
        video_buffer, audio_buffer = None, None
        if video_socket:
            rlist.append(video_socket)
            xlist.append(video_socket)
        if audio_socket:
            rlist.append(audio_socket)
            xlist.append(audio_socket)
        if control_socket:
            xlist.append(control_socket)

        empty = list()
        while xlist:
            r, w, x = select.select(rlist, [], xlist)

            if r:
                if video_socket in r:
                    # 解析视频数据，并同步给listener
                    data = recv_data(video_socket)
                    if data:
                        video_buffer, packets = split_packets(video_buffer, data)
                        if packets:
                            notify_listeners("video", packets)
                    else:
                        if video_socket in rlist:
                            rlist.remove(video_socket)
                        if video_socket in xlist:
                            xlist.remove(video_socket)

                if audio_socket in r:
                    # 解析音频数据，并同步给listener
                    data = recv_data(audio_socket)
                    if data:
                        audio_buffer, packets = split_packets(audio_buffer, data)
                        if packets:
                            notify_listeners("audio", packets)
                    else:
                        if audio_socket in rlist:
                            rlist.remove(audio_socket)
                        if audio_socket in xlist:
                            xlist.remove(audio_socket)

            if x:
                # 有异常发生，关闭socket
                for sock in x:
                    if sock in rlist:
                        rlist.remove(sock)
                    if sock in xlist:
                        xlist.remove(sock)
                    utils.ignore_error(sock.close)

    def _send_control_packet(self, packet: bytes):
        sock = self._control_socket
        if not sock:
            raise ScrcpyError("Control socket is not connected")
        with self._control_lock:
            try:
                sock.send(packet)
            except Exception as e:
                raise ScrcpyError(f"Send scrcpy control packet error: {e}") from None

    def add_init_listener(self, listener: Callable[[], Any]):
        with self._lock:
            listeners = self._listeners.setdefault("init", list())
            listeners.append(listener)

    def remove_init_listener(self, listener: Callable[[], Any]):
        with self._lock:
            listeners = self._listeners.get("init", list())
            if listener in listeners:
                listeners.remove(listener)

    def add_stop_listener(self, listener: Callable[[], Any]):
        with self._lock:
            listeners = self._listeners.setdefault("stop", list())
            listeners.append(listener)

    def remove_stop_listener(self, listener: Callable[[], Any]):
        with self._lock:
            listeners = self._listeners.get("stop", list())
            if listener in listeners:
                listeners.remove(listener)

    def add_video_listener(self, listener: Callable[[bytes], Any]):
        with self._lock:
            listeners = self._listeners.setdefault("video", list())
            listeners.append(listener)

    def remove_video_listener(self, listener: Callable[[bytes], Any]):
        with self._lock:
            listeners = self._listeners.get("video", list())
            if listener in listeners:
                listeners.remove(listener)

    def add_audio_listener(self, listener: Callable[[bytes], Any]):
        with self._lock:
            listeners = self._listeners.setdefault("audio", list())
            listeners.append(listener)

    def remove_audio_listener(self, listener: Callable[[bytes], Any]):
        with self._lock:
            listeners = self._listeners.get("audio", list())
            if listener in listeners:
                listeners.remove(listener)

    #################################################################################
    #
    # control相关协议
    #     链接：https://github.com/Genymobile/scrcpy/blob/master/app/tests/test_control_msg_serialize.c
    #
    # 下面control的代码完全参（拷）考（贝）py-scrcpy-client
    #     链接：https://github.com/leng-yue/py-scrcpy-client/blob/main/scrcpy/control.py
    #
    #################################################################################

    def inject_keycode(
            self,
            keycode: int,
            action: int = ScrcpyKeyEvent.ACTION_DOWN,
            repeat: int = 0
    ):
        """
        Send keycode to device

        Args:
            keycode: ScrcpyKeyEvent.KEYCODE_*
            action: ACTION_DOWN | ACTION_UP
            repeat: repeat count
        """
        self._send_control_packet(
            struct.pack(
                ">BBiii",
                ScrcpyControlMessage.TYPE_INJECT_KEYCODE,
                action,
                keycode,
                repeat,
                0,
            )
        )

    def inject_text(self, text: str):
        """
        Send text to device

        Args:
            text: text to send
        """
        buffer = text.encode("utf-8")
        self._send_control_packet(
            struct.pack(
                ">Bi",
                ScrcpyControlMessage.TYPE_INJECT_TEXT,
                len(buffer),
            ) + buffer
        )

    def inject_touch_event(
            self,
            width: int,
            height: int,
            x: int,
            y: int,
            action: int = ScrcpyMotionEvent.ACTION_DOWN,
            pointer_id: int = 0x88888888,
    ):
        """
        Touch screen

        Args:
            width: screen width
            height: screen height
            x: horizontal position
            y: vertical position
            action: ACTION_DOWN | ACTION_UP | ACTION_MOVE
            pointer_id: Default using virtual id -1, you can specify it to emulate multi finger touch
        """
        x, y = min(max(x, 0), width), min(max(y, 0), height)
        self._send_control_packet(
            struct.pack(
                ">BBqiiHHHii",
                ScrcpyControlMessage.TYPE_INJECT_TOUCH_EVENT,
                action,
                pointer_id,
                int(x),
                int(y),
                int(width),
                int(height),
                0xFFFF,
                1,
                1,
            )
        )

    def inject_scroll_event(
            self,
            width: int,
            height: int,
            x: int,
            y: int,
            h: int,
            v: int
    ):
        """
        Scroll screen

        Args:
            width: screen width
            height: screen height
            x: horizontal position
            y: vertical position
            h: horizontal movement
            v: vertical movement
        """

        x, y = min(max(x, 0), width), min(max(y, 0), height)
        self._send_control_packet(
            struct.pack(
                ">BiiHHii",
                ScrcpyControlMessage.TYPE_INJECT_SCROLL_EVENT,
                int(x),
                int(y),
                int(width),
                int(height),
                int(h),
                int(v),
            )
        )

    def back_or_turn_screen_on(self, action: int = ScrcpyKeyEvent.ACTION_DOWN):
        """
        If the screen is off, it is turned on only on ACTION_DOWN

        Args:
            action: ACTION_DOWN | ACTION_UP
        """
        self._send_control_packet(
            struct.pack(
                ">BB",
                ScrcpyControlMessage.TYPE_BACK_OR_SCREEN_ON,
                action,
            )
        )

    def expand_notification_panel(self):
        """
        Expand notification panel
        """
        self._send_control_packet(
            struct.pack(
                ">B",
                ScrcpyControlMessage.TYPE_EXPAND_NOTIFICATION_PANEL,
            )
        )

    def expand_settings_panel(self):
        """
        Expand settings panel
        """
        self._send_control_packet(
            struct.pack(
                ">B",
                ScrcpyControlMessage.TYPE_EXPAND_SETTINGS_PANEL,
            )
        )

    def collapse_panels(self):
        """
        Collapse all panels
        """
        self._send_control_packet(
            struct.pack(
                ">B",
                ScrcpyControlMessage.TYPE_COLLAPSE_PANELS,
            )
        )

    def set_display_power(self, mode: int = ScrcpySurfaceControl.POWER_MODE_NORMAL) -> bytes:
        """
        Set screen power mode

        Args:
            mode: POWER_MODE_OFF | POWER_MODE_NORMAL
        """
        self._send_control_packet(
            struct.pack(
                ">Bb",
                ScrcpyControlMessage.TYPE_SET_DISPLAY_POWER,
                mode
            )
        )
        return struct.pack(">b", mode)

    def rotate_device(self):
        """
        Rotate device
        """
        self._send_control_packet(
            struct.pack(
                ">B",
                ScrcpyControlMessage.TYPE_ROTATE_DEVICE,
            )
        )

    def swipe(
            self,
            width: int,
            height: int,
            start_x: int,
            start_y: int,
            end_x: int,
            end_y: int,
            move_step_length: int = 5,
            move_steps_delay: float = 0.005,
    ) -> None:
        """
        Swipe on screen

        Args:
            width: screen width
            height: screen height
            start_x: start horizontal position
            start_y: start vertical position
            end_x: start horizontal position
            end_y: end vertical position
            move_step_length: length per step
            move_steps_delay: sleep seconds after each step
        :return:
        """
        self.inject_touch_event(width, height, start_x, start_y, ScrcpyMotionEvent.ACTION_DOWN)

        next_x, next_y = start_x, start_y
        if end_x > width:
            end_x = width
        if end_y > height:
            end_y = height

        decrease_x = True if start_x > end_x else False
        decrease_y = True if start_y > end_y else False
        while True:
            if decrease_x:
                next_x -= move_step_length
                if next_x < end_x:
                    next_x = end_x
            else:
                next_x += move_step_length
                if next_x > end_x:
                    next_x = end_x

            if decrease_y:
                next_y -= move_step_length
                if next_y < end_y:
                    next_y = end_y
            else:
                next_y += move_step_length
                if next_y > end_y:
                    next_y = end_y

            self.inject_touch_event(width, height, next_x, next_y, ScrcpyMotionEvent.ACTION_MOVE)

            if next_x == end_x and next_y == end_y:
                self.inject_touch_event(width, height, next_x, next_y, ScrcpyMotionEvent.ACTION_UP)
                break
            time.sleep(move_steps_delay)

    def stop(self):
        sock = self._video_socket
        if sock:
            utils.ignore_error(sock.close)
        sock = self._audio_socket
        if sock:
            utils.ignore_error(sock.close)
        sock = self._control_socket
        if sock:
            utils.ignore_error(sock.close)


if __name__ == '__main__':
    import logging
    import av
    import cv2

    from .. import rich
    from ..types import CacheQueue

    rich.init_logging(level=logging.DEBUG, show_level=True)

    video_codec = av.codec.CodecContext.create("h264", "r")
    video_frames = CacheQueue[av.VideoFrame](100)


    def on_init(session: ScrcpySession):
        session.inject_keycode(ScrcpyKeyEvent.KEYCODE_HOME, ScrcpyMotionEvent.ACTION_DOWN)
        session.inject_keycode(ScrcpyKeyEvent.KEYCODE_HOME, ScrcpyMotionEvent.ACTION_UP)


    def on_video(raw_packet: bytes):
        packets = video_codec.parse(raw_packet)
        if not packets:
            return
        for packet in packets:
            for frame in video_codec.decode(packet):
                video_frames.put(frame)


    session = ScrcpySession()
    session.add_init_listener(lambda: on_init(session))
    session.add_video_listener(on_video)
    with session.start(video=True, audio=False, control=True, video_codec="h264", max_fps=30):
        cv2.namedWindow("scrcpy", cv2.WINDOW_NORMAL)
        while True:
            frame = video_frames.get()
            if frame:
                cv2.imshow("scrcpy", frame.to_ndarray(format="bgr24"))
            if cv2.waitKey(1) == 27:
                break
