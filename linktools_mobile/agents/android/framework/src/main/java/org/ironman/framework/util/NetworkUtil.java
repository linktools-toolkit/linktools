package org.ironman.framework.util;

import android.text.TextUtils;

import org.ironman.framework.Const;
import org.ironman.framework.bean.net.InetSocket;
import org.ironman.framework.bean.net.Socket;
import org.ironman.framework.bean.net.UnixSocket;

import java.io.File;
import java.io.IOException;
import java.net.Inet4Address;
import java.net.Inet6Address;
import java.net.InetAddress;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by hu on 19-2-13.
 */

public class NetworkUtil {

    private static final String TAG = NetworkUtil.class.getSimpleName();

    private static final String PROC_NET_TCP = "/proc/net/tcp";
    private static final String PROC_NET_TCP6 = "/proc/net/tcp6";
    private static final String PROC_NET_UDP = "/proc/net/udp";
    private static final String PROC_NET_UDP6 = "/proc/net/udp6";
    private static final String PROC_NET_RAW = "/proc/net/raw";
    private static final String PROC_NET_RAW6 = "/proc/net/raw6";
    private static final String PROC_NET_UNIX = "/proc/net/unix";

    public static List<InetSocket> getTcpSockets() throws IOException {
        List<InetSocket> sockets = new ArrayList<>();
        parse("tcp", PROC_NET_TCP, tcpParser, sockets);
        if (new File(PROC_NET_TCP6).exists()) {
            parse("tcp6", PROC_NET_TCP6, tcpParser, sockets);
        }
        return sockets;
    }

    public static List<InetSocket> getUdpSockets() throws IOException {
        List<InetSocket> sockets = new ArrayList<>();
        parse("udp", PROC_NET_UDP, udpParser, sockets);
        if (new File(PROC_NET_UDP6).exists()) {
            parse("udp6", PROC_NET_UDP6, udpParser, sockets);
        }
        return sockets;
    }

    public static List<InetSocket> getRawSockets() throws IOException {
        List<InetSocket> sockets = new ArrayList<>();
        parse("raw", PROC_NET_RAW, rawParser, sockets);
        if (new File(PROC_NET_RAW6).exists()) {
            parse("raw6", PROC_NET_RAW6, rawParser, sockets);
        }
        return sockets;
    }

    public static List<UnixSocket> getUnixSockets() throws IOException {
        List<UnixSocket> sockets = new ArrayList<>();
        parse("unix", PROC_NET_UNIX, unixParser, sockets);
        return sockets;
    }

    private static <T extends Socket> void parse(String proto, String path, Parser<T> parser, List<T> list) throws IOException {
        String result = FileUtil.readString(path);
        String[] lines = result.split(Const.LINE_SEPARATOR);
        for (int i = 1; i < lines.length; i++) {
            try {
                list.add(parser.parse(proto, lines[i].trim()));
            } catch (Exception e) {
                LogUtil.printStackTrace(TAG, e);
            }
        }
    }

    private static byte[] littleEndian2Bytes(String hex, int size) {
        byte[] bytes = new byte[size];
        for (int i = 0; i < bytes.length; i++) {
            int end = 8 + (i >> 2) * 8 - (i & 3) * 2;
            int start = end - 2;
            bytes[i] = (byte) CommonUtil.parseInt(hex.substring(start, end), 16, 0);
        }
        return bytes;
    }

    private static InetAddress parseInetAddress(String address) throws Exception {
        if (address.length() > 8) {
            return Inet6Address.getByAddress(littleEndian2Bytes(address, 16));
        } else {
            return Inet4Address.getByAddress(littleEndian2Bytes(address, 4));
        }
    }

    private interface Parser<T extends Socket> {
        T parse(String proto, String line) throws Exception;
    }

    private static final Parser<InetSocket> tcpParser = (proto, line) -> {

        String[] detail = line.split(" +");
        if (detail.length < 10) {
            throw new Exception();
        }

        InetAddress localAddress = parseInetAddress(detail[1].split(":")[0]);
        int localPort = CommonUtil.parseInt(detail[1].split(":")[1], 16, 0);
        InetAddress remoteAddress = parseInetAddress(detail[2].split(":")[0]);
        int remotePort = CommonUtil.parseInt(detail[2].split(":")[1], 16, 0);
        long transmitQueue = CommonUtil.parseLong(detail[4].split(":")[0], 16, 0);
        long receiveQueue = CommonUtil.parseLong(detail[4].split(":")[1], 16, 0);
        int state = CommonUtil.parseInt(detail[3], 16, 0);
        int uid = CommonUtil.parseInt(detail[7], 0);
        long inode = CommonUtil.parseInt(detail[9], 0);

        InetSocket socket = new InetSocket();
        socket.proto = proto;

        switch (state) {
            case 0:
                socket.state = "";
                break;
            case Const.INET_STATE_ESTABLISHED:
                socket.state = "ESTABLISHED";
                break;
            case Const.INET_STATE_SYN_SENT:
                socket.state = "SYN_SENT";
                break;
            case Const.INET_STATE_SYN_RECV:
                socket.state = "SYN_RECV";
                break;
            case Const.INET_STATE_FIN_WAIT1:
                socket.state = "FIN_WAIT1";
                break;
            case Const.INET_STATE_FIN_WAIT2:
                socket.state = "FIN_WAIT2";
                break;
            case Const.INET_STATE_TIME_WAIT:
                socket.state = "TIME_WAIT";
                break;
            case Const.INET_STATE_CLOSE:
                socket.state = "CLOSE";
                break;
            case Const.INET_STATE_CLOSE_WAIT:
                socket.state = "CLOSE_WAIT";
                break;
            case Const.INET_STATE_LAST_ACK:
                socket.state = "LAST_ACK";
                break;
            case Const.INET_STATE_LISTEN:
                socket.state = "LISTEN";
                break;
            case Const.INET_STATE_CLOSING:
                socket.state = "CLOSING";
                break;
            default:
                socket.state = "UNKNOWN";
                break;
        }

        socket.localAddress = localAddress.getHostAddress();
        socket.localPort = localPort;
        socket.remoteAddress = remoteAddress.getHostAddress();
        socket.remotePort = remotePort;
        socket.listening = socket.remotePort == 0;
        socket.transmitQueue = transmitQueue;
        socket.receiveQueue = receiveQueue;
        socket.uid = uid;
        socket.inode = inode;

        return socket;
    };

    private static final Parser<InetSocket> udpParser = (proto, line) -> {
        String[] detail = line.split(" +");
        if (detail.length < 10) {
            throw new Exception();
        }

        InetAddress localAddress = parseInetAddress(detail[1].split(":")[0]);
        int localPort = CommonUtil.parseInt(detail[1].split(":")[1], 16, 0);
        InetAddress remoteAddress = parseInetAddress(detail[2].split(":")[0]);
        int remotePort = CommonUtil.parseInt(detail[2].split(":")[1], 16, 0);
        long transmitQueue = CommonUtil.parseLong(detail[4].split(":")[0], 16, 0);
        long receiveQueue = CommonUtil.parseLong(detail[4].split(":")[1], 16, 0);
        int state = CommonUtil.parseInt(detail[3], 16, 0);
        int uid = CommonUtil.parseInt(detail[7], 0);
        long inode = CommonUtil.parseInt(detail[9], 0);

        InetSocket socket = new InetSocket();
        socket.proto = proto;

        switch (state) {
            case Const.INET_STATE_ESTABLISHED:
                socket.state = "ESTABLISHED";
                break;
            case Const.INET_STATE_CLOSE:
                socket.state = "CLOSE";
                break;
            default:
                socket.state = "UNKNOWN";
                break;
        }

        socket.localAddress = localAddress.getHostAddress();
        socket.localPort = localPort;
        socket.remoteAddress = remoteAddress.getHostAddress();
        socket.remotePort = remotePort;
        socket.listening = remoteAddress.isAnyLocalAddress();
        socket.transmitQueue = transmitQueue;
        socket.receiveQueue = receiveQueue;
        socket.uid = uid;
        socket.inode = inode;

        return socket;
    };

    private static final Parser<InetSocket> rawParser = (proto, line) -> {
        String[] detail = line.split(" +");
        if (detail.length < 10) {
            throw new Exception();
        }

        InetAddress localAddress = parseInetAddress(detail[1].split(":")[0]);
        int localPort = CommonUtil.parseInt(detail[1].split(":")[1], 16, 0);
        InetAddress remoteAddress = parseInetAddress(detail[2].split(":")[0]);
        int remotePort = CommonUtil.parseInt(detail[2].split(":")[1], 16, 0);
        long transmitQueue = CommonUtil.parseLong(detail[4].split(":")[0], 16, 0);
        long receiveQueue = CommonUtil.parseLong(detail[4].split(":")[1], 16, 0);
        int state = CommonUtil.parseInt(detail[3], 16, 0);
        int uid = CommonUtil.parseInt(detail[7], 0);
        long inode = CommonUtil.parseInt(detail[9], 0);

        InetSocket socket = new InetSocket();
        socket.proto = proto;
        socket.state = String.valueOf(state);
        socket.localAddress = localAddress.getHostAddress();
        socket.localPort = localPort;
        socket.remoteAddress = remoteAddress.getHostAddress();
        socket.remotePort = remotePort;
        socket.listening = remoteAddress.isAnyLocalAddress();
        socket.transmitQueue = transmitQueue;
        socket.receiveQueue = receiveQueue;
        socket.uid = uid;
        socket.inode = inode;

        return socket;
    };

    private static final Parser<UnixSocket> unixParser = (proto, line) -> {
        String[] detail = line.split(" +");
        if (detail.length < 6) {
            throw new Exception();
        }

        long refCount = CommonUtil.parseLong(detail[1], 16, 0);
        long protocol = CommonUtil.parseLong(detail[2], 16, 0);
        long flags = CommonUtil.parseLong(detail[3], 16, 0);
        int type = CommonUtil.parseInt(detail[4], 16, 0);
        int state = CommonUtil.parseInt(detail[5], 16, 0);
        long inode = CommonUtil.parseLong(detail[6], 0);
        String path = detail.length > 7 ? detail[7] : null;

        UnixSocket socket = new UnixSocket();
        if ((int) protocol == 0) {
            socket.proto = proto;
        } else {
            socket.proto = "??";
        }

        switch (type) {
            case Const.SOCK_STREAM:
                socket.type = "STREAM";
                break;
            case Const.SOCK_DGRAM:
                socket.type = "DGRAM";
                break;
            case Const.SOCK_RAW:
                socket.type = "RAW";
                break;
            case Const.SOCK_RDM:
                socket.type = "RDM";
                break;
            case Const.SOCK_SEQPACKET:
                socket.type = "SEQPACKET";
                break;
            default:
                socket.type = "UNKNOWN";
        }

        switch (state) {
            case Const.UNIX_STATE_FREE:
                socket.state = "FREE";
                break;
            case Const.UNIX_STATE_UNCONNECTED:
                /*
                 * Unconnected sockets may be listening
                 * for something.
                 */
                if ((flags & Const.UNIX_FLAG_ACCEPTCON) != 0) {
                    socket.state = "LISTENING";
                } else {
                    socket.state = "";
                }
                break;
            case Const.UNIX_STATE_CONNECTING:
                socket.state = "CONNECTING";
                break;
            case Const.UNIX_STATE_CONNECTED:
                socket.state = "CONNECTED";
                break;
            case Const.UNIX_STATE_DISCONNECTING:
                socket.state = "DISCONNECTING";
                break;
            default:
                socket.state = "UNKNOWN";
        }

        socket.flags = "[ ";
        if ((flags & Const.UNIX_FLAG_ACCEPTCON) != 0)
            socket.flags = socket.flags.concat("ACC ");
        if ((flags & Const.UNIX_FLAG_WAITDATA) != 0)
            socket.flags = socket.flags.concat("W ");
        if ((flags & Const.UNIX_FLAG_NOSPACE) != 0)
            socket.flags = socket.flags.concat("N ");
        socket.flags = socket.flags.concat("]");

        socket.refCnt = refCount;
        socket.inode = inode;
        socket.listening = (state == Const.UNIX_STATE_UNCONNECTED) && ((flags & Const.UNIX_FLAG_ACCEPTCON) != 0);

        if (!TextUtils.isEmpty(path)) {
            socket.path = path;
            socket.readable = FileUtil.canRead(path);
            socket.writable = FileUtil.canWrite(path);
        }

        return socket;
    };

}
