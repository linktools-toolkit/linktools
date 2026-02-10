package org.ironman.framework;

/**
 * Created by hu on 19-2-13.
 */

public class Const {

    public static final String LINE_SEPARATOR = "\n";

    public static final int SOCK_DGRAM = 1;
    public static final int SOCK_STREAM = 2;
    public static final int SOCK_RAW = 3;
    public static final int SOCK_RDM = 4;
    public static final int SOCK_SEQPACKET = 5;
    public static final int SOCK_DCCP = 6;
    public static final int SOCK_PACKET = 10;

    public static final int INET_STATE_ESTABLISHED = 1;
    public static final int INET_STATE_SYN_SENT = 2;
    public static final int INET_STATE_SYN_RECV = 3;
    public static final int INET_STATE_FIN_WAIT1 = 4;
    public static final int INET_STATE_FIN_WAIT2 = 5;
    public static final int INET_STATE_TIME_WAIT = 6;
    public static final int INET_STATE_CLOSE = 7;
    public static final int INET_STATE_CLOSE_WAIT = 8;
    public static final int INET_STATE_LAST_ACK = 9;
    public static final int INET_STATE_LISTEN = 10;
    public static final int INET_STATE_CLOSING = 11;

    public static final int UNIX_STATE_FREE = 0;         /* not allocated                */
    public static final int UNIX_STATE_UNCONNECTED = 1;         /* unconnected to any socket    */
    public static final int UNIX_STATE_CONNECTING = 2;         /* in process of connecting     */
    public static final int UNIX_STATE_CONNECTED = 3;         /* connected to socket          */
    public static final int UNIX_STATE_DISCONNECTING = 4;         /* in process of disconnecting  */

    public static final int UNIX_FLAG_ACCEPTCON = (1 << 16);   /* performed a listen           */
    public static final int UNIX_FLAG_WAITDATA = (1 << 17);   /* wait data to read            */
    public static final int UNIX_FLAG_NOSPACE = (1 << 18);   /* no space to write            */
}
