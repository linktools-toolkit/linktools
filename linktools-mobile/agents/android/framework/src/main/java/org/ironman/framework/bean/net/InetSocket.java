package org.ironman.framework.bean.net;

public class InetSocket extends Socket {

    public String localAddress;
    public int localPort;
    public String remoteAddress;
    public int remotePort;
    public int uid;
    public long transmitQueue;
    public long receiveQueue;
}
