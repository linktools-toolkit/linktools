package org.ironman.framework.bean.os;

import android.os.IBinder;

import java.util.List;

public class Service {
    public String name;
    public String desc;
    public IBinder binder;
    public Process owner;
    public List<Process> users;
}
