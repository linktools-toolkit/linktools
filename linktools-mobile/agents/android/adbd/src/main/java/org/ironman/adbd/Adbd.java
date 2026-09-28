package org.ironman.adbd;

import android.os.Parcel;
import android.os.Parcelable;
import android.os.Process;

/**
 * Created by hu on 18-5-22.
 */

public class Adbd implements Parcelable {

    static {
        System.loadLibrary("adbd");
    }

    private int mIsDaemon = 0;
    private int mServerPort = 0;
    private int mProcessId = 0;
    private String[] mEnvironments = null;

    public Adbd(boolean deamon, int port, String... envs) {
        mIsDaemon = deamon ? 1 : 0;
        mServerPort = port;
        mEnvironments = envs;
    }

    public boolean isDaemon() {
        return mIsDaemon == 1;
    }

    public int getPort() {
        return mServerPort;
    }

    public int getPid() {
        return mProcessId;
    }

    public String[] getEnvironments() {
        return mEnvironments;
    }

    /**
     * 运行
     * @param traceMask -1显示所有log
     * @return
     */
    public boolean run(int traceMask) {
        if (!isRunning()) {
            mProcessId = nativeRun(mIsDaemon, mServerPort, getEnvironments(), traceMask);
        }
        return mProcessId > 0;
    }

    /**
     * 结束进程
     */
    public void kill() {
        if (isRunning()) {
            Process.killProcess(mProcessId);
            mProcessId = 0;
        }
    }

    /**
     * 进程还在不在
     * @return
     */
    public boolean isRunning() {
        return mProcessId > 0 && nativeIsRunning(mProcessId);
    }

    private native static int nativeRun(int daemon, int port, String[] envs, int traceMask);
    private native static boolean nativeIsRunning(int pid);

    private Adbd(Parcel in) {
        mIsDaemon = in.readInt();
        mServerPort = in.readInt();
        mProcessId = in.readInt();
    }

    @Override
    public int describeContents() {
        return 0;
    }

    @Override
    public void writeToParcel(Parcel dest, int flags) {
        dest.writeInt(mIsDaemon);
        dest.writeInt(mServerPort);
        dest.writeInt(mProcessId);
    }

    public static final Creator<Adbd> CREATOR = new Creator<Adbd>() {
        @Override
        public Adbd createFromParcel(Parcel in) {
            return new Adbd(in);
        }

        @Override
        public Adbd[] newArray(int size) {
            return new Adbd[size];
        }
    };
}
