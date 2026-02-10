package org.ironman.adbd;

import android.app.PendingIntent;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.ServiceConnection;
import android.os.IBinder;

import java.util.List;

/**
 * Created by hu on 18-5-23.
 */

public class AdbdManager {

    private static final AdbdManager sInstance = new AdbdManager();

    public interface OnErrorListener {
        void onError(Exception e);
    }

    public interface OnKillAllListener extends OnErrorListener {
        void onKillAll();
    }

    public interface OnGetAllListener extends OnErrorListener {
        void onGetAll(List<Adbd> adbds);
    }

    public interface OnRunListener extends OnErrorListener {
        void onRun();
    }

    public static void run(Context packageContext, final int port) {
        run(packageContext, port, null);
    }

    public static void run(Context packageContext, final int port, final OnRunListener listener) {
        sInstance.bindService(packageContext, new OnBindServiceListener() {
            @Override
            public void onBinder(IAdbdInterface service) {
                try {
                    service.run(port);
                    if (listener != null) {
                        listener.onRun();
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                    if (listener != null) {
                        listener.onError(e);
                    }
                }
            }
        });
    }

    public static void killAll(Context packageContext) {
        killAll(packageContext, null);
    }

    public static void killAll(Context packageContext, final OnKillAllListener listener) {
        sInstance.bindService(packageContext, new OnBindServiceListener() {
            @Override
            public void onBinder(IAdbdInterface service) {
                try {
                    service.killAll();
                    if (listener != null) {
                        listener.onKillAll();
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                    if (listener != null) {
                        listener.onError(e);
                    }
                }
            }
        });
    }

    public static void getAll(Context packageContext, final OnGetAllListener listener) {
        sInstance.bindService(packageContext, new OnBindServiceListener() {
            @Override
            public void onBinder(IAdbdInterface service) {
                try {
                    assert listener != null;
                    listener.onGetAll(service.getAll());
                } catch (Exception e) {
                    e.printStackTrace();
                    listener.onError(e);
                }
            }
        });
    }

    public static void startForeground(Context context, String title, String text, PendingIntent pendingIntent) {
        Intent intent = new Intent(context, AdbdService.class);
        intent.putExtra(AdbdService.COMMAND, AdbdService.START_FOREGROUND_COMMAND);
        intent.putExtra(AdbdService.TITLE, title);
        intent.putExtra(AdbdService.TEXT, text);
        intent.putExtra(AdbdService.PENDING_INTENT, pendingIntent);
        context.startService(intent);
    }

    public static void stopForeground(Context context) {
        Intent intent = new Intent(context, AdbdService.class);
        intent.putExtra(AdbdService.COMMAND, AdbdService.STOP_FOREGROUND_COMMAND);
        context.startService(intent);
    }

    private IAdbdInterface mService = null;

    private interface OnBindServiceListener {
        void onBinder(IAdbdInterface service);
    }

    private void bindService(Context packageContext, final OnBindServiceListener listener) {
        if (mService == null) {
            Intent intent = new Intent();
            intent.setClass(packageContext, AdbdService.class);
            packageContext.bindService(intent, new ServiceConnection() {
                @Override
                public void onServiceConnected(ComponentName name, IBinder service) {
                    mService = IAdbdInterface.Stub.asInterface(service);
                    listener.onBinder(mService);
                }
                @Override
                public void onServiceDisconnected(ComponentName name) {
                    mService = null;
                }
                @Override
                public void onBindingDied(ComponentName name) {
                    mService = null;
                }
            }, Context.BIND_AUTO_CREATE);
        } else {
            listener.onBinder(mService);
        }
    }
}
