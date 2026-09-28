package org.ironman.adbd;

import android.annotation.SuppressLint;
import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Intent;
import android.graphics.BitmapFactory;
import android.os.Build;
import android.os.IBinder;
import android.os.RemoteException;
import android.util.Log;
import android.util.SparseArray;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by hu on 18-5-23.
 */

public class AdbdService extends Service {

    private static final String TAG = AdbdService.class.getSimpleName();

    private static final int FOREGROUND_ID = 12138;

    public static final String COMMAND = "COMMAND";
    public static final String TITLE = "TITLE";
    public static final String TEXT = "TEXT";
    public static final String PENDING_INTENT = "PENDING_INTENT";

    public static final int DEFAULT_COMMAND = 0;
    public static final int START_FOREGROUND_COMMAND = 1000;
    public static final int STOP_FOREGROUND_COMMAND = 1001;

    private static final boolean DEBUG = false;
    private static final SparseArray<Adbd> sAdbds = new SparseArray<>();

    private final IBinder mBinder = new IAdbdInterface.Stub() {
        @Override
        public boolean run(final int port) throws RemoteException {

            log("adbd run on port %d", port);

            if (port <= 0 || port > 65535) {
                throw remoteException("adbd port must be a positive number less than 65535");
            }

            Adbd adbd = sAdbds.get(port);
            if (adbd != null && adbd.isRunning()) {
                throw remoteException("adbd port %d is running", port);
            }

            String dataPath = getApplicationContext().getFilesDir().getParent();
            adbd = new Adbd(true, port, "ADB_DATA_PATH=" + dataPath);
            if (adbd.run(DEBUG ? -1 : 0)) {
                sAdbds.put(port, adbd);
            } else {
                return false;
            }

            return true;
        }

        @Override
        public void killAll() throws RemoteException {
            for (int i = 0; i < sAdbds.size(); i++) {
                Adbd adbd = sAdbds.valueAt(i);
                if (adbd.isRunning()) {
                    adbd.kill();
                }
            }
            sAdbds.clear();
        }

        @Override
        public List<Adbd> getAll() throws RemoteException {
            List<Adbd> adbds = new ArrayList<>(sAdbds.size());
            for (int i = 0; i < sAdbds.size(); i++) {
                Adbd adbd = sAdbds.valueAt(i);
                if (adbd.isRunning()) {
                    adbds.add(adbd);
                }
            }
            sAdbds.clear();
            for (Adbd adbd : adbds) {
                sAdbds.put(adbd.getPort(), adbd);
            }
            return adbds;
        }
    };

    @Override
    public IBinder onBind(Intent intent) {
        return mBinder;
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        try {
            int common = intent.getIntExtra(COMMAND, DEFAULT_COMMAND);
            switch (common) {
                case START_FOREGROUND_COMMAND:
                    String title = intent.getStringExtra(TITLE);
                    String text = intent.getStringExtra(TEXT);
                    PendingIntent pendingIntent = intent.getParcelableExtra(PENDING_INTENT);
                    startForeground(title, text, pendingIntent);
                    break;
                case STOP_FOREGROUND_COMMAND:
                    stopForeground();
                    break;
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        return super.onStartCommand(intent, flags, startId);
    }

    private RemoteException remoteException(String format, Object... args) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.ICE_CREAM_SANDWICH_MR1) {
            return new RemoteException(String.format(format, args));
        } else {
            return new RemoteException();
        }
    }

    private void log(String format, Object... args) {
        if (DEBUG) {
            Log.i(TAG, String.format(format, args));
        }
    }

    @SuppressLint("WrongConstant")
    private void startForeground(String title, String text, PendingIntent pendingIntent) {
        String channelId = "adbd_fore_service";
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationChannel channel = new NotificationChannel(channelId, channelId, NotificationManager.IMPORTANCE_HIGH);
            NotificationManager notificationManager = (NotificationManager) getSystemService(NOTIFICATION_SERVICE);
            if (notificationManager != null) {
                notificationManager.createNotificationChannel(channel);
            }
            Notification notification = new Notification.Builder(getApplicationContext(), channelId)
                    .setContentTitle(title)
                    .setContentText(text)
                    .setSmallIcon(R.drawable.adbd_ic_main_logo)
                    .setLargeIcon(BitmapFactory.decodeResource(getResources(), R.drawable.adbd_ic_main_logo))
                    .setWhen(System.currentTimeMillis())
                    .setContentIntent(pendingIntent)
                    .build();
            AdbdService.this.startForeground(FOREGROUND_ID, notification);
        }
    }

    private void stopForeground() {
        stopForeground(true);
    }
}
