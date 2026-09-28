package org.ironman.ui;

import android.Manifest;
import android.app.PendingIntent;
import android.content.Context;
import android.content.Intent;
import android.os.Build;
import android.os.Bundle;
import android.os.Looper;
import android.os.Message;
import android.text.TextUtils;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import org.ironman.adbd.Adbd;
import org.ironman.adbd.AdbdManager;

import java.lang.ref.WeakReference;
import java.net.Inet4Address;
import java.net.InetAddress;
import java.net.NetworkInterface;
import java.net.SocketException;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;

public class AdbdActivity extends BaseActivity implements View.OnClickListener {

    private static final int DEFAULT_START_PORT = 5555;
    private static final int DEFAULT_END_PORT = 5558;
    private static final String[] PERMISSIONS;

    static {
        List<String> permissions = new ArrayList<>();
        permissions.add(Manifest.permission.CALL_PHONE);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            permissions.add(Manifest.permission.POST_NOTIFICATIONS);
            permissions.add(Manifest.permission.READ_MEDIA_IMAGES);
            permissions.add(Manifest.permission.READ_MEDIA_VIDEO);
            permissions.add(Manifest.permission.READ_MEDIA_AUDIO);
        }
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) {
            permissions.add(Manifest.permission.WRITE_EXTERNAL_STORAGE);
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.JELLY_BEAN &&
                Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) {
            permissions.add(Manifest.permission.READ_EXTERNAL_STORAGE);
        }
        PERMISSIONS = permissions.toArray(new String[0]);
    }


    private Handler mHandler = null;
    private TextView mIp = null;
    private EditText mStartPort = null;
    private EditText mEndPort = null;
    private TextView mStatus = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_adbd);

        requestPermissions(PERMISSIONS, deniedPermissions -> {
            if (deniedPermissions.length > 0) {
                showToast("please grant all permissions");
                startSettingsActivity();
            }
        });

        mHandler = new Handler(this);

        mIp = findViewById(R.id.tv_ip);
        mStartPort = findViewById(R.id.et_start_port);
        mStartPort.setHint(String.valueOf(DEFAULT_START_PORT));
        mEndPort = findViewById(R.id.et_end_port);
        mEndPort.setHint(String.valueOf(DEFAULT_END_PORT));
        mStatus = findViewById(R.id.tv_status);

        Button startAdbd = findViewById(R.id.btn_start_adbd);
        startAdbd.setOnClickListener(this);
        Button stopAdbd = findViewById(R.id.btn_stop_adbd);
        stopAdbd.setOnClickListener(this);

        refreshStatus();
    }

    @Override
    public void onClick(View v) {
        if (v.getId() == R.id.btn_start_adbd) {

            String strStart = mStartPort.getText().toString();
            String strEnd = mEndPort.getText().toString();

            int start = DEFAULT_START_PORT;
            if (!TextUtils.isEmpty(strStart)) {
                try {
                    start = Integer.parseInt(strStart);
                } catch (NumberFormatException e) {
                    // ignore
                }
            }

            int end = DEFAULT_END_PORT;
            if (!TextUtils.isEmpty(strEnd)) {
                try {
                    end = Integer.parseInt(strEnd);
                } catch (NumberFormatException e) {
                    // ignore
                }
            }

            for (int i = start; i <= end; i++) {
                final int port = i;
                AdbdManager.run(getApplicationContext(), port, new AdbdManager.OnRunListener() {
                    @Override
                    public void onRun() {
                        showToast("start port " + port);
                        Context context = getApplicationContext();
                        Intent intent = new Intent(context, AdbdActivity.class);
                        int flags = PendingIntent.FLAG_ONE_SHOT;
                        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.S) {
                            flags = PendingIntent.FLAG_MUTABLE;
                        }
                        PendingIntent pendingIntent = PendingIntent.getActivity(context, 0, intent, flags);
                        AdbdManager.startForeground(context, "Adbd", "adbd is running ...", pendingIntent);
                    }

                    @Override
                    public void onError(Exception e) {
                        showToast(e.getMessage());
                    }
                });
            }

        } else if (v.getId() == R.id.btn_stop_adbd) {

            AdbdManager.killAll(getApplicationContext(), new AdbdManager.OnKillAllListener() {
                @Override
                public void onKillAll() {
                    showToast("stop all adbd");
                    AdbdManager.stopForeground(getApplicationContext());
                }

                @Override
                public void onError(Exception e) {
                    showToast(e.getMessage());
                }
            });
        }
    }

    private void setIpStatus() {
        StringBuilder ipsSb = new StringBuilder();
        try {
            for (Enumeration<NetworkInterface> nets = NetworkInterface.getNetworkInterfaces();
                 nets.hasMoreElements(); ) {
                NetworkInterface net = nets.nextElement();
                StringBuilder ipSb = new StringBuilder();
                for (Enumeration<InetAddress> addrs = net.getInetAddresses();
                     addrs.hasMoreElements(); ) {
                    InetAddress addr = addrs.nextElement();
                    if (addr instanceof Inet4Address && !addr.isLoopbackAddress()) {
                        ipSb.append(addr.getHostAddress()).append(" ");
                    }
                }
                String ip = ipSb.toString();
                if (!TextUtils.isEmpty(ip)) {
                    ipsSb.append(net.getDisplayName()).append("=").append(ip).append("\n");
                }
            }
        } catch (SocketException e) {
            e.printStackTrace();
        }
        mIp.setText(ipsSb);
    }

    private void setPortStatus(List<Adbd> adbds) {
        if (adbds == null || adbds.isEmpty()) {
            AdbdManager.stopForeground(getApplicationContext());
            mStatus.setText("");
            return;
        }
        StringBuilder sb = new StringBuilder("opening: ");
        for (Adbd adbd : adbds) {
            sb.append(adbd.getPort()).append(", ");
        }
        sb.delete(sb.lastIndexOf(", "), sb.length());
        mStatus.setText(sb.toString());
    }

    public void refreshStatus() {
        mHandler.sendMessageDelayed(Message.obtain(), 1000);
    }

    public void showToast(String message) {
        Toast.makeText(getApplicationContext(), message, Toast.LENGTH_SHORT).show();
    }

    private static class Handler extends android.os.Handler {

        private final WeakReference<AdbdActivity> mActivty;

        private Handler(AdbdActivity activity) {
            super(Looper.getMainLooper());
            mActivty = new WeakReference<>(activity);
        }

        @Override
        public void handleMessage(Message msg) {
            super.handleMessage(msg);

            AdbdActivity activity = mActivty.get();
            if (activity != null) {

                activity.setIpStatus();

                AdbdManager.getAll(activity.getApplicationContext(), new AdbdManager.OnGetAllListener() {
                    @Override
                    public void onGetAll(List<Adbd> adbds) {
                        AdbdActivity activity = mActivty.get();
                        if (activity != null) {
                            activity.setPortStatus(adbds);
                            activity.refreshStatus();
                        }
                    }

                    @Override
                    public void onError(Exception e) {
                        AdbdActivity activity = mActivty.get();
                        if (activity != null) {
                            activity.showToast(e.getMessage());
                        }
                    }
                });
            }
        }
    }
}
