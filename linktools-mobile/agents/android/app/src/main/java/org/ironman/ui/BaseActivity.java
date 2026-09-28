package org.ironman.ui;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import androidx.annotation.NonNull;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.appcompat.app.AppCompatActivity;
import android.util.SparseArray;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by hu on 18-12-3.
 */

public class BaseActivity extends AppCompatActivity {

    private static volatile int sRequestCode = 3721;
    private static final SparseArray<Callback> sCallbackMap = new SparseArray<>();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
    }

    public void requestPermissions(String[] permissions, RequestPermissionsCallback callback) {
        synchronized (this) {
            int requestCode = sRequestCode++ & ~0x80000000;
            sCallbackMap.put(requestCode, callback);
            ActivityCompat.requestPermissions(this, permissions, requestCode);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        RequestPermissionsCallback listener = (RequestPermissionsCallback) sCallbackMap.get(requestCode);
        if (listener != null) {
            sCallbackMap.remove(requestCode);
            List<String> deny = new ArrayList<>();
            for (int i = 0; i < permissions.length; i++) {
                if (grantResults[i] == PackageManager.PERMISSION_GRANTED) {
                    // ignore
                } else if (ContextCompat.checkSelfPermission(this, permissions[i]) != PackageManager.PERMISSION_GRANTED) {
                    deny.add(permissions[i]);
                }
            }
            listener.onResult(deny.toArray(new String[0]));
        }
    }

    public void startActivityForResult(Intent intent, StartActivityResultCallback callback) {
        synchronized (this) {
            int requestCode = sRequestCode++ & ~0x80000000;
            sCallbackMap.put(requestCode, callback);
            startActivityForResult(intent, requestCode);
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        StartActivityResultCallback callback = (StartActivityResultCallback) sCallbackMap.get(requestCode);
        if (callback != null) {
            sCallbackMap.remove(requestCode);
            callback.onResult(data);
        }
    }

    private interface Callback {

    }

    public interface RequestPermissionsCallback extends Callback {
        void onResult(String[] deniedPermissions);
    }

    public interface StartActivityResultCallback extends Callback {
        void onResult(Intent data);
    }

    @SuppressLint("ObsoleteSdkInt")
    public void startSettingsActivity() {
        Intent intent = new Intent();
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
        if (Build.VERSION.SDK_INT >= 9) {
            intent.setAction("android.settings.APPLICATION_DETAILS_SETTINGS");
            intent.setData(Uri.fromParts("package", getPackageName(), null));
        } else {
            intent.setAction(Intent.ACTION_VIEW);
            intent.setClassName("com.android.settings", "com.android.settings.InstalledAppDetails");
            intent.putExtra("com.android.settings.ApplicationPkgName", getPackageName());
        }
        startActivity(intent);
    }
}