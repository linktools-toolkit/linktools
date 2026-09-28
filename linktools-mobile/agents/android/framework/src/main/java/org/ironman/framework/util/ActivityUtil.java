package org.ironman.framework.util;

import android.annotation.TargetApi;
import android.app.ActivityManager;
import android.content.ComponentName;
import android.content.Intent;
import android.os.Build;
import android.provider.Settings;

import org.ironman.framework.Environment;
import org.ironman.framework.Singleton;
import org.ironman.framework.proxy.ActivityManagerProxy;

import java.util.List;

public class ActivityUtil {

    private static final String TAG = ActivityUtil.class.getSimpleName();

    public static String getTopActivity() {
        ActivityManager am = Environment.getActivityManager();
        List<ActivityManager.RunningTaskInfo> tasks = am.getRunningTasks(0);
        if (tasks != null && !tasks.isEmpty()) {
            ComponentName topActivity = tasks.get(0).topActivity;
            if (topActivity != null) {
                return topActivity.getClassName();
            }
        }
        return null;
    }

    @TargetApi(Build.VERSION_CODES.LOLLIPOP)
    public static void startUsageAccessSettings() {
        startActivity(new Intent(Settings.ACTION_USAGE_ACCESS_SETTINGS));
    }

    public static void startActivity(Intent intent) {
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
        intent.addFlags(Intent.FLAG_ACTIVITY_MULTIPLE_TASK);
        Environment.getApplication().startActivity(intent);
    }

    public static void startService(Intent intent) {
        Environment.getApplication().startService(intent);
    }

    public static void sendBroadcast(Intent intent) {
        Environment.getApplication().sendBroadcast(intent);
    }
}
