package org.ironman.framework.proxy;

import android.app.ActivityThread;
import android.content.pm.PackageManager;
import android.util.Log;

import org.ironman.framework.util.LogUtil;
import org.ironman.framework.util.ReflectHelper;

import java.lang.reflect.Field;
import java.lang.reflect.Proxy;

public class PackageManagerProxy extends AbstractProxy {

    private static final String TAG = PackageManagerProxy.class.getSimpleName();

    private Object mPackageManager = null;
    private Field mPackageManagerField = null;

    private Object mPackageManager2 = null;
    private Field mPackageManagerField2 = null;

    @Override
    protected void internalInit() throws Exception {
        ReflectHelper helper = ReflectHelper.getDefault();

        try {
            mPackageManager = ActivityThread.getPackageManager();
            mPackageManagerField = helper.getField(ActivityThread.class, "sPackageManager");
        } catch (Exception e) {
            LogUtil.printStackTrace(TAG, e, "Failed to get ActivityThread.sPackageManager");
        }

        try {
            PackageManager pm = ActivityThread.currentApplication().getPackageManager();
            mPackageManagerField2 = helper.getField(pm, "mPM");
            mPackageManager2 = mPackageManagerField2.get(pm);
        } catch (Exception e) {
            LogUtil.printStackTrace(TAG, e, "Failed to get ApplicationPackageManager.mPM");
        }
    }

    @Override
    protected void internalHook() throws Exception {
        if (mPackageManagerField != null && mPackageManager != null) {
            Log.d(TAG, "Hook " + ActivityThread.class.getName() + "." + mPackageManagerField.getName());
            mPackageManagerField.set(
                    ActivityThread.class,
                    Proxy.newProxyInstance(
                            mPackageManager.getClass().getClassLoader(),
                            mPackageManager.getClass().getInterfaces(),
                            (proxy, method, args) -> handle(mPackageManager, method, args)
                    )
            );
        }

        if (mPackageManagerField2 != null && mPackageManager2 != null) {
            PackageManager pm = ActivityThread.currentApplication().getPackageManager();
            Log.d(TAG, "Hook " + pm.getClass().getName() + "." + mPackageManagerField2.getName());
            mPackageManagerField2.set(
                    pm,
                    Proxy.newProxyInstance(
                            mPackageManager2.getClass().getClassLoader(),
                            mPackageManager2.getClass().getInterfaces(),
                            (proxy, method, args) -> handle(mPackageManager2, method, args)
                    )
            );
        }
    }

    @Override
    protected void internalUnhook() throws Exception {
        if (mPackageManagerField != null && mPackageManager != null) {
            Log.d(TAG, "Unhook " + ActivityThread.class.getName() + "." + mPackageManagerField.getName());
            mPackageManagerField.set(
                    ActivityThread.class,
                    mPackageManager
            );
        }
        if (mPackageManagerField2 != null && mPackageManager2 != null) {
            PackageManager pm = ActivityThread.currentApplication().getPackageManager();
            Log.d(TAG, "Unhook " + pm.getClass().getName() + "." + mPackageManagerField2.getName());
            mPackageManagerField2.set(
                    pm,
                    mPackageManager2
            );
        }
    }
}
