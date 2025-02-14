package org.ironman.framework.proxy;

import android.app.ActivityThread;
import android.content.pm.PackageManager;

import org.ironman.framework.util.LogUtil;
import org.ironman.framework.util.ReflectHelper;

import java.lang.reflect.Field;

public class PermissionManagerProxy extends AbstractProxy {

    private static final String TAG = PermissionManagerProxy.class.getSimpleName();

    @Override
    protected void internalInit() throws Exception {
        ReflectHelper helper = ReflectHelper.getDefault();

        try {
            Class<?> holder = ActivityThread.class;
            Object pm = helper.invoke(ActivityThread.class, "getPermissionManager");
            Field field = helper.getField(ActivityThread.class, "sPermissionManager");

            registerHookHandler(new IHookHandler() {
                @Override
                public void hook() throws Exception {
                    LogUtil.d(TAG, "Hook " + holder.getName() + "." + field.getName());
                    field.set(holder, newProxyInstance(pm));
                }

                @Override
                public void unhook() throws Exception {
                    LogUtil.d(TAG, "Unhook " + holder.getName() + "." + field.getName());
                    field.set(holder, pm);
                }
            });
        } catch (Exception e) {
            LogUtil.i(TAG, "Failed to get ActivityThread.sPermissionManager: %s", e);
        }

        try {
            PackageManager packageManager = getApplication().getPackageManager();
            Object holder = helper.invoke(packageManager, "getPermissionManager");
            Object pm = helper.get(holder, "mPermissionManager");
            Field field = helper.getField(holder, "mPermissionManager");

            registerHookHandler(new IHookHandler() {
                @Override
                public void hook() throws Exception {
                    LogUtil.d(TAG, "Hook " + holder.getClass().getName() + "." + field.getName());
                    field.set(holder, newProxyInstance(pm));
                }

                @Override
                public void unhook() throws Exception {
                    LogUtil.d(TAG, "Unhook " + holder.getClass().getName() + "." + field.getName());
                    field.set(holder, pm);
                }
            });
        } catch (Exception e) {
            LogUtil.i(TAG, "Failed to get PermissionManager.mPermissionManager: %s", e);
        }
    }
}
