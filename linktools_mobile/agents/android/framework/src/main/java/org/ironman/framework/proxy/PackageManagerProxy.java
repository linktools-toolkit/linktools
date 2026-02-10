package org.ironman.framework.proxy;

import android.app.ActivityThread;
import android.content.pm.PackageManager;

import org.ironman.framework.util.LogUtil;
import org.ironman.framework.util.ReflectHelper;

import java.lang.reflect.Field;
import java.lang.reflect.Proxy;

public class PackageManagerProxy extends AbstractProxy {

    private static final String TAG = PackageManagerProxy.class.getSimpleName();

    @Override
    protected void internalInit() throws Exception {
        ReflectHelper helper = ReflectHelper.getDefault();

        try {
            Class<?> holder = ActivityThread.class;
            Object pm = ActivityThread.getPackageManager();
            Field field = helper.getField(holder, "sPackageManager");

            registerHookHandler(new IHookHandler() {
                @Override
                public void hook() throws Exception {
                    LogUtil.d(TAG, "Hook " + holder.getName() + "." + field.getName());
                    field.set(
                            holder,
                            Proxy.newProxyInstance(
                                    pm.getClass().getClassLoader(),
                                    pm.getClass().getInterfaces(),
                                    (proxy, method, args) -> invokeProxyHandler(pm, method, args)
                            )
                    );
                }

                @Override
                public void unhook() throws Exception {
                    LogUtil.d(TAG, "Unhook " + holder.getName() + "." + field.getName());
                    field.set(holder, pm);
                }
            });
        } catch (Exception e) {
            LogUtil.i(TAG, "Failed to get ActivityThread.sPackageManager: %s", e);
        }

        try {
            PackageManager holder = getApplication().getPackageManager();
            Field field = helper.getField(holder, "mPM");
            Object pm = field.get(holder);
            if (pm == null) {
                throw new NullPointerException(field.getName() + " is null");
            }

            registerHookHandler(new IHookHandler() {
                @Override
                public void hook() throws Exception {
                    LogUtil.d(TAG, "Hook " + holder.getClass().getName() + "." + field.getName());
                    field.set(holder, newProxyInstance(pm));
                }

                @Override
                public void unhook() throws Exception {
                    LogUtil.d(TAG, "Unhook " + pm.getClass().getName() + "." + field.getName());
                    field.set(holder, pm);
                }
            });

        } catch (Exception e) {
            LogUtil.i(TAG, "Failed to get ApplicationPackageManager.mPM: %s", e);
        }
    }
}
