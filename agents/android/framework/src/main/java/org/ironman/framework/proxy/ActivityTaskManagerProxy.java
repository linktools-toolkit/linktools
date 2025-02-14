package org.ironman.framework.proxy;

import android.os.Build;

import org.ironman.framework.util.LogUtil;
import org.ironman.framework.util.ReflectHelper;

import java.lang.reflect.Field;

public class ActivityTaskManagerProxy extends AbstractProxy {

    private static final String TAG = ActivityManagerProxy.class.getSimpleName();

    @Override
    protected void internalInit() throws Exception {
        ReflectHelper helper = ReflectHelper.getDefault();

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            // "android/app/ActivityTaskManager.java"
            Object holder = helper.get(
                    "android.app.ActivityTaskManager",
                    "IActivityTaskManagerSingleton"
            );
            Object atm = helper.invoke(holder, "get");
            Field field = helper.getField(holder, "mInstance");

            registerHookHandler(new IHookHandler() {
                @Override
                public void hook() throws Exception {
                    LogUtil.d(TAG, "Hook " + holder.getClass().getName() + "." + field.getName());
                    field.set(holder, newProxyInstance(atm));
                }

                @Override
                public void unhook() throws Exception {
                    LogUtil.d(TAG, "Unhook " + holder.getClass().getName() + "." + field.getName());
                    field.set(holder, atm);
                }
            });
        }
    }
}
