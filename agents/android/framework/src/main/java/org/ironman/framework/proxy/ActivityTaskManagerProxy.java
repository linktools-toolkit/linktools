package org.ironman.framework.proxy;

import android.os.Build;

import org.ironman.framework.util.LogUtil;
import org.ironman.framework.util.ReflectHelper;

import java.lang.reflect.Field;
import java.lang.reflect.Proxy;

public class ActivityTaskManagerProxy extends AbstractProxy {

    private static final String TAG = ActivityManagerProxy.class.getSimpleName();

    private Object mActivityTaskManagerSingleton = null;
    private Object mActivityTaskManager = null;
    private Field mSingletonInstanceField = null;

    @Override
    protected void internalInit() throws Exception {
        ReflectHelper helper = ReflectHelper.getDefault();

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            // "android/app/ActivityTaskManager.java"
            mActivityTaskManagerSingleton = helper.get(
                    "android.app.ActivityTaskManager",
                    "IActivityTaskManagerSingleton"
            );
            mActivityTaskManager = helper.invoke(
                    mActivityTaskManagerSingleton,
                    "get"
            );
            mSingletonInstanceField = helper.getField(
                    mActivityTaskManagerSingleton,
                    "mInstance"
            );
        }

    }

    @Override
    protected void internalHook() throws Exception {
        if (mActivityTaskManagerSingleton != null && mActivityTaskManager != null && mSingletonInstanceField != null) {
            LogUtil.d(TAG, "Hook " + mActivityTaskManagerSingleton.getClass().getName() + "." + mSingletonInstanceField.getName());
            mSingletonInstanceField.set(
                    mActivityTaskManagerSingleton,
                    Proxy.newProxyInstance(
                            mActivityTaskManager.getClass().getClassLoader(),
                            mActivityTaskManager.getClass().getInterfaces(),
                            (proxy, method, args) -> handle(mActivityTaskManager, method, args)
                    )
            );
        }
    }

    @Override
    protected void internalUnhook() throws Exception {
        if (mActivityTaskManagerSingleton != null && mActivityTaskManager != null && mSingletonInstanceField != null) {
            LogUtil.d(TAG, "Unhook " + mActivityTaskManagerSingleton.getClass().getName() + "." + mSingletonInstanceField.getName());
            mSingletonInstanceField.set(
                    mActivityTaskManagerSingleton,
                    mActivityTaskManager
            );
        }
    }
}
