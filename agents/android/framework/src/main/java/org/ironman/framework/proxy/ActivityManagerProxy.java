package org.ironman.framework.proxy;

import android.os.Build;

import org.ironman.framework.util.LogUtil;
import org.ironman.framework.util.ReflectHelper;

import java.lang.reflect.Field;
import java.lang.reflect.Proxy;

public class ActivityManagerProxy extends AbstractProxy {

    private static final String TAG = ActivityManagerProxy.class.getSimpleName();

    private Object mActivityManager = null;
    private Object mActivityManagerSingleton = null;
    private Field mSingletonInstanceField = null;

    @Override
    protected void internalInit() throws Exception {
        ReflectHelper helper = ReflectHelper.getDefault();

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            mActivityManagerSingleton = helper.get(
                    "android.app.ActivityManager",
                    "IActivityManagerSingleton"
            );
            mActivityManager = helper.invoke(
                    mActivityManagerSingleton,
                    "get"
            );
            mSingletonInstanceField = helper.getField(
                    mActivityManagerSingleton,
                    "mInstance"
            );
        } else {
            mActivityManagerSingleton = helper.get(
                    "android.app.ActivityManagerNative",
                    "gDefault"
            );
            mActivityManager = helper.invoke(
                    mActivityManagerSingleton,
                    "get"
            );
            mSingletonInstanceField = helper.getField(
                    mActivityManagerSingleton,
                    "mInstance"
            );
        }
    }

    @Override
    protected void internalHook() throws Exception {
        if (mActivityManagerSingleton != null && mActivityManager != null && mSingletonInstanceField != null) {
            LogUtil.d(TAG, "Hook " + mActivityManagerSingleton.getClass().getName() + "." + mSingletonInstanceField.getName());
            mSingletonInstanceField.set(
                    mActivityManagerSingleton,
                    Proxy.newProxyInstance(
                            mActivityManager.getClass().getClassLoader(),
                            mActivityManager.getClass().getInterfaces(),
                            (proxy, method, args) -> handle(mActivityManager, method, args)
                    )
            );
        }
    }

    @Override
    protected void internalUnhook() throws Exception {
        if (mActivityManagerSingleton != null && mActivityManager != null && mSingletonInstanceField != null) {
            LogUtil.d(TAG, "Unhook " + mActivityManagerSingleton.getClass().getName() + "." + mSingletonInstanceField.getName());
            mSingletonInstanceField.set(
                    mActivityManagerSingleton,
                    mActivityManager
            );
        }
    }
}
