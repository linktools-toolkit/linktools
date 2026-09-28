package org.ironman.framework;

import android.app.ActivityManager;
import android.app.ActivityThread;
import android.app.Application;
import android.content.Context;
import android.content.pm.PackageManager;
import android.os.Looper;
import android.os.Process;
import android.text.TextUtils;

import org.ironman.framework.proxy.AbstractProxy;
import org.ironman.framework.proxy.ActivityManagerProxy;
import org.ironman.framework.proxy.ActivityTaskManagerProxy;
import org.ironman.framework.proxy.PackageManagerProxy;
import org.ironman.framework.proxy.PermissionManagerProxy;
import org.ironman.framework.util.LogUtil;
import org.ironman.framework.util.ReflectHelper;

import java.io.File;

public final class Environment {

    private static final String TAG = Environment.class.getSimpleName();

    private static final Singleton<Application> sApplication = new Singleton<Application>() {
        @Override
        protected Application create() {
            if (Looper.getMainLooper() == null) {
                Looper.prepareMainLooper();
            }
            if (ActivityThread.currentActivityThread() == null) {
                ActivityThread.systemMain();
                try {
                    Fixer fixer = new Fixer(ActivityThread.currentApplication());
                    fixer.fixPackageName();
                    fixer.fixAppPath();
                    fixer.fixDataPath();
                    fixer.fixLibraryPath();
                    fixer.fixServices();
                } catch (Exception e) {
                    LogUtil.printStackTrace(TAG, e);
                }
            }
            return ActivityThread.currentApplication();
        }
    };

    public static Application getApplication() {
        return sApplication.get();
    }

    public static PackageManager getPackageManager() {
        return getApplication().getPackageManager();
    }

    public static String getPackageName() {
        return getApplication().getPackageName();
    }

    public static ActivityManager getActivityManager() {
        return getSystemService(Context.ACTIVITY_SERVICE);
    }

    @SuppressWarnings("unchecked")
    public static <T> T getSystemService(String name) {
        return (T) Environment.getApplication().getSystemService(name);
    }

    private static class Fixer {

        private final ReflectHelper helper = ReflectHelper.getDefault();
        private final Application application;
        private final Object baseContext;
        private final Object loadedApk;

        public Fixer(Application application) {
            try {
                this.application = application;
                this.baseContext = helper.get(application, "mBase");
                this.loadedApk = helper.get(baseContext, "mPackageInfo");
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        public void fixPackageName() {
            String packageName = System.getenv("APP_PACKAGE_NAME");
            if (TextUtils.isEmpty(packageName)) {
                PackageManager packageManager = application.getPackageManager();
                String[] packages = packageManager.getPackagesForUid(Process.myUid());
                if (packages != null && packages.length > 0) {
                    packageName = packages[0];
                }
            }
            if (TextUtils.isEmpty(packageName)) {
                return;
            }
            LogUtil.i(TAG, "Fix package name: " + packageName);
            try {
                helper.set(loadedApk, "mPackageName", packageName);
                // context.getBasePackageName()
                helper.set(baseContext, "mBasePackageName", packageName);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
            try {
                // context.getOpPackageName()
                helper.set(baseContext, "mOpPackageName", packageName);
            } catch (NoSuchFieldException e) {
                // ignore
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
            try {
                // android 12: context.getOpPackageName()
                Object source = helper.get(baseContext, "mAttributionSource");
                Object state = helper.get(source, "mAttributionSourceState");
                helper.set(state, "packageName", packageName);
            } catch (NoSuchFieldException e) {
                // ignore
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        public void fixAppPath() {
            String appDir = System.getenv("APP_PATH");
            if (TextUtils.isEmpty(appDir)) {
                appDir = System.getenv("CLASSPATH");
            }
            if (TextUtils.isEmpty(appDir)) {
                return;
            }
            File appDirFile = new File(appDir);
            LogUtil.i(TAG, "Fix app path: " + appDirFile.getAbsolutePath());
            try {
                helper.set(loadedApk, "mAppDir", appDirFile.getAbsolutePath());
                helper.set(loadedApk, "mResDir", appDirFile.getAbsolutePath());
            } catch (Exception e) {
                LogUtil.printStackTrace(TAG, e, null);
            }
        }

        public void fixDataPath() {
            String dataDir = System.getenv("APP_DATA_PATH");
            if (TextUtils.isEmpty(dataDir)) {
                return;
            }
            File dataDirFile = new File(dataDir);
            LogUtil.i(TAG, "Fix data path: " + dataDirFile.getAbsolutePath());
            try {
                helper.set(loadedApk, "mDataDir", dataDirFile.getAbsolutePath());
                helper.set(loadedApk, "mDataDirFile", dataDirFile);
            } catch (Exception e) {
                LogUtil.printStackTrace(TAG, e, null);
            }
            try {
                helper.set(loadedApk, "mDeviceProtectedDataDirFile", dataDirFile);
            } catch (NoSuchFieldException e) {
                // ignore
            } catch (Exception e) {
                LogUtil.printStackTrace(TAG, e, null);
            }
            try {
                helper.set(loadedApk, "mCredentialProtectedDataDirFile", dataDirFile);
            } catch (NoSuchFieldException e) {
                // ignore
            } catch (Exception e) {
                LogUtil.printStackTrace(TAG, e, null);
            }
            if (!dataDirFile.exists() && !dataDirFile.mkdirs()) {
                // ignore
            }
        }

        public void fixLibraryPath() {
            String libDir = System.getenv("APP_LIBRARY_PATH");
            if (TextUtils.isEmpty(libDir)) {
                return;
            }
            File libDirFile = new File(libDir);
            LogUtil.i(TAG, "Fix library path: " + libDirFile.getAbsolutePath());
            try {
                helper.set(loadedApk, "mLibDir", libDirFile.getAbsolutePath());
            } catch (Exception e) {
                LogUtil.printStackTrace(TAG, e, null);
            }
            if (!libDirFile.exists() && !libDirFile.mkdirs()) {
                // ignore
            }
        }

        public void fixServices() {
            AbstractProxy[] activityManagerProxies = {
                    AbstractProxy.get(ActivityManagerProxy.class),
                    AbstractProxy.get(ActivityTaskManagerProxy.class),
            };

            for (AbstractProxy proxy : activityManagerProxies) {
                proxy.registerProxyHandler(new String[]{
                        "startActivity",
                        "startService",
                        "broadcastIntent",
                }, (handler, obj, method, args) -> {
                    LogUtil.d(TAG, "%s: set applicationThread null", method.getName());
                    args[0] = null; // set applicationThread = null;
                    return handler.handle(obj, method, args);
                });
            }

            AbstractProxy.get(ActivityManagerProxy.class).hook();
            AbstractProxy.get(ActivityTaskManagerProxy.class).hook();
            AbstractProxy.get(PackageManagerProxy.class).hook();
            AbstractProxy.get(PermissionManagerProxy.class).hook();
        }
    }

}
