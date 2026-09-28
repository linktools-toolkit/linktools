package org.ironman.framework.util;

import android.os.IBinder;
import android.os.Parcel;
import android.os.ServiceManager;

public class ServiceUtil {

    private static final String TAG = ServiceUtil.class.getSimpleName();

    public static String[] listServices() {
        String[] services = null;
        try {
            services = ServiceManager.listServices();
        } catch (Exception e) {
            LogUtil.printStackTrace(TAG, e);
        }
        if (services == null) {
            services = new String[0];
        }
        return services;
    }

    public static IBinder getService(String service, int timeout) {
        long timeMillis = System.currentTimeMillis() + timeout;

        do {
            IBinder binder = ServiceManager.getService(service);
            if (binder != null) {
                return binder;
            }
            try {
                Thread.sleep(100);
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        } while (System.currentTimeMillis() < timeMillis);

        return null;
    }

    public static abstract class Callback {

        public void onTransacting(IBinder binder, Parcel data) throws Exception {
            data.writeInterfaceToken(binder.getInterfaceDescriptor());
        }

        public void onTransacted(IBinder binder, boolean result, Parcel reply) throws Exception {
            reply.readException();
        }

        public void onError(Exception e) {
            throw new RuntimeException(e);
        }
    }

    public static boolean transact(String service, int code, Callback callback) throws Exception {
        return transact(ServiceManager.getService(service), code, callback);
    }

    public static boolean transact(IBinder binder, int code, Callback callback) {
        boolean result = false;
        Parcel data = Parcel.obtain();
        Parcel reply = Parcel.obtain();
        try {
            try {
                callback.onTransacting(binder, data);
                result = binder.transact(code, data, reply, 0);
                callback.onTransacted(binder, result, reply);
            } catch (Exception th) {
                callback.onError(th);
            }
        } finally {
            data.recycle();
            reply.recycle();
        }
        return result;
    }

}
