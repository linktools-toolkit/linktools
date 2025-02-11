package org.ironman.framework.proxy;

import org.ironman.framework.util.LogUtil;

import java.lang.reflect.Method;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public abstract class AbstractProxy {

    private static final String TAG = AbstractProxy.class.getSimpleName();
    private static final Map<Class<? extends AbstractProxy>, AbstractProxy> sInstances = new HashMap<>();

    private boolean mInit = false;
    private boolean mHooked = false;
    protected final Map<String, ProxyHandlerHolder> mProxyHandler = new ConcurrentHashMap<>();

    protected abstract void internalInit() throws Exception;
    protected abstract void internalHook() throws Exception;
    protected abstract void internalUnhook() throws Exception;

    protected AbstractProxy() {

    }

    public static <T extends AbstractProxy> T get(Class<T> klass) {
        synchronized (AbstractProxy.class) {
            AbstractProxy proxy = sInstances.get(klass);
            if (proxy == null) {
                try {
                    proxy = klass.newInstance();
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
                sInstances.put(klass, proxy);
            }
            return (T) proxy;
        }
    }

    public void hook() {
        synchronized (this) {
            if (!mHooked) {
                try {
                    if (!mInit) {
                        internalInit();
                        mInit = true;
                    }
                    internalHook();
                    mHooked = true;
                } catch (Exception e) {
                    LogUtil.printStackTrace(TAG, e, null);
                }
            }
        }
    }

    public void unhook() {
        synchronized (this) {
            if (mHooked) {
                try {
                    internalUnhook();
                    mHooked = false;
                } catch (Exception e) {
                    LogUtil.printStackTrace(TAG, e, null);
                }
            }
        }
    }

    public void registerProxyHandler(String method, IProxyHandler handler) {
        synchronized (this) {
            if (handler != null) {
                ProxyHandlerHolder holder = mProxyHandler.get(method);
                mProxyHandler.put(method, new ProxyHandlerHolder(handler, holder));
            } else {
                mProxyHandler.remove(method);
            }
        }
    }

    public void registerProxyHandler(String[] methods, IProxyHandler handler) {
        synchronized (this) {
            if (handler != null) {
                for (String method : methods) {
                    ProxyHandlerHolder holder = mProxyHandler.get(method);
                    mProxyHandler.put(method, new ProxyHandlerHolder(handler, holder));
                }
            } else {
                for (String method : methods) {
                    mProxyHandler.remove(method);
                }
            }
        }
    }

    protected Object handle(Object obj, Method method, Object[] args) throws Throwable {
        ProxyHandlerHolder handler = mProxyHandler.get(method.getName());
        return handler == null || handler.handler == null ?
                method.invoke(obj, args) :
                handler.handler.handle(handler, obj, method, args);
    }
}
