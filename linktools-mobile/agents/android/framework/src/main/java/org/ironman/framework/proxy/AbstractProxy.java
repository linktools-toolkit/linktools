package org.ironman.framework.proxy;

import android.app.ActivityThread;
import android.app.Application;

import org.ironman.framework.util.LogUtil;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public abstract class AbstractProxy {

    private static final String TAG = AbstractProxy.class.getSimpleName();
    private static final Map<Class<? extends AbstractProxy>, AbstractProxy> sInstances = new HashMap<>();

    private boolean mInit = false;
    private boolean mHooked = false;

    protected final List<IHookHandler> mHookHandler = new ArrayList<>();
    protected final Map<String, ProxyHandlerHolder> mProxyHandler = new ConcurrentHashMap<>();

    protected abstract void internalInit() throws Exception;

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
                    for (IHookHandler handler : mHookHandler) {
                        try {
                            handler.hook();
                        } catch (Exception e) {
                            LogUtil.printStackTrace(TAG, e, null);
                        }
                    }
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
                    ListIterator<IHookHandler> it = mHookHandler.listIterator(mHookHandler.size());
                    while (it.hasPrevious()) {
                        IHookHandler handler = it.previous();
                        try {
                            handler.unhook();
                        } catch (Exception e) {
                            LogUtil.printStackTrace(TAG, e, null);
                        }
                    }
                    mHooked = false;
                } catch (Exception e) {
                    LogUtil.printStackTrace(TAG, e, null);
                }
            }
        }
    }

    protected Application getApplication() {
        return ActivityThread.currentApplication();
    }

    protected void registerHookHandler(IHookHandler handler) {
        mHookHandler.add(handler);
    }

    protected Object newProxyInstance(Object obj) {
        return Proxy.newProxyInstance(
                obj.getClass().getClassLoader(),
                obj.getClass().getInterfaces(),
                (proxy, method, args) -> invokeProxyHandler(obj, method, args)
        );
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

    protected Object invokeProxyHandler(Object obj, Method method, Object[] args) throws Throwable {
        LogUtil.v(TAG, "handle: %s.%s", method.getDeclaringClass(), method.getName());
        ProxyHandlerHolder current = mProxyHandler.get(method.getName());
        if (current == null || current.handler == null) {
            try {
                return method.invoke(obj, args);
            } catch (InvocationTargetException e) {
                throw e.getTargetException();
            }
        }
        return current.handler.handle(current, obj, method, args);
    }
}
