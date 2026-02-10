package org.ironman.framework.proxy;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;

public class ProxyHandlerHolder {
    protected final IProxyHandler handler;
    protected final ProxyHandlerHolder next;

    public ProxyHandlerHolder(IProxyHandler handler, ProxyHandlerHolder next) {
        this.handler = handler;
        this.next = next;
    }

    public Object handle(Object obj, Method method, Object[] args) throws Throwable {
        if (next == null || next.handler == null) {
            try {
                return method.invoke(obj, args);
            } catch (InvocationTargetException e) {
                throw e.getTargetException();
            }
        }
        return next.handler.handle(next, obj, method, args);
    }
}