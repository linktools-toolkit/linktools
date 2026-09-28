package org.ironman.framework.proxy;

import java.lang.reflect.Method;

public interface IProxyHandler {

    Object handle(ProxyHandlerHolder handler, Object obj, Method method, Object[] args) throws Throwable;

}
