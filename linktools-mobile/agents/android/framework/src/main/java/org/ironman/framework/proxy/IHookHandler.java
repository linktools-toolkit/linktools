package org.ironman.framework.proxy;

public interface IHookHandler {

    void hook() throws Exception;

    void unhook() throws Exception;

}
