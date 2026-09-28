// IAdbdInterface.aidl
package org.ironman.adbd;

import org.ironman.adbd.Adbd;

// Declare any non-default types here with import statements

interface IAdbdInterface {
    /**
     * Demonstrates some basic types that you can use as parameters
     * and return values in AIDL.
     */
    boolean run(int port);

    void killAll();

    List<Adbd> getAll();
}
