package org.ironman.framework.util;

import android.util.Log;

/**
 * Created by hu on 18-12-17.
 */

public class LogUtil {

    private static LogImpl sLogImpl = new LogImpl() {

        @Override
        public void v(final String tag, final String format, final Object... args) {
            Log.v(tag, (args == null || args.length == 0) ? format : String.format(format, args));
        }

        @Override
        public void i(final String tag, final String format, final Object... args) {
            Log.i(tag, (args == null || args.length == 0) ? format : String.format(format, args));
        }

        @Override
        public void d(final String tag, final String format, final Object... args) {
            Log.d(tag, (args == null || args.length == 0) ? format : String.format(format, args));
        }

        @Override
        public void w(final String tag, final String format, final Object... args) {
            Log.w(tag, (args == null || args.length == 0) ? format : String.format(format, args));
        }

        @Override
        public void e(final String tag, final String format, final Object... args) {
            Log.e(tag, (args == null || args.length == 0) ? format : String.format(format, args));
        }

        @Override
        public void printStackTrace(String tag, Throwable tr, String format, Object... args) {
            String log = (args == null || args.length == 0) ? format : String.format(format, args);
            if (log == null) {
                log = "";
            }
            Log.e(tag, log + "  " + Log.getStackTraceString(tr));
        }
    };
    ;

    public static void setLogImpl(LogImpl impl) {
        sLogImpl = impl;
    }

    public static LogImpl getImpl() {
        return sLogImpl;
    }

    public static void v(final String tag, final String format, final Object... args) {
        if (sLogImpl != null) {
            sLogImpl.v(tag, format, args);
        }
    }

    public static void e(final String tag, final String format, final Object... args) {
        if (sLogImpl != null) {
            sLogImpl.e(tag, format, args);
        }
    }

    public static void w(final String tag, final String format, final Object... args) {
        if (sLogImpl != null) {
            sLogImpl.w(tag, format, args);
        }
    }

    public static void i(final String tag, final String format, final Object... args) {
        if (sLogImpl != null) {
            sLogImpl.i(tag, format, args);
        }
    }

    public static void d(final String tag, final String format, final Object... args) {
        if (sLogImpl != null) {
            sLogImpl.d(tag, format, args);
        }
    }

    public static void printStackTrace(String tag, Throwable tr) {
        if (sLogImpl != null) {
            sLogImpl.printStackTrace(tag, tr, null);
        }
    }

    public static void printStackTrace(String tag, Throwable tr, final String format, final Object... args) {
        if (sLogImpl != null) {
            sLogImpl.printStackTrace(tag, tr, format, args);
        }
    }

    public interface LogImpl {

        void v(final String tag, final String format, final Object... args);

        void i(final String tag, final String format, final Object... args);

        void w(final String tag, final String format, final Object... args);

        void d(final String tag, final String format, final Object... args);

        void e(final String tag, final String format, final Object... args);

        void printStackTrace(String tag, Throwable tr, final String format, final Object... args);

    }
}
