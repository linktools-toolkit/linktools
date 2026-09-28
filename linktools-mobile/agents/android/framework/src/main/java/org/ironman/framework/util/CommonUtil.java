package org.ironman.framework.util;

import android.os.Parcel;

import java.io.Closeable;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;

/**
 * Created by hu on 19-2-13.
 */

public class CommonUtil {

    private static final String TAG = CommonUtil.class.getSimpleName();
    private static final String DEV_NULL = "/dev/null";

    public static int parseInt(Object object) {
        return parseInt(object, 10, 0);
    }

    public static int parseInt(Object object, int defValue) {
        return parseInt(object, 10, defValue);
    }

    public static int parseInt(Object object, int radix, int defValue) {
        try {
            return Integer.parseInt(String.valueOf(object), radix);
        } catch (Exception e) {
            return defValue;
        }
    }

    public static long parseLong(Object object) {
        return parseLong(object, 10, 0);
    }

    public static long parseLong(Object object, long defValue) {
        return parseLong(object, 10, defValue);
    }

    public static long parseLong(Object object, int radix, long defValue) {
        try {
            return Long.parseLong(String.valueOf(object), radix);
        } catch (Exception e) {
            return defValue;
        }
    }

    public static void closeQuietly(Closeable closeable) {
        try {
            if (closeable != null) {
                closeable.close();
            }
        } catch (IOException e) {
            LogUtil.printStackTrace(TAG, e, null);
        }
    }

    public static void runQuietly(Runnable runnable) {
        PrintStream out = System.out;
        PrintStream err = System.err;
        FileOutputStream os = null;
        PrintStream ps = null;

        try {
            try {
                os = new FileOutputStream(DEV_NULL);
                ps = new PrintStream(os);
                System.setOut(ps);
                System.setErr(ps);
            } catch (Exception e) {
                LogUtil.printStackTrace(TAG, e, null);
            }

            runnable.run();

        } finally {
            if (out != System.out) {
                System.setOut(out);
            }
            if (err != System.err) {
                System.setErr(err);
            }
            CommonUtil.closeQuietly(ps);
            CommonUtil.closeQuietly(os);
        }
    }

    private static byte getByte(byte[] bytes, int index) {
        if (bytes != null && index >= 0 && index < bytes.length) {
            return bytes[index];
        }
        return 0;
    }

    private static void setByte(byte[] bytes, int index, byte value) {
        if (bytes != null && index >= 0 && index < bytes.length) {
            bytes[index] = value;
        }
    }

    public static int bytes2Int(byte[] bytes, int offset) {
        return (getByte(bytes, offset + 3) & 0xFF |
                (getByte(bytes, offset + 2) & 0xFF) << 8 |
                (getByte(bytes, offset + 1) & 0xFF) << 16 |
                (getByte(bytes, offset) & 0xFF) << 24);
    }

    public static void int2Bytes(int num, byte[] bytes, int offset) {
        setByte(bytes, offset, (byte) ((num >> 24) & 0xFF));
        setByte(bytes, offset + 1, (byte) ((num >> 16) & 0xFF));
        setByte(bytes, offset + 2, (byte) ((num >> 8) & 0xFF));
        setByte(bytes, offset + 3, (byte) ((num) & 0xFF));
    }

    public static short reverseBytes(short num) {
        return (short)(
                (((num >> 8) & 0xFF)) |
                (((num) & 0xFF) << 8)
        );
    }

    public static int reverseBytes(int num) {
        return (((num & (0xFF << 24) >> 24)) |
                ((num & (0xFF << 16)) >> 8) |
                ((num & (0xFF << 8)) << 8) |
                ((num & (0xFF)) << 24));
    }

    public static void writeBytes(Parcel data, byte[] bytes) {
        for (int i = 0; i < bytes.length; i += 4) {
            data.writeInt(getByte(bytes, i) & 0xFF |
                    (getByte(bytes, i + 1) & 0xFF) << 8 |
                    (getByte(bytes, i + 2) & 0xFF) << 16 |
                    (getByte(bytes, i + 3) & 0xFF) << 24);
        }
    }

    public static byte[] readBytes(Parcel data) {
        byte[] bytes = new byte[data.dataAvail()];
        for (int i = 0; i < bytes.length; i += 4) {
            int num = data.readInt();
            setByte(bytes, i + 3, (byte) ((num >> 24) & 0xFF));
            setByte(bytes, i + 2, (byte) ((num >> 16) & 0xFF));
            setByte(bytes, i + 1, (byte) ((num >> 8) & 0xFF));
            setByte(bytes, i, (byte) ((num) & 0xFF));
        }
        return bytes;
    }
}
