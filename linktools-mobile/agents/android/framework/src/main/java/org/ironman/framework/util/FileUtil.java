package org.ironman.framework.util;

import org.ironman.framework.Const;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

/**
 * Created by hu on 19-2-13.
 */

public class FileUtil {

    public static File[] listFiles(File file) {
        try {
            return file.listFiles();
        } catch (SecurityException e) {
            return null;
        }
    }

    public static boolean canWrite(String fileName) {
        return canWrite(new File(fileName));
    }

    public static boolean canWrite(File file) {
        try {
            return file.canWrite();
        } catch (SecurityException e) {
            return false;
        }
    }

    public static boolean canRead(String fileName) {
        return canRead(new File(fileName));
    }

    public static boolean canRead(File file) {
        try {
            return file.canRead();
        } catch (SecurityException e) {
            return false;
        }
    }

    public static boolean canExecute(String fileName) {
        return canExecute(new File(fileName));
    }

    public static boolean canExecute(File file) {
        try {
            return file.canExecute();
        } catch (SecurityException e) {
            return false;
        }
    }

    public static boolean isDirectory(String fileName) {
        return isDirectory(new File(fileName));
    }

    public static boolean isDirectory(File file) {
        try {
            return file.isDirectory();
        } catch (SecurityException e) {
            return false;
        }
    }

    public static String readString(String fileName) throws IOException {
        return readString(new File(fileName));
    }

    public static String readString(File file) throws IOException {
        StringBuilder result = new StringBuilder();
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(new FileReader(file));
            String line;
            while ((line = reader.readLine()) != null) {
                result.append(line);
                result.append(Const.LINE_SEPARATOR);
            }
        } finally {
            CommonUtil.closeQuietly(reader);
        }

        return result.toString();
    }

    public static void writeString(String fileName, String buffer) throws IOException {
        writeString(new File(fileName), buffer);
    }

    public static void writeString(File file, String buffer) throws IOException {
        FileWriter writer = null;
        try {
            writer = new FileWriter(file);
            writer.write(buffer);
        } finally {
            CommonUtil.closeQuietly(writer);
        }
    }

    public static void copy(String src, String dest) throws IOException {
        FileInputStream fis = null;
        FileOutputStream fos = null;

        try {
            fis = new FileInputStream(src);
            fos = new FileOutputStream(dest);
            int lenth = 0;
            byte[] buffer = new byte[1024];
            while (-1 != (lenth = fis.read(buffer))) {
                fos.write(buffer, 0, lenth);
            }
            fos.flush();
        } finally {
            CommonUtil.closeQuietly(fis);
            CommonUtil.closeQuietly(fos);
        }
    }

}
