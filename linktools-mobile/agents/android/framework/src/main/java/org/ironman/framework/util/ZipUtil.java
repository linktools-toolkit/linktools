package org.ironman.framework.util;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

public class ZipUtil {

    private static final String TAG = ZipUtil.class.getSimpleName();
    private static final int BUFFER_SIZE = 1024 * 10;

    private static void compressFile(File source, File path, ZipOutputStream zos) throws IOException {
        if (!source.canRead()) {
            LogUtil.e(TAG, "permission denied: %s", source);
            return;
        }

        LogUtil.d(TAG, "compress：%s", path);

        BufferedInputStream bis = null;
        FileInputStream fis = null;
        zos.putNextEntry(new ZipEntry(path.getAbsolutePath()));

        int read;
        byte[] buffer = new byte[BUFFER_SIZE];

        try {
            fis = new FileInputStream(source);
            bis = new BufferedInputStream(fis, BUFFER_SIZE);
            while ((read = bis.read(buffer, 0, BUFFER_SIZE)) != -1) {
                zos.write(buffer, 0, read);
            }
        } finally {
            CommonUtil.closeQuietly(bis);
            CommonUtil.closeQuietly(fis);
        }
    }

    private static void compress(File source, File base, ZipOutputStream zos) throws IOException {
        if (!source.exists()) {
            LogUtil.e(TAG, "not exists: %s", source);
            return;
        }

        if (source.isFile()) {
            compressFile(source, new File(base, source.getName()), zos);
            return;
        }

        try{
            File[] files = source.listFiles();
            if (files != null && files.length > 0) {
                for (File file : source.listFiles()) {
                    File path = new File(base, file.getName());
                    if (file.isFile()) {
                        compressFile(file, path, zos);
                    } else if (file.isDirectory()) {
                        compress(file, path, zos);
                    }
                }
            }
        } catch  (IOException e) {
            LogUtil.printStackTrace(TAG, e, null);
        }
    }

    public static void compress(String source, String dest) throws IOException {
        ZipOutputStream zos = null;
        try {
            zos = new ZipOutputStream(new FileOutputStream(dest));
            compress(new File(source), new File(""), zos);
        } finally{
            CommonUtil.closeQuietly(zos);
        }
    }
}
