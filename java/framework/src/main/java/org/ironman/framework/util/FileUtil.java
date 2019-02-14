package org.ironman.framework.util;

import org.ironman.framework.Const;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

/**
 * Created by hu on 19-2-13.
 */

public class FileUtil {

    public static boolean canWrite(String fileName) {
        return new File(fileName).canWrite();
    }

    public static boolean canRead(String fileName) {
        return new File(fileName).canRead();
    }

    public static boolean canExecute(String fileName) {
        return new File(fileName).canExecute();
    }

    public static String readFile(String fileName) throws IOException {
        StringBuilder result;
        BufferedReader reader = null;

        try {
            reader = new BufferedReader(new FileReader(fileName));
            result = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                result.append(line);
                result.append(Const.LINE_BREAK);
            }
        } finally {
            CommonUtil.closeQuietly(reader);
        }

        return result.toString();
    }

}
