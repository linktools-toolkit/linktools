package org.ironman.framework.util;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class CommandUtil {

    public static String execCommand(String... commands) throws IOException {
        Process proccess = null;
        BufferedReader stdout = null;
        BufferedReader stderr = null;

        try {
            proccess = Runtime.getRuntime().exec(commands);

            stdout = new BufferedReader(new InputStreamReader(proccess.getInputStream()));
            stderr = new BufferedReader(new InputStreamReader(proccess.getErrorStream()));

            String line;
            StringBuilder sb = new StringBuilder();
            while ((line = stdout.readLine()) != null) {
                sb.append(line).append('\n');
            }

            return sb.toString();
        } finally {
            if (stdout != null) {
                try {
                    stdout.close();
                } catch (Exception e) {
                    // ignore
                }
            }
            if (stderr != null) {
                try {
                    stderr.close();
                } catch (Exception e) {
                    // ignore
                }
            }
            if (proccess != null) {
                try {
                    proccess.destroy();
                } catch (Exception e) {
                    // ignore
                }
            }
        }
    }

}
