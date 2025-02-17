package org.ironman.framework.util;

import org.ironman.framework.bean.os.Process;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class ProcessUtil {

    private static final String TAG = ProcessUtil.class.getSimpleName();

    public static String getProcessName(int pid, String defaultValue) {
        try {
            return FileUtil.readString("/proc/" + pid + "/cmdline");
        } catch (IOException e) {
            LogUtil.printStackTrace(TAG, e);
            return defaultValue;
        }
    }

    public static List<Process> getProcessList() {
        List<Process> processList = new ArrayList<>();

        File proc = new File("/proc/");
        File[] files = proc.listFiles((dir, name) -> name.matches("^\\d+$"));
        if (files == null) {
            return processList;
        }
        for (File dir : files) {
            Process process = new Process();

            process.pid = Integer.parseInt(dir.getName());

            BufferedReader reader = null;
            try {
                reader = new BufferedReader(new FileReader(new File(dir, "stat")));
                String line;

                while ((line = reader.readLine()) != null) {
                    Pattern pattern = Pattern.compile("^(\\S+)\\s+\\((.+)\\)\\s+(\\S+)\\s+(\\S+)\\s+(\\S+)\\s+(\\S+)\\s+(\\S+)\\s+(\\S+)\\s+(\\S+)\\s+(\\S+)\\s+(\\S+)\\s+(\\S+)\\s+(\\S+)\\s+(\\S+)\\s+(\\S+)\\s+(\\S+)\\s+(\\S+)\\s+(\\S+)\\s+(.*)$");
                    Matcher matcher = pattern.matcher(line);
                    if (matcher.matches()) {
                        process.cmd = "[" + matcher.group(2) + "]";
                        process.name = process.cmd;
                        process.state = matcher.group(3);
                        process.ppid = Long.parseLong(matcher.group(4));
                        process.pgid = Long.parseLong(matcher.group(5));
                        process.sid = Long.parseLong(matcher.group(6));
                        process.tty = Long.parseLong(matcher.group(7));
                        process.utime = Long.parseLong(matcher.group(8));
                        process.stime = Long.parseLong(matcher.group(9));
                        process.nice = Long.parseLong(matcher.group(12));
                        process.startTime = Long.parseLong(matcher.group(13));
                        process.vsz = Long.parseLong(matcher.group(14));
                        process.rss = Long.parseLong(matcher.group(15));
                    }
                }
            } catch (IOException e) {
                // ignore
            } finally {
                if (reader != null) {
                    CommonUtil.closeQuietly(reader);
                    reader = null;
                }
            }

            try {
                reader = new BufferedReader(new FileReader(new File(dir, "status")));
                String line;

                while ((line = reader.readLine()) != null) {
                    if (line.startsWith("Uid:")) {
                        process.uid = Long.parseLong(line.split("\\s+")[1]);
                    } else if (line.startsWith("Gid:")) {
                        process.gid = Long.parseLong(line.split("\\s+")[1]);
                    }
                }

                reader.close();
            } catch (IOException e) {
                // ignore
            } finally {
                if (reader != null) {
                    CommonUtil.closeQuietly(reader);
                    reader = null;
                }
            }

            try {
                reader = new BufferedReader(new FileReader(new File(dir, "cmdline")));
                String line;
                StringBuilder buffer = new StringBuilder();
                while ((line = reader.readLine()) != null) {
                    buffer.append(line);
                }

                String cmdline = buffer.toString().trim().replace('\u0000', ' ');
                if (!cmdline.isEmpty()) {
                    process.cmd = cmdline;
                    String[] args = cmdline.split(" +");
                    if (args.length > 0) {
                        String name = args[0];
                        int index = name.lastIndexOf("/");
                        process.name = index >= 0 ? name.substring(index + 1) : name;
                    }
                }

                reader.close();
            } catch (IOException e) {
                // ignore
            } finally {
                if (reader != null) {
                    CommonUtil.closeQuietly(reader);
                    reader = null;
                }
            }

            processList.add(process);

        }

        return processList;
    }

}
