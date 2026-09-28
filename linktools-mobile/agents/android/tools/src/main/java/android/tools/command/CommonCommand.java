package android.tools.command;

import android.app.Application;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.text.TextUtils;
import android.tools.ICommand;
import android.tools.Output;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;

import org.ironman.annotation.Subcommand;
import org.ironman.framework.Environment;
import org.ironman.framework.bean.os.File;
import org.ironman.framework.util.ActivityUtil;
import org.ironman.framework.util.FileUtil;
import org.ironman.framework.util.GsonUtil;
import org.ironman.framework.util.NetworkUtil;
import org.ironman.framework.util.PackageUtil;
import org.ironman.framework.util.ProcessUtil;

import java.net.URI;
import java.util.ArrayList;
import java.util.List;

@Subcommand(order = 0)
@Parameters(commandNames = "common")
public class CommonCommand implements ICommand {

    @Parameter(names = {"--list-process"}, order = 0, description = "List all processes")
    private boolean list_process = false;

    @Parameter(names = {"--list-file"}, order = 0, description = "List all files")
    private String list_file = null;

    @Parameter(names = {"--list-tcp-sock"}, order = 2, description = "List all tcp sockets")
    private boolean list_tcp_sock = false;

    @Parameter(names = {"--list-udp-sock"}, order = 3, description = "List all udp sockets")
    private boolean list_udp_sock = false;

    @Parameter(names = {"--list-raw-sock"}, order = 4, description = "List all raw sockets")
    private boolean list_raw_sock = false;

    @Parameter(names = {"--list-unix-sock"}, order = 5, description = "List all unix sockets")
    private boolean list_unix_sock = false;

    @Parameter(names = {"--top-package"}, order = 100, description = "Display top-level package")
    private boolean top_package = false;

    @Parameter(names = {"--top-activity"}, order = 102, description = "Display top-level activity")
    private boolean top_activity = false;

    @Parameter(names = {"--get-clipboard"}, order = 103, description = "Get clipboard content")
    private boolean get_clipboard = false;

    @Parameter(names = {"--set-clipboard"}, order = 104, description = "Set clipboard content")
    private String set_clipboard = null;

    @Parameter(names = {"--usage-access"}, order = 200, description = "Start usage access settings")
    private boolean usage_access = false;

    @Override
    public void execute(JCommander commander) throws Exception {
        if (list_process) {
            Output.out.println(GsonUtil.toJson(ProcessUtil.getProcessList()));
        } else if (!TextUtils.isEmpty(list_file)) {
            java.io.File dir = new java.io.File(list_file);
            java.io.File[] files = FileUtil.listFiles(dir);
            if (files == null || files.length == 0) {
                Output.out.println(GsonUtil.toJson(new File[0]));
                return;
            }
            URI uri = dir.toURI();
            List<File> result = new ArrayList<>(files.length);
            for (java.io.File file : files) {
                File f = new File();
                f.name = uri.relativize(file.toURI()).getPath();
                f.path = file.getAbsolutePath();
                f.isDirectory = FileUtil.isDirectory(file);
                f.readable = FileUtil.canRead(file);
                f.writable = FileUtil.canWrite(file);
                f.executable = FileUtil.canExecute(file);
                result.add(f);
            }
            Output.out.println(GsonUtil.toJson(result));
        } else if (list_tcp_sock) {
            Output.out.println(GsonUtil.toJson(NetworkUtil.getTcpSockets()));
        } else if (list_udp_sock) {
            Output.out.println(GsonUtil.toJson(NetworkUtil.getUdpSockets()));
        } else if (list_raw_sock) {
            Output.out.println(GsonUtil.toJson(NetworkUtil.getRawSockets()));
        } else if (list_unix_sock) {
            Output.out.println(GsonUtil.toJson(NetworkUtil.getUnixSockets()));
        } else if (top_package) {
            String packageName = PackageUtil.getTopPackage();
            if (!TextUtils.isEmpty(packageName)) {
                Output.out.println(packageName);
            }
        } else if (top_activity) {
            String activityName = ActivityUtil.getTopActivity();
            if (!TextUtils.isEmpty(activityName)) {
                Output.out.println(activityName);
            }
        } else if (get_clipboard) {
            Application app = Environment.getApplication();
            ClipboardManager cm = (ClipboardManager) app.getSystemService(Context.CLIPBOARD_SERVICE);
            ClipData data = cm.getPrimaryClip();
            if (data != null && data.getItemCount() > 0) {
                Output.out.println(data.getItemAt(0).coerceToText(app));
            }
        } else if (!TextUtils.isEmpty(set_clipboard)) {
            Application app = Environment.getApplication();
            ClipboardManager cm = (ClipboardManager) app.getSystemService(Context.CLIPBOARD_SERVICE);
            cm.setPrimaryClip(ClipData.newPlainText("text", set_clipboard));
        } else if (usage_access) {
            ActivityUtil.startUsageAccessSettings();
        }
    }
}
