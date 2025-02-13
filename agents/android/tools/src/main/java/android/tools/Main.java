package android.tools;

import android.annotation.SuppressLint;
import android.tools.command.PluginCommand;
import android.tools.exception.UsageException;
import android.tools.processor.CommandUtils;
import android.util.Log;
import com.beust.jcommander.JCommander;

import org.ironman.framework.util.LogUtil;

import java.io.File;
import java.io.OutputStream;
import java.io.PrintStream;

import dalvik.system.PathClassLoader;

public class Main {

    private static final String TAG = "android-tools";

    private void printLogo() {
        Output.out.println("┌───────────────────────────────────────────────────────┐");
        Output.out.println("│             Output from android.tools.Main            │");
        Output.out.println("└───────────────────────────────────────────────────────┘");
    }

    @SuppressLint("PrivateApi")
    private IPlugin loadPlugin() throws Exception {
        String path = System.getenv("AGENT_PLUGIN_PATH");
        if (path != null) {
            if (!new File(path).exists()) {
                throw new Exception("Plugin not found: " + path);
            }
            PathClassLoader classLoader = new PathClassLoader(path, IPlugin.class.getClassLoader());
            try {
                Class<?> klass = classLoader.loadClass("android.tools.Plugin");
                return (IPlugin) klass.newInstance();
            } catch (ClassNotFoundException e) {
                throw new Exception("Plugin class 'android.tools.Plugin' not found: " + path);
            }
        }
        return null;
    }

    private void internalMain(String[] args) throws Throwable {
        IPlugin plugin = loadPlugin();

        JCommander commander = JCommander.newBuilder()
                .programName(Main.class.getName())
                .addObject(this)
                .build();
        CommandUtils.addCommands(commander);
        if (plugin != null) {
            commander.addCommand("plugin", new PluginCommand());
            plugin.init(commander.getCommands().get("plugin"));
        }

        commander.parse(args);
        JCommander subCommander = commander.getCommands().get(commander.getParsedCommand());
        if (subCommander != null) {
            for (Object command : subCommander.getObjects()) {
                if (command instanceof ICommand) {
                    LogUtil.i(TAG, "Execute %s", command.getClass().getSimpleName());
                    ((ICommand) command).execute(subCommander);
                }
            }
        } else {
            throw new UsageException(commander);
        }
    }

    public static void main(String[] args) {
        Output.out.setStream(System.out);
        Output.err.setStream(System.err);

        System.setOut(new PrintStream(new LoggerOutputStream() {
            @Override
            protected void log(String message) {
                Log.i("system.out", message);
            }
        }));
        System.setErr(new PrintStream(new LoggerOutputStream() {
            @Override
            protected void log(String message) {
                Log.e("system.err", message);
            }
        }));

        Thread.setDefaultUncaughtExceptionHandler((t, e) -> {
            Output.err.println(e);
            System.exit(-1);
        });

        try {
            Main main = new Main();
            main.printLogo();
            main.internalMain(args);
        } catch (UsageException e) {
            StringBuilder sb = new StringBuilder();
            e.getCommander().getUsageFormatter().usage(sb);
            Output.err.println(sb.toString());
            System.exit(-1);
        } catch (Throwable th) {
            Output.err.println(th);
            System.exit(-1);
        }
    }

    private static abstract class LoggerOutputStream extends OutputStream {

        private static class Cache {
            int length = 0;
            byte[] data = new byte[4 * 1024];
        }

        private static final ThreadLocal<Cache> local = new ThreadLocal<Cache>() {
            @Override
            protected Cache initialValue() {
                return new Cache();
            }
        };

        @Override
        public void write(int b) {
            Cache cache = local.get();
            if (cache != null) {
                if (b != '\n') {
                    cache.data[cache.length++] = (byte) b;
                    if (cache.length < cache.data.length) {
                        return;
                    }
                }
                log(new String(cache.data, 0, cache.length));
                cache.length = 0;
            }
        }

        protected abstract void log(String message);
    }
}
