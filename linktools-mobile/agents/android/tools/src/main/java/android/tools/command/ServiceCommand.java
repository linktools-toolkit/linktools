package android.tools.command;

import android.os.IBinder;
import android.os.Parcel;
import android.os.ServiceManager;
import android.text.TextUtils;
import android.tools.ICommand;
import android.tools.Main;
import android.tools.Output;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;

import org.ironman.annotation.Subcommand;
import org.ironman.framework.bean.os.Process;
import org.ironman.framework.bean.os.Service;
import org.ironman.framework.util.CommandUtil;
import org.ironman.framework.util.CommonUtil;
import org.ironman.framework.util.FileUtil;
import org.ironman.framework.util.GsonUtil;
import org.ironman.framework.util.LogUtil;
import org.ironman.framework.util.ProcessUtil;
import org.ironman.framework.util.ServiceUtil;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Scanner;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Created by hu on 18-12-18.
 */

@Subcommand(order = 2)
@Parameters(commandNames = "service")
public class ServiceCommand implements ICommand {

    private static final String TAG = ServiceCommand.class.getSimpleName();

    private static final String PATH_DEBUG_FS = "/sys/kernel/debug/binder/proc/";
    private static final String NODE_PREFIX = "Service Node: ";
    private static final Pattern NODE_PATTERN = Pattern.compile("^" + NODE_PREFIX + "\\D*(\\d+)$");

    @Parameter(names = {"--names"}, order = 1, variableArity = true, description = "Display target services")
    private List<String> names = new ArrayList<>();

    @Parameter(names = {"--detail"}, order = 2, description = "Display detail information.")
    private boolean detail = false;

    @Parameter(names = {"--node"}, order = 3, description = "Get service node", hidden = true)
    private String node = null;

    @Override
    public void execute(JCommander commander) {
        if (!TextUtils.isEmpty(node)) {
            if (canReadDebugFS()) {
                Output.out.println(NODE_PREFIX + getServiceNode(node));
            }
        } else {
            Output.out.println(GsonUtil.toJson(listServices()));
        }
    }

    private boolean canReadDebugFS() {
        try {
            File debugFS = new File(PATH_DEBUG_FS, String.valueOf(android.os.Process.myPid()));
            return debugFS.exists() && debugFS.canRead();
        } catch (SecurityException e) {
            return false;
        }
    }

    private List<Service> listServices() {
        List<String> allServices = Arrays.asList(ServiceUtil.listServices());

        List<String> targetServices = names;
        if (!targetServices.isEmpty()) {
            Iterator<String> it = targetServices.iterator();
            while (it.hasNext()) {
                if (!allServices.contains(it.next())) {
                    it.remove();
                }
            }
        } else {
            targetServices = allServices;
        }

        ConcurrentMap<Integer, Service> nodes = new ConcurrentHashMap<>();

        boolean multiThreading = detail && !targetServices.isEmpty();
        boolean fetchServiceNode = detail && canReadDebugFS();

        ThreadPoolExecutor executor = null;
        if (multiThreading) {
            executor = new ThreadPoolExecutor(
                    100,
                    100,
                    0,
                    TimeUnit.SECONDS,
                    new LinkedBlockingQueue<>());
        }

        List<Service> result = new ArrayList<>();
        for (String name : targetServices) {
            Service service = new Service();
            service.name = name;
            if (multiThreading) {
                executor.submit(() -> {
                    try {
                        service.binder = ServiceManager.getService(service.name);
                        service.desc = service.binder.getInterfaceDescriptor();
                    } catch (Exception e) {
                        service.binder = null;
                        service.desc = "";
                    }

                    try {
                        if (fetchServiceNode) {
                            int node = getServiceNodeFromSubprocess(service.name);
                            if (node != Integer.MIN_VALUE) {
                                nodes.put(node, service);
                            }
                        }
                    } catch (IOException e) {
                        // ignore
                    }
                });
            }
            result.add(service);
        }

        if (multiThreading) {
            executor.shutdown();
            try {
                if (!executor.awaitTermination(60, TimeUnit.SECONDS)) {
                    executor.shutdownNow();
                }
            } catch (InterruptedException ex) {
                executor.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }

        if (!nodes.isEmpty()) {
            iterateProcFs(nodes);
        }

        return result;
    }

    private Map<Integer, String> probeService(String service) {
        Map<Integer, String> map = new HashMap<>();

        if (ServiceManager.getService(service) != null) {
            for (int code = 1; code <= 1000; code++) {
                int transactCode = code;
                IBinder binder = ServiceUtil.getService(service, 3_000);
                if (binder != null && binder.isBinderAlive()) {
                    ServiceUtil.transact(binder, code, new ServiceUtil.Callback() {

                        @Override
                        public void onTransacted(IBinder binder, boolean result, Parcel reply) {
                            reply.readException();
                            map.put(transactCode, "success");
                        }

                        @Override
                        public void onError(Exception e) {
                            map.put(transactCode, String.format("%s: %s", e.getClass().getName(), e.getMessage()));
                        }
                    });
                }
            }
        }

        return map;
    }

    private static int getServiceNodeFromSubprocess(String service) throws IOException {
        String output = CommandUtil.execCommand(
                "app_process", "/", Main.class.getName(),
                "service", "--node", service);
        for (String line : output.split("\n")) {
            if (line.startsWith(NODE_PREFIX)) {
                Matcher matcher = NODE_PATTERN.matcher(line);
                if (matcher.matches()) {
                    int node = CommonUtil.parseInt(matcher.group(1), Integer.MIN_VALUE);
                    if (node != Integer.MIN_VALUE) {
                        return node;
                    }
                }
            }
        }
        return Integer.MIN_VALUE;
    }

    private static int getServiceNode(String service) {
        IBinder binder = null;
        LogUtil.i(TAG, "trying " + service + " ...");
        try {
            binder = ServiceManager.getService(service);
            LogUtil.i(TAG, "service got is " + binder);
            try {
                return getSelfHoldingNode();
            } catch (IOException e) {
                LogUtil.printStackTrace(TAG, e);
            }
        } catch (Exception e) {
            System.err.println("we did not find service " + service + ".");
        }
        return Integer.MIN_VALUE;
    }

    private static int getSelfHoldingNode() throws IOException {
        String path = new File(PATH_DEBUG_FS, String.valueOf(android.os.Process.myPid())).getAbsolutePath();
        String binderStat = FileUtil.readString(path);
//        LogUtil.i(TAG, binderStat);
        return extractStatAndGetServiceNode(binderStat);
    }

    protected static int extractStatAndGetServiceNode(String binderStat) {
        //find first "context binder"
        //first ref is usually service manager
        int svcMgrNodeIndex = binderStat.indexOf("context binder");
        if (svcMgrNodeIndex == -1) {
            throw new IllegalArgumentException("unreachable: the process does not have context binder");
        }
        svcMgrNodeIndex = binderStat.indexOf("node", svcMgrNodeIndex + 1);
        if (svcMgrNodeIndex == -1) {
            //wtf? cannot find any node?
            throw new IllegalArgumentException("cannot find any node in binder stat");
        }
        //next ref is the service we opened
        int svcNodeIndex = binderStat.indexOf("node", svcMgrNodeIndex + 1);
        Scanner scanner = new Scanner(binderStat.substring(svcNodeIndex + 1));
        scanner.next();
        return scanner.nextInt();
    }

    private static void iterateProcFs(Map<Integer, Service> nodes) {
        File debugFS = new File(PATH_DEBUG_FS);
        List<Process> processes = ProcessUtil.getProcessList();
        for (Process process : processes) {
            if (android.os.Process.myPid() != process.pid) {
                try {
                    File file = new File(debugFS, String.valueOf(process.pid));
                    String binderStat = FileUtil.readString(file);
                    procUserOrOwner(process, binderStat, nodes);
                } catch (IOException e) {
                    LogUtil.printStackTrace(TAG, e);
                    //this pid may have died while we iterate. ignore exception
                }
            }
        }
    }

    private static void procUserOrOwner(Process process, String binderStat, Map<Integer, Service> nodes) {
        for (Integer nodeId : nodes.keySet()) {
            Service service = nodes.get(nodeId);
            int beginIndex = binderStat.indexOf("context binder");
            int endIndex = binderStat.indexOf("binder proc state", beginIndex + 15);
            String symbol = String.format(Locale.getDefault(), "node %d", nodeId);
            if (beginIndex == -1) {
                //this process only holds one kind of binder, but not what we desired
                continue;
            }
            if (endIndex != -1) {
                binderStat = binderStat.substring(beginIndex + 1, endIndex);
            }
            for (String line : binderStat.split("\n")) {
                line = line.trim();
                if (line.contains(symbol + " ") || line.contains(symbol + ":")) {
                    if (line.startsWith("ref ")) {
                        //this process uses this binder node
                        service.owner = process;
                    } else if (line.startsWith("node ")) {
                        //this process holds this binder node
                        if (service.users == null) {
                            service.users = new ArrayList<>();
                        }
                        service.users.add(process);
                    } else {
                        //???wtf
                    }
                }
            }
        }
    }

}
