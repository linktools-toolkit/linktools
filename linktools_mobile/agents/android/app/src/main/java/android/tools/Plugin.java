package android.tools;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameters;

import java.util.Map;

@Parameters(commandNames = "env")
public class Plugin implements IPlugin, ICommand {

    @Override
    public void init(JCommander builder) {
        builder.addCommand(this);
    }

    @Override
    public void execute(JCommander commander) throws Exception {
        Map<String, String> env = System.getenv();
        for (String key : env.keySet()) {
            Output.out.println(key + "=" + env.get(key));
        }
    }
}
