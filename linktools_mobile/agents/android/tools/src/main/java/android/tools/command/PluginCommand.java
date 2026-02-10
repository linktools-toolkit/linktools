package android.tools.command;

import android.tools.ICommand;
import android.tools.exception.UsageException;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameters;

@Parameters()
public class PluginCommand implements ICommand {

    @Override
    public void execute(JCommander commander) throws Exception {
        JCommander jCommander = commander.getCommands().get(commander.getParsedCommand());
        if (jCommander != null) {
            for (Object command : jCommander.getObjects()) {
                if (command instanceof ICommand) {
                    ((ICommand) command).execute(jCommander);
                }
            }
        }  else {
            throw new UsageException(commander);
        }
    }

}
