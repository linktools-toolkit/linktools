package debug;

import android.tools.ICommand;

import com.beust.jcommander.Parameters;

@Subcommand()
@Parameters(commandNames = "debug")
public class Command implements ICommand {

    @Override
    public void execute(JCommander commander) throws Exception {
        throw new Exception("not implemented");
    }
}
