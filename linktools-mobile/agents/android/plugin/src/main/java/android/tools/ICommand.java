package android.tools;

import com.beust.jcommander.JCommander;

/**
 * Created by hu on 18-12-17.
 */

public interface ICommand {

    void execute(JCommander commander) throws Exception;

}
