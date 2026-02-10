package android.tools.exception;

import com.beust.jcommander.JCommander;

public class UsageException extends Exception {

    private final JCommander commander;

    public UsageException(JCommander commander) {
        this.commander = commander;
    }

    public JCommander getCommander() {
        return commander;
    }
}
