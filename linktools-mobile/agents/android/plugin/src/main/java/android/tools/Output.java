package android.tools;

import android.util.Log;

import java.io.PrintStream;

/**
 * Created by hu on 18-12-19.
 */

public interface Output {

    Output out = new OutputImpl();
    Output err = new OutputImpl();

    class OutputImpl implements Output {

        private PrintStream printStream;

        @Override
        public PrintStream getStream() {
            return printStream;
        }

        @Override
        public Output setStream(PrintStream stream) {
            this.printStream = stream;
            return this;
        }

        @Override
        public Output println(String format, Object... args) {
            printStream.println(args.length > 0 ? String.format(format, args) : format);
            return this;
        }

        @Override
        public Output println(Object object) {
            printStream.println(object);
            return this;
        }

        @Override
        public Output println(Throwable th) {
            printStream.println(Log.getStackTraceString(th));
            return this;
        }

        @Override
        public Output println() {
            printStream.println();
            return this;
        }
    }

    PrintStream getStream();
    Output setStream(PrintStream stream);
    Output println(Object object);
    Output println(String format, Object... args);
    Output println(Throwable th);
    Output println();
}
