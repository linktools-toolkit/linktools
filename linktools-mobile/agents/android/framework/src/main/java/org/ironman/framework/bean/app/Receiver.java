package org.ironman.framework.bean.app;

import android.content.pm.PackageParser;
import android.text.TextUtils;

@SuppressWarnings({"rawtypes", "unchecked"})
public class Receiver extends Component {

    public Permission permission;

    public Receiver(PackageParser.Activity receiver) {
        super(receiver, receiver.info);
        if (!TextUtils.isEmpty(receiver.info.permission)) {
            permission = new Permission(receiver.info.permission);
        }
    }
}
