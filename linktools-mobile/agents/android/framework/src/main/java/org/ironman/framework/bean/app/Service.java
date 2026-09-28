package org.ironman.framework.bean.app;

import android.content.pm.PackageParser;
import android.text.TextUtils;

@SuppressWarnings({"rawtypes", "unchecked"})
public class Service extends Component {

    public Permission permission;

    public Service(PackageParser.Service service) {
        super(service, service.info);
        if (!TextUtils.isEmpty(service.info.permission)) {
            permission = new Permission(service.info.permission);
        }
    }
}
