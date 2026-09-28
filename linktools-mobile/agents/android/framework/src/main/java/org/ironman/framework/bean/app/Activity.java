package org.ironman.framework.bean.app;

import android.content.pm.PackageParser;
import android.text.TextUtils;

@SuppressWarnings({"rawtypes", "unchecked"})
public class Activity extends Component {

    public Permission permission;

    public Activity(PackageParser.Activity activity) {
        super(activity, activity.info);
        if (!TextUtils.isEmpty(activity.info.permission)) {
            permission = new Permission(activity.info.permission);
        }
    }
}
