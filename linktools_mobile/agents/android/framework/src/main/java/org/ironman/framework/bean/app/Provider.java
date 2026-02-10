package org.ironman.framework.bean.app;

import android.content.pm.PackageParser;
import android.text.TextUtils;

import java.util.ArrayList;
import java.util.List;

@SuppressWarnings({"rawtypes", "unchecked"})
public class Provider extends Component {

    public String authority;
    public Permission readPermission;
    public Permission writePermission;
    public List<PatternMatcher> uriPermissionPatterns;
    public List<PathPermission> pathPermissions;

    public Provider(PackageParser.Provider provider) {
        super(provider, provider.info);

        authority = provider.info.authority;
        if (!TextUtils.isEmpty(provider.info.readPermission)) {
            readPermission = new Permission(provider.info.readPermission);
        }
        if (!TextUtils.isEmpty(provider.info.writePermission)) {
            writePermission = new Permission(provider.info.writePermission);
        }
        if (provider.info.uriPermissionPatterns != null) {
            uriPermissionPatterns = new ArrayList<>(provider.info.uriPermissionPatterns.length);
            for (android.os.PatternMatcher uriPermissionPattern : provider.info.uriPermissionPatterns) {
                uriPermissionPatterns.add(new PatternMatcher(uriPermissionPattern));
            }
        }
        if (provider.info.pathPermissions != null) {
            pathPermissions = new ArrayList<>(provider.info.pathPermissions.length);
            for (android.content.pm.PathPermission pathPermission : provider.info.pathPermissions) {
                pathPermissions.add(new PathPermission(pathPermission));
            }
        }
    }
}
