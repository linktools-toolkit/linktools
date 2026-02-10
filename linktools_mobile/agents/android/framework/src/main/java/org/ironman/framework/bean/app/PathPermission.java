package org.ironman.framework.bean.app;


public class PathPermission extends PatternMatcher {

    public Permission readPermission;
    public Permission writePermission;

    public PathPermission(android.content.pm.PathPermission pathPermission) {
        super(pathPermission);
        readPermission = new Permission(pathPermission.getReadPermission());
        writePermission = new Permission(pathPermission.getWritePermission());
    }
}