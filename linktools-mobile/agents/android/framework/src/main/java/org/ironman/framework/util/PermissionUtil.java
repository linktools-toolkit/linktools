package org.ironman.framework.util;

import android.content.pm.PackageManager;
import android.content.pm.PermissionInfo;
import android.text.TextUtils;

import org.ironman.framework.Environment;
import org.ironman.framework.bean.app.Permission;

/**
 * Created by hu on 18-12-17.
 */

public class PermissionUtil {

    private static final String TAG = PermissionUtil.class.getSimpleName();

    public static Permission.Protection getProtection(String permission) {
        if (!TextUtils.isEmpty(permission)) {
            try {
                return getProtection(Environment.getPackageManager().getPermissionInfo(permission, -1));
            } catch (PackageManager.NameNotFoundException e) {
                LogUtil.printStackTrace(TAG, e, null);
            }
        }
        return Permission.Protection.normal;
    }

    public static Permission.Protection getProtection(PermissionInfo permissionInfo) {
        switch (permissionInfo.protectionLevel & PermissionInfo.PROTECTION_MASK_BASE) {
            case PermissionInfo.PROTECTION_DANGEROUS:
                return Permission.Protection.dangerous;
            case PermissionInfo.PROTECTION_NORMAL:
                return Permission.Protection.normal;
            case PermissionInfo.PROTECTION_SIGNATURE:
                return Permission.Protection.signature;
            case PermissionInfo.PROTECTION_SIGNATURE_OR_SYSTEM:
                return Permission.Protection.signatureOrSystem;
            default:
                return Permission.Protection.normal;
        }
    }

    public static boolean isDangerousOrNormal(PermissionInfo permissionInfo) {
        switch (permissionInfo.protectionLevel & PermissionInfo.PROTECTION_MASK_BASE) {
            case PermissionInfo.PROTECTION_DANGEROUS:
                return true;
            case PermissionInfo.PROTECTION_NORMAL:
                return true;
            case PermissionInfo.PROTECTION_SIGNATURE:
                return false;
            case PermissionInfo.PROTECTION_SIGNATURE_OR_SYSTEM:
                return false;
            default:
                return true;
        }
    }

    public static boolean isDangerousOrNormal(String permission) {
        if (!TextUtils.isEmpty(permission)) {
            try {
                PackageManager pm = Environment.getPackageManager();
                return isDangerousOrNormal(pm.getPermissionInfo(permission, -1));
            } catch (PackageManager.NameNotFoundException e) {
                LogUtil.printStackTrace(TAG, e, null);
            }
        }
        return true;
    }
}
