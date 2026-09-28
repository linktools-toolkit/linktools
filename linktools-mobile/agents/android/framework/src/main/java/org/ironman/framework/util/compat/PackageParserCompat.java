package org.ironman.framework.util.compat;

import android.content.pm.PackageParser;
import android.os.Build;
import android.util.DisplayMetrics;
import android.util.Singleton;

import org.ironman.framework.util.LogUtil;

import java.io.File;
import java.lang.reflect.Constructor;
import java.lang.reflect.Method;

import static android.os.Build.VERSION_CODES.LOLLIPOP;

public class PackageParserCompat {

    private static final String TAG = PackageParserCompat.class.getSimpleName();

    private static final Singleton<Constructor<PackageParser>> sConstructor = new Singleton<Constructor<PackageParser>>() {
        @Override
        protected Constructor<PackageParser> create() {
            try {
                if (Build.VERSION.SDK_INT >= LOLLIPOP) {
                    //noinspection JavaReflectionMemberAccess
                    return PackageParser.class.getConstructor();
                } else {
                    //noinspection JavaReflectionMemberAccess
                    return PackageParser.class.getConstructor(String.class);
                }
            } catch (NoSuchMethodException e) {
                LogUtil.printStackTrace(TAG, e, null);
            }

            return null;
        }
    };

    private static final Singleton<Method> sParsePackageMethod = new Singleton<Method>() {
        @Override
        protected Method create() {
            try {
                if (Build.VERSION.SDK_INT >= LOLLIPOP) {
                    //noinspection JavaReflectionMemberAccess
                    return PackageParser.class.getMethod("parsePackage", File.class, int.class);
                } else {
                    //noinspection JavaReflectionMemberAccess
                    return PackageParser.class.getMethod("parsePackage", File.class, String.class, DisplayMetrics.class, int.class);
                }
            } catch (NoSuchMethodException e) {
                LogUtil.printStackTrace(TAG, e, null);
            }

            return null;
        }
    };

    public static PackageParser createParser(File packageFile) throws Exception {
        if (Build.VERSION.SDK_INT >= LOLLIPOP) {
            return sConstructor.get().newInstance();
        } else {
            return sConstructor.get().newInstance(packageFile.getAbsolutePath());
        }
    }

    public static PackageParser.Package parsePackage(PackageParser parser, File packageFile, int flags) throws Exception {
        if (Build.VERSION.SDK_INT >= LOLLIPOP) {
            return (PackageParser.Package) sParsePackageMethod.get().invoke(parser, packageFile, flags);
        } else {
            return (PackageParser.Package) sParsePackageMethod.get().invoke(parser, packageFile, null, new DisplayMetrics(), flags);
        }
    }

    public static PackageParser.Package parsePackage(File packageFile, int flags) {
        try {
            return parsePackage(createParser(packageFile.getAbsoluteFile()), packageFile, flags);
        } catch (Exception e) {
            LogUtil.printStackTrace(TAG, e, null);
        }
        return null;
    }

}
