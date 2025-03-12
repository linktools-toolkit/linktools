package org.ironman.framework.bean.app;

import android.annotation.SuppressLint;
import android.content.ComponentName;
import android.content.pm.ComponentInfo;
import android.content.pm.PackageManager;
import android.content.pm.PackageParser;

import org.ironman.framework.Environment;

import java.util.ArrayList;
import java.util.List;

public class Component<II extends PackageParser.IntentInfo> {

    public String name;
    public boolean exported;
    public boolean enabled;
    public List<IntentFilter> intents;

    @SuppressLint("SwitchIntDef")
    public Component(PackageParser.Component<II> component, ComponentInfo info) {
        PackageManager packageManager = Environment.getPackageManager();
        ComponentName componentName = component.getComponentName();

        name = componentName.getClassName();
        exported = info.exported;

        switch (packageManager.getComponentEnabledSetting(componentName)) {
            case PackageManager.COMPONENT_ENABLED_STATE_DEFAULT:
                enabled = info.enabled;
                break;
            case PackageManager.COMPONENT_ENABLED_STATE_ENABLED:
                enabled = true;
                break;
            case PackageManager.COMPONENT_ENABLED_STATE_DISABLED:
            case PackageManager.COMPONENT_ENABLED_STATE_DISABLED_USER:
            case PackageManager.COMPONENT_ENABLED_STATE_DISABLED_UNTIL_USED:
                enabled = false;
                break;
        }

        if (component.intents != null && !component.intents.isEmpty()) {
            intents = new ArrayList<>(component.intents.size());
            for (II intent : component.intents) {
                intents.add(new IntentFilter(intent));
            }
        }
    }
}
