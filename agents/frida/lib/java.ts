/**
 *  用于方便调用frida的java方法
 *  https://github.com/frida/frida-java-bridge/blob/main/lib/class-factory.js
 * 
 *  frida class
 *  └┬─ $classWrapper
 *   │  └─ className
 *   ├─ $getClassHandle
 *   ├─ constructor
 *   ├─ $dispose
 *   └─ $isSameObject
 *
 *  method class
 *  └┬─ methodName
 *   ├─ holder
 *   │  └─ className
 *   ├─ type
 *   ├─ handle
 *   ├─ implementation
 *   ├─ returnType
 *   ├─ argumentTypes
 *   └─ canInvokeWith
 */

import * as Log from "./log"

type HookOpts = {
    method?: boolean;
    thread?: boolean;
    stack?: boolean;
    args?: boolean;
    result?: boolean;
    error?: boolean;
    page?: boolean;
    extras?: {
        [name: string]: any
    };
}
type HookImpl<T extends Java.Members<T>> = (obj: Java.Wrapper<T>, args: any[]) => any;

class Objects {

    excludeHookPackages: string[] = [
        "java.",
        "javax.",
        "android.",
        "androidx.",
    ]

    get objectClass(): Java.Wrapper {
        return Java.use("java.lang.Object");
    }

    get classClass(): Java.Wrapper {
        return Java.use("java.lang.Class");
    }

    get classLoaderClass(): Java.Wrapper {
        return Java.use("java.lang.ClassLoader");
    }

    get stringClass(): Java.Wrapper {
        return Java.use("java.lang.String");
    }

    get threadClass(): Java.Wrapper {
        return Java.use("java.lang.Thread");
    }

    get throwableClass(): Java.Wrapper {
        return Java.use("java.lang.Throwable");
    }

    get uriClass(): Java.Wrapper {
        return Java.use("android.net.Uri");
    }

    get urlClass(): Java.Wrapper {
        return Java.use("java.net.URL");
    }

    get mapClass(): Java.Wrapper {
        return Java.use("java.util.Map");
    }

    get hashSetClass(): Java.Wrapper {
        return Java.use("java.util.HashSet");
    }

    get applicationContext(): Java.Wrapper {
        const activityThreadClass = Java.use("android.app.ActivityThread");
        return activityThreadClass.currentApplication().getApplicationContext();
    }

    get currentActivity(): Java.Wrapper | null {
        try {
            const activityThreadClass = Java.use("android.app.ActivityThread");
            const activityClientRecordClass = Java.use("android.app.ActivityThread$ActivityClientRecord");
            const activityClientRecords = activityThreadClass.currentActivityThread().mActivities.value.values();
            const it = activityClientRecords.iterator();
            while (it.hasNext()) {
                const activityClientRecord = Java.cast(it.next(), activityClientRecordClass);
                if (!activityClientRecord.paused.value) {
                    return activityClientRecord.activity.value;
                }
            }
            return null;
        } catch (e) {
            return null;
        }
    }
}

export const o = new Objects();

/**
 * 判断两个对象是否为同一个java对象
 * @param obj1 对象1
 * @param obj2 对象2
 * @returns 是否相同
 */
export function isSameObject<T extends Java.Members<T> = {}>(obj1: Java.Wrapper<T>, obj2: Java.Wrapper<T>) {
    if (obj1 === obj2) {
        return true;
    } else if (obj1 == null || obj2 == null) {
        return false;
    } else if (obj1.hasOwnProperty("$isSameObject")) {
        return obj1.$isSameObject(obj2);
    }
    return false;
}

/**
 * 获取java对象的handle
 * @param obj java对象
 * @returns handle
 */
export function getObjectHandle<T extends Java.Members<T> = {}>(obj: Java.Wrapper<T>): NativePointer {
    if (obj == null) {
        return null;
    } else if (obj.hasOwnProperty("$h")) {
        return obj.$h;
    }
    return void 0;
}

/**
 * 获取类名
 * @param clazz 类对象
 * @returns 类名
 */
export function getClassName<T extends Java.Members<T> = {}>(clazz: Java.Wrapper<T>): string {
    var className = clazz.$className;
    if (className != void 0) {
        return className;
    }
    className = clazz.__name__;
    if (className != void 0) {
        return className;
    }
    if (clazz.$classWrapper != void 0) {
        className = clazz.$classWrapper.$className;
        if (className != void 0) {
            return className;
        }
        className = clazz.$classWrapper.__name__;
        if (className != void 0) {
            return className;
        }
    }
    Log.e("Cannot get class name: " + clazz);
}

/**
 * 获取真实方法对象
 * @param clazz 类对象
 * @param methodName 方法名
 * @returns 方法对象
 */
export function getClassMethod<T extends Java.Members<T> = {}>(clazz: Java.Wrapper<T>, methodName: string): Java.MethodDispatcher<T> {
    var method = clazz[methodName];
    if (method !== void 0) {
        return method;
    }
    if (methodName[0] == "$") {
        method = clazz["_" + methodName];
        if (method !== void 0) {
            return method;
        }
    }
    return void 0;
}

/**
 * 获取java类的类对象
 * @param className java类名
 * @param classloader java类所在的ClassLoader
 * @returns 类对象
 */
export function findClass<T extends Java.Members<T> = {}>(className: string, classloader: Java.Wrapper<{}> = void 0): Java.Wrapper<T> {
    if (classloader !== void 0 && classloader != null) {
        return Java.ClassFactory.get(classloader).use(className);
    } else {
        if (parseInt(Java.androidVersion) < 7) {
            return Java.use(className);
        }
        var error = null;
        var loaders = Java.enumerateClassLoadersSync();
        for (var loader of loaders) {
            try {
                var clazz = findClass<T>(className, loader);
                if (clazz != null) {
                    return clazz;
                }
            } catch (e) {
                if (error == null) {
                    error = e;
                }
            }
        }
        throw error;
    }
}

/**
 * hook指定方法对象
 * @param clazz java类名/类对象
 * @param method java方法名/方法对象
 * @param signatures java方法签名，为null表示不设置签名
 * @param impl hook实现，如调用原函数： function(obj, args) { return this(obj, args); }
 */
export function hookMethod<T extends Java.Members<T> = {}>(
    clazz: string | Java.Wrapper<T>,
    method: string | Java.Method<T>,
    signatures: (string | Java.Wrapper<T>)[],
    impl: HookImpl<T> | HookOpts = null
): void {
    var targetMethod: any = method;
    if (typeof (targetMethod) === "string") {
        var methodName = targetMethod;
        var targetClass: any = clazz;
        if (typeof (targetClass) === "string") {
            targetClass = findClass(targetClass);
        }
        const method = getClassMethod(targetClass, methodName);
        if (method === void 0 || method.overloads === void 0) {
            throw Error("Cannot find method: " + getClassName(targetClass) + "." + methodName);
        }
        if (signatures != null) {
            var targetSignatures: any[] = signatures;
            for (var i in targetSignatures) {
                if (typeof (targetSignatures[i]) !== "string") {
                    targetSignatures[i] = getClassName(targetSignatures[i]);
                }
            }
            targetMethod = method.overload.apply(method, targetSignatures);
        } else if (method.overloads.length == 1) {
            targetMethod = method.overloads[0];
        } else {
            throw Error(getClassName(targetClass) + "." + methodName + " has too many overloads");
        }
    }
    $defineMethodProperties(targetMethod);
    $hookMethod(targetMethod, impl);
}

/**
 * hook指定方法名的所有重载
 * @param clazz java类名/类对象
 * @param methodName java方法名
 * @param impl hook实现，如调用原函数： function(obj, args) { return this(obj, args); }
 */
export function hookMethods<T extends Java.Members<T> = {}>(
    clazz: string | Java.Wrapper<T>,
    methodName: string,
    impl: HookImpl<T> | HookOpts = null
): void {
    var targetClass: any = clazz;
    if (typeof (targetClass) === "string") {
        targetClass = findClass(targetClass);
    }
    var method = getClassMethod(targetClass, methodName);
    if (method === void 0 || method.overloads === void 0) {
        throw Error("Cannot find method: " + getClassName(targetClass) + "." + methodName);
    }
    for (var i = 0; i < method.overloads.length; i++) {
        const targetMethod = method.overloads[i];
        /* 过滤一些不存在的方法（拿不到返回值） */
        if (targetMethod.returnType !== void 0 &&
            targetMethod.returnType.className !== void 0) {
            $defineMethodProperties(targetMethod);
            $hookMethod(targetMethod, impl);
        }
    }
}

/**
 * hook指定类的所有构造方法
 * @param clazz java类名/类对象
 * @param impl hook实现，如调用原函数： function(obj, args) { return this(obj, args); }
 */
export function hookAllConstructors<T extends Java.Members<T> = {}>(
    clazz: string | Java.Wrapper<T>,
    impl: HookImpl<T> | HookOpts = null
): void {
    var targetClass: any = clazz;
    if (typeof (targetClass) === "string") {
        targetClass = findClass(targetClass);
    }
    hookMethods(targetClass, "$init", impl);
}

/**
 * hook指定类的所有成员方法
 * @param clazz java类名/类对象
 * @param impl hook实现，如调用原函数： function(obj, args) { return this(obj, args); }
 */
export function hookAllMethods<T extends Java.Members<T> = {}>(
    clazz: string | Java.Wrapper<T>,
    impl: HookImpl<T> | HookOpts = null
): void {
    var targetClass: any = clazz;
    if (typeof (targetClass) === "string") {
        targetClass = findClass(targetClass);
    }
    var methodNames = [];
    var superJavaClass = null;
    var targetJavaClass = targetClass.class;
    while (targetJavaClass != null) {
        var methods = targetJavaClass.getDeclaredMethods();
        for (let i = 0; i < methods.length; i++) {
            const method = methods[i];
            var methodName = method.getName();
            if (methodNames.indexOf(methodName) < 0) {
                methodNames.push(methodName);
                hookMethods(targetClass, methodName, impl);
            }
        }
        superJavaClass = targetJavaClass.getSuperclass();
        targetJavaClass.$dispose();
        if (superJavaClass == null) {
            // 不知道为啥，com.android.org.bouncycastle.crypto.paddings.BlockCipherPadding这个类获取superclass的时候会返回null
            break;
        }
        targetJavaClass = Java.cast(superJavaClass, o.classClass);
        if ($isExcludeClass(targetJavaClass.getName())) {
            break;
        }
    }
}

/**
 * hook指定类的所有方法（构造、成员方法）
 * @param clazz java类名/类对象
 * @param impl hook实现，如调用原函数： function(obj, args) { return this(obj, args); }
 */
export function hookClass<T extends Java.Members<T> = {}>(
    clazz: string | Java.Wrapper<T>,
    impl: HookImpl<T> | HookOpts = null
): void {
    var targetClass: any = clazz;
    if (typeof (targetClass) === "string") {
        targetClass = findClass(targetClass);
    }
    hookAllConstructors(targetClass, impl);
    hookAllMethods(targetClass, impl);
}

/**
 * 获取hook实现，调用原方法并发送调用事件
 * @param options hook选项，如：{stack: true, args: true, thread: true}
 * @returns hook实现
 */
export function getEventImpl<T extends Java.Members<T> = {}>(options: HookOpts): HookImpl<T> {
    const hookOpts: HookOpts = {};
    hookOpts.method = parseBoolean(options.method, true);
    hookOpts.thread = parseBoolean(options.thread, false);
    hookOpts.stack = parseBoolean(options.stack, false);
    hookOpts.args = parseBoolean(options.args, false);
    hookOpts.result = parseBoolean(options.result, hookOpts.args);
    hookOpts.error = parseBoolean(options.error, hookOpts.args);
    hookOpts.page = parseBoolean(options.page, false);
    hookOpts.extras = {};
    if (options.extras != null) {
        for (let i in options.extras) {
            hookOpts.extras[i] = options.extras[i];
        }
    }

    return function (obj, args) {
        const event = {};
        for (const key in hookOpts.extras) {
            event[key] = hookOpts.extras[key];
        }
        if (hookOpts.method !== false) {
            event["class_name"] = obj.$className;
            event["method_name"] = this.name;
            event["method_simple_name"] = this.methodName;
        }
        if (hookOpts.thread !== false) {
            event["thread_id"] = Process.getCurrentThreadId();
            event["thread_name"] = o.threadClass.currentThread().getName();
        }
        if (hookOpts.args !== false) {
            event["args"] = pretty2Json(args);
        }
        if (hookOpts.result !== false) {
            event["result"] = null;
        }
        if (hookOpts.error !== false) {
            event["error"] = null;
        }
        if (hookOpts.page !== false) {
            const activity = o.currentActivity;
            event["page"] = activity ? activity.$className : null;
        }
        try {
            const result = this(obj, args);
            if (hookOpts.result !== false) {
                event["result"] = pretty2Json(result);
            }
            return result;
        } catch (e) {
            if (hookOpts.error !== false) {
                event["error"] = pretty2Json(e);
            }
            throw e;
        } finally {
            if (hookOpts.stack !== false) {
                event["stack"] = pretty2Json(getStackTrace());
            }
            Log.event(event);
        }
    };
}

/**
 * 判断对象是不是java对象
 * @param obj js对象
 * @returns obj为java对象，则返回为true，否则为false
 */
export function isJavaObject(obj: any): boolean {
    if (obj instanceof Object) {
        if (obj.hasOwnProperty("class") && obj.class instanceof Object) {
            const javaClass = obj.class;
            if (javaClass.hasOwnProperty("getName") &&
                javaClass.hasOwnProperty("getDeclaredClasses") &&
                javaClass.hasOwnProperty("getDeclaredFields") &&
                javaClass.hasOwnProperty("getDeclaredMethods")) {
                return true;
            }
        }
    }
    return false;
}

/**
 * 判断对象是否为java数组
 * @param obj js对象
 * @returns obj为java数组，则返回为true，否则为false
 */
export function isJavaArray(obj: any): boolean {
    if (obj instanceof Object) {
        if (obj.hasOwnProperty("class") && obj.class instanceof Object) {
            const javaClass = obj.class;
            if (javaClass.hasOwnProperty("isArray") && javaClass.isArray()) {
                return true;
            }
        }
    }
    return false;
}

/**
 * java数组转为js数组
 * @param clazz java类名/类对象
 * @param array java数组
 * @returns js数组
 */
export function fromJavaArray<T extends Java.Members<T> = {}>(
    clazz: string | Java.Wrapper<T>,
    array: Java.Wrapper<T>
): Java.Wrapper<T>[] {
    var targetClass: any = clazz;
    if (typeof (targetClass) === "string") {
        targetClass = findClass(targetClass);
    }
    var result = [];
    var env = Java.vm.getEnv();
    for (var i = 0; i < env.getArrayLength(array.$handle); i++) {
        result.push(Java.cast(env.getObjectArrayElement(array.$handle, i), targetClass))
    }
    return result;
}

/**
 * 获取枚举值
 * @param clazz java类名/类对象
 * @param name java枚举名称
 * @returns java枚举值
 */
export function getJavaEnumValue<T extends Java.Members<T> = {}>(
    clazz: string | Java.Wrapper<T>,
    name: string
): Java.Wrapper<T> {
    var targetClass: any = clazz;
    if (typeof (targetClass) === "string") {
        targetClass = findClass(targetClass);
    }
    var values = targetClass.class.getEnumConstants();
    if (!(values instanceof Array)) {
        values = fromJavaArray(targetClass, values);
    }
    for (var i = 0; i < values.length; i++) {
        if (values[i].toString() === name) {
            return values[i];
        }
    }
    throw new Error("Name of " + name + " does not match " + targetClass);
}

/**
 * 获取当前java栈
 * @returns java栈对象
 */
export function getStackTrace<T extends Java.Members<T> = {}>(th: Java.Wrapper<T> = void 0): Java.Wrapper<T>[] {
    const result = [];
    const elements = (th || o.throwableClass.$new()).getStackTrace();
    for (let i = 0; i < elements.length; i++) {
        result.push(elements[i]);
    }
    return result;
}

type UseClassCallback = (clazz: Java.Wrapper<{}>) => void;
type UseClassCallbackSet = Set<UseClassCallback>;
let $useClassCallbackMap: Map<string, UseClassCallbackSet> = null;

function $registerUseClassCallback(map: Map<string, UseClassCallbackSet>) {
    const classLoaders = o.hashSetClass.$new();

    const tryLoadClasses = function (classLoader: Java.Wrapper<{}>) {
        let it = map.entries();
        let result: IteratorResult<[string, UseClassCallbackSet]>;
        while (result = it.next(), !result.done) {
            const name = result.value[0];
            const callbacks = result.value[1];
            let clazz = null;
            try {
                clazz = findClass(name, classLoader);
            } catch (e) {
                // ignore
            }
            if (clazz != null) {
                map.delete(name);
                callbacks.forEach(function (callback, _sameCallback, _set) {
                    try {
                        callback(clazz);
                    } catch (e) {
                        Log.w("Call JavaHelper.use callback error: " + e);
                    }
                });
            }
        }
    }

    const classClass = o.classClass;
    const classLoaderClass = o.classLoaderClass;

    hookMethod(
        classClass,
        "forName",
        ["java.lang.String", "boolean", classLoaderClass],
        function (obj, args) {
            const classLoader = args[2];
            if (classLoader != null && !classLoaders.contains(classLoader)) {
                classLoaders.add(classLoader);
                tryLoadClasses(classLoader);
            }
            return this(obj, args);
        }
    );

    hookMethod(
        classLoaderClass,
        "loadClass",
        ["java.lang.String", "boolean"],
        function (obj, args) {
            const classLoader = obj;
            if (!classLoaders.contains(classLoader)) {
                classLoaders.add(classLoader);
                tryLoadClasses(classLoader);
            }
            return this(obj, args);
        }
    );
}

/**
 * 类似Java.use，如果当前未加载目标类名，则会监控classloader，直到加载目标类名为止
 * @param className 类名
 * @param callback 加载目标类时的回调
 */
export function use(className: string, callback: UseClassCallback) {
    let targetClass: Java.Wrapper<{}> = null;
    try {
        targetClass = findClass(className);
    } catch (e) {
        if ($useClassCallbackMap == null) {
            $useClassCallbackMap = new Map<string, UseClassCallbackSet>();
            $registerUseClassCallback($useClassCallbackMap);
        }
        if ($useClassCallbackMap.has(className)) {
            let callbackSet = $useClassCallbackMap.get(className);
            if (callbackSet !== void 0) {
                callbackSet.add(callback);
            }
        } else {
            let callbackSet = new Set<UseClassCallback>();
            callbackSet.add(callback);
            $useClassCallbackMap.set(className, callbackSet);
        }
        return;
    }
    try {
        callback(targetClass);
    } catch (e) {
        Log.w("Call JavaHelper.use callback error: " + e);
    }
}


/**
 * 开启webview debug
 */
export function setWebviewDebuggingEnabled() {

    Log.w("Android Enable Webview Debugging");

    ignoreError(() => {
        let WebView = findClass("android.webkit.WebView");
        hookMethods(WebView, "setWebContentsDebuggingEnabled", function (obj, args) {
            Log.d(`${WebView}.setWebContentsDebuggingEnabled: ${args[0]}`);
            args[0] = true;
            return this(obj, args);
        })
        hookMethods(WebView, "loadUrl", function (obj, args) {
            Log.d(`${WebView}.loadUrl: ${args[0]}`);
            WebView.setWebContentsDebuggingEnabled(true);
            return this(obj, args);
        })
    });

    ignoreError(() => {
        let UCWebView = findClass("com.uc.webview.export.WebView");
        hookMethods(UCWebView, "setWebContentsDebuggingEnabled", function (obj, args) {
            Log.d(`${UCWebView}.setWebContentsDebuggingEnabled: ${args[0]}`);
            args[0] = true;
            return this(obj, args);
        })
        hookMethods(UCWebView, "loadUrl", function (obj, args) {
            Log.d(`${UCWebView}.loadUrl: ${args[0]}`);
            UCWebView.setWebContentsDebuggingEnabled(true);
            return this(obj, args);
        })
    });
}

/**
 * 绕过ssl pinning
 */
export function bypassSslPinning() {

    Log.w("Android Bypass ssl pinning");

    const arraysClass = Java.use("java.util.Arrays");

    ignoreError(() => hookMethods(
        "com.android.org.conscrypt.TrustManagerImpl",
        "checkServerTrusted",
        function (obj, args) {
            Log.d('SSL bypassing ' + this);
            if (this.returnType.type == 'void') {
                return;
            } else if (this.returnType.type == "pointer" && this.returnType.className == "java.util.List") {
                return arraysClass.asList(args[0]);
            }
        })
    );

    ignoreError(() => hookMethods(
        "com.google.android.gms.org.conscrypt.Platform",
        "checkServerTrusted",
        function (obj, args) {
            Log.d('SSL bypassing ' + this);
        })
    );

    ignoreError(() => hookMethods(
        "com.android.org.conscrypt.Platform",
        "checkServerTrusted",
        function (obj, args) {
            Log.d('SSL bypassing ' + this);
        })
    );

    ignoreError(() => hookMethods(
        "okhttp3.CertificatePinner",
        "check",
        function (obj, args) {
            Log.d('SSL bypassing ' + this);
            if (this.returnType.type == "boolean") {
                return true;
            }
        })
    );

    ignoreError(() => hookMethods(
        "okhttp3.CertificatePinner",
        "check$okhttp",
        function (obj, args) {
            Log.d('SSL bypassing ' + this);
        })
    );

    ignoreError(() => hookMethods(
        "com.android.okhttp.CertificatePinner",
        "check",
        function (obj, args) {
            Log.d('SSL bypassing ' + this);
            if (this.returnType.type == "boolean") {
                return true;
            }
        })
    );

    ignoreError(() => hookMethods(
        "com.android.okhttp.CertificatePinner",
        "check$okhttp",
        function (obj, args) {
            Log.d('SSL bypassing ' + this);
            return void 0;
        })
    );

    ignoreError(() => hookMethods(
        "com.android.org.conscrypt.TrustManagerImpl",
        "verifyChain",
        function (obj, args) {
            Log.d('SSL bypassing ' + this);
            return args[0];
        })
    );
}

export function chooseClassLoader(className) {
    Log.w("choose classloder: " + className);

    Java.enumerateClassLoaders({
        onMatch: function (loader) {
            try {
                const clazz = loader.findClass(className);
                if (clazz != null) {
                    Log.i("choose classloader: " + loader);
                    Reflect.set(Java.classFactory, "loader", loader);
                }
            } catch (e) {
                Log.e(pretty2Json(e));
            }
        }, onComplete: function () {
            Log.d("enumerate classLoaders complete");
        }
    });
}

export function traceClasses(include: string, exclude: string = void 0, options: any = void 0) {

    include = include != null ? include.trim().toLowerCase() : "";
    exclude = exclude != null ? exclude.trim().toLowerCase() : "";
    options = options != null ? options : { stack: true, args: true };

    Log.w("trace classes, include: " + include + ", exclude: " + exclude + ", options: " + JSON.stringify(options));

    Java.enumerateLoadedClasses({
        onMatch: function (className) {
            const targetClassName: string = className.toString().toLowerCase();
            if (targetClassName.indexOf(include) >= 0) {
                if (exclude == "" || targetClassName.indexOf(exclude) < 0) {
                    hookAllMethods(className, getEventImpl(options));
                }
            }
        }, onComplete: function () {
            Log.d("enumerate classLoaders complete");
        }
    });
}

export function runOnCreateContext(fn: (context: Java.Wrapper<{}>) => any) {
    hookMethods("android.app.ContextImpl", "createAppContext", function (obj, args) {
        const context = this(obj, args);
        fn(context);
        return context;
    });
}

export function runOnCreateApplication(fn: (application: Java.Wrapper<{}>) => any) {
    hookMethods("android.app.LoadedApk", "makeApplication", function (obj, args) {
        const app = this(obj, args);
        fn(app);
        return app;
    });
}

function $prettyClassName(className: string) {
    if (className.startsWith("[L") && className.endsWith(";")) {
        return `${className.substring(2, className.length - 1)}[]`;
    } else if (className.startsWith("[")) {
        switch (className.substring(1, 2)) {
            case "B": return "byte[]";
            case "C": return "char[]";
            case "D": return "double[]";
            case "F": return "float[]";
            case "I": return "int[]";
            case "S": return "short[]";
            case "J": return "long[]";
            case "Z": return "boolean[]";
            case "V": return "void[]";
        }
    }
    return className;
}

/**
 * 为method添加properties
 * @param method 方法对象
 */
function $defineMethodProperties<T extends Java.Members<T> = {}>(method: Java.Method<T>): void {

    Object.defineProperties(method, {
        className: {
            configurable: true,
            enumerable: true,
            writable: false,
            value: getClassName(method.holder)
        },
        name: {
            configurable: true,
            enumerable: true,
            get() {
                const ret = $prettyClassName(this.returnType.className);
                const name = $prettyClassName(this.className) + "." + this.methodName;
                let args = "";
                if (this.argumentTypes.length > 0) {
                    args = $prettyClassName(this.argumentTypes[0].className);
                    for (let i = 1; i < this.argumentTypes.length; i++) {
                        args = args + ", " + $prettyClassName(this.argumentTypes[i].className);
                    }
                }
                return ret + " " + name + "(" + args + ")";
            }
        },
        toString: {
            configurable: true,
            value: function () {
                return this.name;
            }
        }
    });
}

/**
 * hook指定方法对象
 * @param method 方法对象
 * @param impl hook实现，如调用原函数： function(obj, args) { return this(obj, args); }
 */
function $hookMethod<T extends Java.Members<T> = {}>(
    method: Java.Method<T>,
    impl: HookImpl<T> | HookOpts = null
): void {
    if (impl != null) {
        const proxy = new Proxy(method, {
            apply: function (target, thisArg: any, argArray: any[]) {
                const obj = argArray[0];
                const args = argArray[1];
                return target.apply(obj, args);
            }
        });
        const hookImpl = isFunction(impl) ? impl as HookImpl<T> : getEventImpl(impl as HookOpts);
        method.implementation = function () {
            return hookImpl.call(proxy, this, Array.prototype.slice.call(arguments));
        };
        Log.i("Hook method: " + method);
    } else {
        method.implementation = null;
        Log.i("Unhook method: " + method);
    }
}

function $isExcludeClass(className: string) {
    for (const i in o.excludeHookPackages) {
        if (className.indexOf(o.excludeHookPackages[i]) == 0) {
            return true;
        }
    }
    return false;
}

export function getErrorStack(error: any) {
    try {
        const handle = getObjectHandle(error);
        if (handle !== void 0) {
            const throwable = Java.cast(handle, o.throwableClass);
            let items = [];
            for (let item of getStackTrace(throwable)) {
                items.push(`    at ${item}`);
            }
            return items.length > 0 ? `${throwable}\n${items.join("\n")}` : `${throwable}`;
        }
    } catch (e) {
        Log.d(`getErrorStack error: ${e}`);
    }
    return void 0;
}
