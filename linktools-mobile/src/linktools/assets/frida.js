"use strict";
(() => {
  var __defProp = Object.defineProperty;
  var __export = (target, all) => {
    for (var name in all)
      __defProp(target, name, { get: all[name], enumerable: true });
  };

  // lib/log.ts
  var log_exports = {};
  __export(log_exports, {
    DEBUG: () => DEBUG,
    ERROR: () => ERROR,
    INFO: () => INFO,
    WARNING: () => WARNING,
    d: () => d,
    e: () => e,
    event: () => event,
    exception: () => exception,
    getLevel: () => getLevel,
    i: () => i,
    setLevel: () => setLevel,
    w: () => w
  });
  var DEBUG = 1;
  var INFO = 2;
  var WARNING = 3;
  var ERROR = 4;
  var $level = INFO;
  var $pendingEvents = [];
  var $flushTimer = null;
  function getLevel() {
    return $level;
  }
  function setLevel(level) {
    $level = level;
    d("Set log level: " + level);
  }
  function d(message, data) {
    if ($level <= DEBUG) {
      $send("log", { level: "debug", message }, data);
    }
  }
  function i(message, data) {
    if ($level <= INFO) {
      $send("log", { level: "info", message }, data);
    }
  }
  function w(message, data) {
    if ($level <= WARNING) {
      $send("log", { level: "warning", message }, data);
    }
  }
  function e(message, data) {
    if ($level <= ERROR) {
      $send("log", { level: "error", message }, data);
    }
  }
  function event(message, data) {
    $send("msg", message, data);
  }
  function exception(description, stack) {
    $send("error", { description, stack });
  }
  function $send(type, message, data) {
    const event2 = {};
    event2[type] = message;
    if (data == null) {
      $pendingEvents.push(event2);
      if ($pendingEvents.length >= 50) {
        $flush();
      } else if ($flushTimer === null) {
        $flushTimer = setTimeout($flush, 50);
      }
    } else {
      $flush();
      send({ $events: [event2] }, data);
    }
  }
  function $flush() {
    if ($flushTimer !== null) {
      clearTimeout($flushTimer);
      $flushTimer = null;
    }
    if ($pendingEvents.length === 0) {
      return;
    }
    const events = $pendingEvents;
    $pendingEvents = [];
    send({ $events: events });
  }

  // lib/c.ts
  var c_exports = {};
  __export(c_exports, {
    getDebugSymbolFromAddress: () => getDebugSymbolFromAddress,
    getDescFromAddress: () => getDescFromAddress,
    getEventImpl: () => getEventImpl3,
    getExportFunction: () => getExportFunction,
    hookFunction: () => hookFunction,
    hookFunctionWithCallbacks: () => hookFunctionWithCallbacks,
    hookFunctionWithOptions: () => hookFunctionWithOptions,
    o: () => o3
  });

  // lib/java.ts
  var java_exports = {};
  __export(java_exports, {
    bypassSslPinning: () => bypassSslPinning,
    chooseClassLoader: () => chooseClassLoader,
    findClass: () => findClass,
    fromJavaArray: () => fromJavaArray,
    getClassMethod: () => getClassMethod,
    getClassName: () => getClassName,
    getErrorStack: () => getErrorStack,
    getEventImpl: () => getEventImpl,
    getJavaEnumValue: () => getJavaEnumValue,
    getObjectHandle: () => getObjectHandle,
    getStackTrace: () => getStackTrace,
    hookAllConstructors: () => hookAllConstructors,
    hookAllMethods: () => hookAllMethods,
    hookClass: () => hookClass,
    hookMethod: () => hookMethod,
    hookMethods: () => hookMethods,
    isJavaArray: () => isJavaArray,
    isJavaObject: () => isJavaObject,
    isSameObject: () => isSameObject,
    o: () => o,
    runOnCreateApplication: () => runOnCreateApplication,
    runOnCreateContext: () => runOnCreateContext,
    setWebviewDebuggingEnabled: () => setWebviewDebuggingEnabled,
    traceClasses: () => traceClasses,
    use: () => use
  });
  var Objects = class {
    excludeHookPackages = [
      "java.",
      "javax.",
      "android.",
      "androidx."
    ];
    get objectClass() {
      return Java.use("java.lang.Object");
    }
    get classClass() {
      return Java.use("java.lang.Class");
    }
    get classLoaderClass() {
      return Java.use("java.lang.ClassLoader");
    }
    get stringClass() {
      return Java.use("java.lang.String");
    }
    get threadClass() {
      return Java.use("java.lang.Thread");
    }
    get throwableClass() {
      return Java.use("java.lang.Throwable");
    }
    get uriClass() {
      return Java.use("android.net.Uri");
    }
    get urlClass() {
      return Java.use("java.net.URL");
    }
    get mapClass() {
      return Java.use("java.util.Map");
    }
    get hashSetClass() {
      return Java.use("java.util.HashSet");
    }
    get applicationContext() {
      const activityThreadClass = Java.use("android.app.ActivityThread");
      return activityThreadClass.currentApplication().getApplicationContext();
    }
    get currentActivity() {
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
      } catch (e2) {
        return null;
      }
    }
  };
  var o = new Objects();
  function isSameObject(obj1, obj2) {
    if (obj1 === obj2) {
      return true;
    } else if (obj1 == null || obj2 == null) {
      return false;
    } else if (obj1.hasOwnProperty("$isSameObject")) {
      return obj1.$isSameObject(obj2);
    }
    return false;
  }
  function getObjectHandle(obj) {
    if (obj == null) {
      return void 0;
    } else if (obj.hasOwnProperty("$h")) {
      return obj.$h;
    }
    return void 0;
  }
  function getClassName(clazz) {
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
    e("Cannot get class name: " + clazz);
    return void 0;
  }
  function getClassMethod(clazz, methodName) {
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
  function findClass(className, classloader = void 0) {
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
          var clazz = findClass(className, loader);
          if (clazz != null) {
            return clazz;
          }
        } catch (e2) {
          if (error == null) {
            error = e2;
          }
        }
      }
      throw error;
    }
  }
  function hookMethod(clazz, method, signatures, impl = void 0) {
    var targetMethod = method;
    if (typeof targetMethod === "string") {
      var methodName = targetMethod;
      var targetClass = clazz;
      if (typeof targetClass === "string") {
        targetClass = findClass(targetClass);
      }
      const method2 = getClassMethod(targetClass, methodName);
      if (method2 === void 0 || method2.overloads === void 0) {
        throw Error("Cannot find method: " + getClassName(targetClass) + "." + methodName);
      }
      if (signatures != null) {
        var targetSignatures = signatures;
        for (var i2 in targetSignatures) {
          if (typeof targetSignatures[i2] !== "string") {
            targetSignatures[i2] = getClassName(targetSignatures[i2]);
          }
        }
        targetMethod = method2.overload.apply(method2, targetSignatures);
      } else if (method2.overloads.length == 1) {
        targetMethod = method2.overloads[0];
      } else {
        throw Error(getClassName(targetClass) + "." + methodName + " has too many overloads");
      }
    }
    $defineMethodProperties(targetMethod);
    $hookMethod(targetMethod, impl);
  }
  function hookMethods(clazz, methodName, impl = void 0) {
    var targetClass = clazz;
    if (typeof targetClass === "string") {
      targetClass = findClass(targetClass);
    }
    var method = getClassMethod(targetClass, methodName);
    if (method === void 0 || method.overloads === void 0) {
      throw Error("Cannot find method: " + getClassName(targetClass) + "." + methodName);
    }
    for (var i2 = 0; i2 < method.overloads.length; i2++) {
      const targetMethod = method.overloads[i2];
      if (targetMethod.returnType !== void 0 && targetMethod.returnType.className !== void 0) {
        $defineMethodProperties(targetMethod);
        $hookMethod(targetMethod, impl);
      }
    }
  }
  function hookAllConstructors(clazz, impl = void 0) {
    var targetClass = clazz;
    if (typeof targetClass === "string") {
      targetClass = findClass(targetClass);
    }
    hookMethods(targetClass, "$init", impl);
  }
  function hookAllMethods(clazz, impl = void 0) {
    var targetClass = clazz;
    if (typeof targetClass === "string") {
      targetClass = findClass(targetClass);
    }
    var methodNames = [];
    var superJavaClass = null;
    var targetJavaClass = targetClass.class;
    while (targetJavaClass != null) {
      var methods = targetJavaClass.getDeclaredMethods();
      for (let i2 = 0; i2 < methods.length; i2++) {
        const method = methods[i2];
        var methodName = method.getName();
        if (methodNames.indexOf(methodName) < 0) {
          methodNames.push(methodName);
          hookMethods(targetClass, methodName, impl);
        }
      }
      superJavaClass = targetJavaClass.getSuperclass();
      targetJavaClass.$dispose();
      if (superJavaClass == null) {
        break;
      }
      targetJavaClass = Java.cast(superJavaClass, o.classClass);
      if ($isExcludeClass(targetJavaClass.getName())) {
        break;
      }
    }
  }
  function hookClass(clazz, impl = void 0) {
    var targetClass = clazz;
    if (typeof targetClass === "string") {
      targetClass = findClass(targetClass);
    }
    hookAllConstructors(targetClass, impl);
    hookAllMethods(targetClass, impl);
  }
  function getEventImpl(options) {
    const hookOpts = {};
    hookOpts.method = parseBoolean(options.method, true);
    hookOpts.thread = parseBoolean(options.thread, false);
    hookOpts.stack = parseBoolean(options.stack, false);
    hookOpts.args = parseBoolean(options.args, false);
    hookOpts.result = parseBoolean(options.result, hookOpts.args);
    hookOpts.error = parseBoolean(options.error, hookOpts.args);
    hookOpts.page = parseBoolean(options.page, false);
    hookOpts.extras = {};
    if (options.extras != null) {
      for (let i2 in options.extras) {
        hookOpts.extras[i2] = options.extras[i2];
      }
    }
    return function(obj, args) {
      const event2 = {};
      for (const key in hookOpts.extras) {
        event2[key] = hookOpts.extras[key];
      }
      if (hookOpts.method !== false) {
        event2["class_name"] = obj.$className;
        event2["method_name"] = this.name;
        event2["method_simple_name"] = this.methodName;
      }
      if (hookOpts.thread !== false) {
        event2["thread_id"] = Process.getCurrentThreadId();
        event2["thread_name"] = o.threadClass.currentThread().getName();
      }
      if (hookOpts.args !== false) {
        event2["args"] = pretty2Json(args);
      }
      if (hookOpts.result !== false) {
        event2["result"] = null;
      }
      if (hookOpts.error !== false) {
        event2["error"] = null;
      }
      if (hookOpts.page !== false) {
        const activity = o.currentActivity;
        event2["page"] = activity ? activity.$className : null;
      }
      try {
        const result = this(obj, args);
        if (hookOpts.result !== false) {
          event2["result"] = pretty2Json(result);
        }
        return result;
      } catch (e2) {
        if (hookOpts.error !== false) {
          event2["error"] = pretty2Json(e2);
        }
        throw e2;
      } finally {
        if (hookOpts.stack !== false) {
          event2["stack"] = pretty2Json(getStackTrace());
        }
        event(event2);
      }
    };
  }
  function isJavaObject(obj) {
    if (obj instanceof Object) {
      if (obj.hasOwnProperty("class") && obj.class instanceof Object) {
        const javaClass = obj.class;
        if (javaClass.hasOwnProperty("getName") && javaClass.hasOwnProperty("getDeclaredClasses") && javaClass.hasOwnProperty("getDeclaredFields") && javaClass.hasOwnProperty("getDeclaredMethods")) {
          return true;
        }
      }
    }
    return false;
  }
  function isJavaArray(obj) {
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
  function fromJavaArray(clazz, array) {
    var targetClass = clazz;
    if (typeof targetClass === "string") {
      targetClass = findClass(targetClass);
    }
    var result = [];
    var env = Java.vm.getEnv();
    for (var i2 = 0; i2 < env.getArrayLength(array.$handle); i2++) {
      result.push(Java.cast(env.getObjectArrayElement(array.$handle, i2), targetClass));
    }
    return result;
  }
  function getJavaEnumValue(clazz, name) {
    var targetClass = clazz;
    if (typeof targetClass === "string") {
      targetClass = findClass(targetClass);
    }
    var values = targetClass.class.getEnumConstants();
    if (!(values instanceof Array)) {
      values = fromJavaArray(targetClass, values);
    }
    for (var i2 = 0; i2 < values.length; i2++) {
      if (values[i2].toString() === name) {
        return values[i2];
      }
    }
    throw new Error("Name of " + name + " does not match " + targetClass);
  }
  function getStackTrace(th = void 0) {
    const result = [];
    const elements = (th || o.throwableClass.$new()).getStackTrace();
    for (let i2 = 0; i2 < elements.length; i2++) {
      result.push(elements[i2]);
    }
    return result;
  }
  var $useClassCallbackMap = null;
  function $registerUseClassCallback(map) {
    const classLoaders = o.hashSetClass.$new();
    const tryLoadClasses = function(classLoader) {
      let it = map.entries();
      let result;
      while (result = it.next(), !result.done) {
        const name = result.value[0];
        const callbacks = result.value[1];
        let clazz = null;
        try {
          clazz = findClass(name, classLoader);
        } catch (e2) {
        }
        if (clazz != null) {
          map.delete(name);
          callbacks.forEach(function(callback, _sameCallback, _set) {
            try {
              callback(clazz);
            } catch (e2) {
              w("Call JavaHelper.use callback error: " + e2);
            }
          });
        }
      }
    };
    const classClass = o.classClass;
    const classLoaderClass = o.classLoaderClass;
    hookMethod(classClass, "forName", ["java.lang.String", "boolean", classLoaderClass], function(obj, args) {
      const classLoader = args[2];
      if (classLoader != null && !classLoaders.contains(classLoader)) {
        classLoaders.add(classLoader);
        tryLoadClasses(classLoader);
      }
      return this(obj, args);
    });
    hookMethod(classLoaderClass, "loadClass", ["java.lang.String", "boolean"], function(obj, args) {
      const classLoader = obj;
      if (!classLoaders.contains(classLoader)) {
        classLoaders.add(classLoader);
        tryLoadClasses(classLoader);
      }
      return this(obj, args);
    });
  }
  function use(className, callback) {
    let targetClass = null;
    try {
      targetClass = findClass(className);
    } catch (e2) {
      if ($useClassCallbackMap == null) {
        $useClassCallbackMap = /* @__PURE__ */ new Map();
        $registerUseClassCallback($useClassCallbackMap);
      }
      if ($useClassCallbackMap.has(className)) {
        let callbackSet = $useClassCallbackMap.get(className);
        if (callbackSet !== void 0) {
          callbackSet.add(callback);
        }
      } else {
        let callbackSet = /* @__PURE__ */ new Set();
        callbackSet.add(callback);
        $useClassCallbackMap.set(className, callbackSet);
      }
      return;
    }
    try {
      callback(targetClass);
    } catch (e2) {
      w("Call JavaHelper.use callback error: " + e2);
    }
  }
  function setWebviewDebuggingEnabled() {
    w("Android Enable Webview Debugging");
    ignoreError(() => {
      let WebView = findClass("android.webkit.WebView");
      hookMethods(WebView, "setWebContentsDebuggingEnabled", function(obj, args) {
        d(`${WebView}.setWebContentsDebuggingEnabled: ${args[0]}`);
        args[0] = true;
        return this(obj, args);
      });
      hookMethods(WebView, "loadUrl", function(obj, args) {
        d(`${WebView}.loadUrl: ${args[0]}`);
        WebView.setWebContentsDebuggingEnabled(true);
        return this(obj, args);
      });
    });
    ignoreError(() => {
      let UCWebView = findClass("com.uc.webview.export.WebView");
      hookMethods(UCWebView, "setWebContentsDebuggingEnabled", function(obj, args) {
        d(`${UCWebView}.setWebContentsDebuggingEnabled: ${args[0]}`);
        args[0] = true;
        return this(obj, args);
      });
      hookMethods(UCWebView, "loadUrl", function(obj, args) {
        d(`${UCWebView}.loadUrl: ${args[0]}`);
        UCWebView.setWebContentsDebuggingEnabled(true);
        return this(obj, args);
      });
    });
  }
  function bypassSslPinning() {
    w("Android Bypass ssl pinning");
    const arraysClass = Java.use("java.util.Arrays");
    ignoreError(() => hookMethods("com.android.org.conscrypt.TrustManagerImpl", "checkServerTrusted", function(obj, args) {
      d("SSL bypassing " + this);
      if (this.returnType.type == "void") {
        return;
      } else if (this.returnType.type == "pointer" && this.returnType.className == "java.util.List") {
        return arraysClass.asList(args[0]);
      }
    }));
    ignoreError(() => hookMethods("com.google.android.gms.org.conscrypt.Platform", "checkServerTrusted", function(obj, args) {
      d("SSL bypassing " + this);
    }));
    ignoreError(() => hookMethods("com.android.org.conscrypt.Platform", "checkServerTrusted", function(obj, args) {
      d("SSL bypassing " + this);
    }));
    ignoreError(() => hookMethods("okhttp3.CertificatePinner", "check", function(obj, args) {
      d("SSL bypassing " + this);
      if (this.returnType.type == "boolean") {
        return true;
      }
    }));
    ignoreError(() => hookMethods("okhttp3.CertificatePinner", "check$okhttp", function(obj, args) {
      d("SSL bypassing " + this);
    }));
    ignoreError(() => hookMethods("com.android.okhttp.CertificatePinner", "check", function(obj, args) {
      d("SSL bypassing " + this);
      if (this.returnType.type == "boolean") {
        return true;
      }
    }));
    ignoreError(() => hookMethods("com.android.okhttp.CertificatePinner", "check$okhttp", function(obj, args) {
      d("SSL bypassing " + this);
      return void 0;
    }));
    ignoreError(() => hookMethods("com.android.org.conscrypt.TrustManagerImpl", "verifyChain", function(obj, args) {
      d("SSL bypassing " + this);
      return args[0];
    }));
  }
  function chooseClassLoader(className) {
    w("choose classloder: " + className);
    Java.enumerateClassLoaders({
      onMatch: function(loader) {
        try {
          const clazz = loader.findClass(className);
          if (clazz != null) {
            i("choose classloader: " + loader);
            Reflect.set(Java.classFactory, "loader", loader);
          }
        } catch (e2) {
          e(pretty2Json(e2));
        }
      },
      onComplete: function() {
        d("enumerate classLoaders complete");
      }
    });
  }
  function traceClasses(include, exclude = void 0, options = void 0) {
    include = include != null ? include.trim().toLowerCase() : "";
    exclude = exclude != null ? exclude.trim().toLowerCase() : "";
    options = options != null ? options : { stack: true, args: true };
    w("trace classes, include: " + include + ", exclude: " + exclude + ", options: " + JSON.stringify(options));
    Java.enumerateLoadedClasses({
      onMatch: function(className) {
        const targetClassName = className.toString().toLowerCase();
        if (targetClassName.indexOf(include) >= 0) {
          if (exclude == "" || targetClassName.indexOf(exclude) < 0) {
            hookAllMethods(className, getEventImpl(options));
          }
        }
      },
      onComplete: function() {
        d("enumerate classLoaders complete");
      }
    });
  }
  function runOnCreateContext(fn) {
    hookMethods("android.app.ContextImpl", "createAppContext", function(obj, args) {
      const context = this(obj, args);
      fn(context);
      return context;
    });
  }
  function runOnCreateApplication(fn) {
    hookMethods("android.app.LoadedApk", "makeApplication", function(obj, args) {
      const app = this(obj, args);
      fn(app);
      return app;
    });
  }
  function $prettyClassName(className) {
    if (className.startsWith("[L") && className.endsWith(";")) {
      return `${className.substring(2, className.length - 1)}[]`;
    } else if (className.startsWith("[")) {
      switch (className.substring(1, 2)) {
        case "B":
          return "byte[]";
        case "C":
          return "char[]";
        case "D":
          return "double[]";
        case "F":
          return "float[]";
        case "I":
          return "int[]";
        case "S":
          return "short[]";
        case "J":
          return "long[]";
        case "Z":
          return "boolean[]";
        case "V":
          return "void[]";
      }
    }
    return className;
  }
  function $defineMethodProperties(method) {
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
            for (let i2 = 1; i2 < this.argumentTypes.length; i2++) {
              args = args + ", " + $prettyClassName(this.argumentTypes[i2].className);
            }
          }
          return ret + " " + name + "(" + args + ")";
        }
      },
      toString: {
        configurable: true,
        value: function() {
          return this.name;
        }
      }
    });
  }
  function $hookMethod(method, impl = void 0) {
    if (impl != void 0) {
      const proxy = new Proxy(method, {
        apply: function(target, thisArg, argArray) {
          const obj = argArray[0];
          const args = argArray[1];
          return target.apply(obj, args);
        }
      });
      const hookImpl = isFunction(impl) ? impl : getEventImpl(impl);
      method.implementation = function() {
        return hookImpl.call(proxy, this, Array.prototype.slice.call(arguments));
      };
      i("Hook method: " + method);
    } else {
      method.implementation = null;
      i("Unhook method: " + method);
    }
  }
  function $isExcludeClass(className) {
    for (const i2 in o.excludeHookPackages) {
      if (className.indexOf(o.excludeHookPackages[i2]) == 0) {
        return true;
      }
    }
    return false;
  }
  function getErrorStack(error) {
    try {
      const handle = getObjectHandle(error);
      if (handle !== void 0) {
        const throwable = Java.cast(handle, o.throwableClass);
        let items = [];
        for (let item of getStackTrace(throwable)) {
          items.push(`    at ${item}`);
        }
        return items.length > 0 ? `${throwable}
${items.join("\n")}` : `${throwable}`;
      }
    } catch (e2) {
      d(`getErrorStack error: ${e2}`);
    }
    return void 0;
  }

  // lib/objc.ts
  var objc_exports = {};
  __export(objc_exports, {
    bypassSslPinning: () => bypassSslPinning2,
    convert2ObjcObject: () => convert2ObjcObject,
    getEventImpl: () => getEventImpl2,
    hookMethod: () => hookMethod2,
    hookMethods: () => hookMethods2,
    o: () => o2
  });
  var Objects2 = class {
    get currentViewController() {
      try {
        let currentViewController = ObjC.classes["UIApplication"].sharedApplication().keyWindow().rootViewController();
        while (currentViewController) {
          const presentedViewController = currentViewController.presentedViewController();
          if (presentedViewController) {
            currentViewController = presentedViewController;
          } else {
            if (currentViewController.isKindOfClass_(ObjC.classes["UINavigationController"])) {
              currentViewController = currentViewController.visibleViewController();
            } else if (currentViewController.isKindOfClass_(ObjC.classes["UITabBarController"])) {
              currentViewController = currentViewController.selectedViewController();
            } else {
              break;
            }
          }
        }
        return currentViewController;
      } catch (e2) {
        return null;
      }
    }
  };
  var o2 = new Objects2();
  function hookMethod2(clazz, method, impl = void 0) {
    var targetClass = clazz;
    if (typeof targetClass === "string") {
      targetClass = ObjC.classes[targetClass];
    }
    if (targetClass === void 0) {
      throw Error('cannot find class "' + clazz + '"');
    }
    var targetMethod = method;
    if (typeof targetMethod === "string") {
      targetMethod = targetClass[targetMethod];
    }
    if (targetMethod === void 0) {
      throw Error('cannot find method "' + method + '" in class "' + targetClass + '"');
    }
    $defineMethodProperties2(targetClass, targetMethod);
    $hookMethod2(targetMethod, impl);
  }
  function hookMethods2(clazz, name, impl = void 0) {
    var targetClass = clazz;
    if (typeof targetClass === "string") {
      targetClass = ObjC.classes[targetClass];
    }
    if (targetClass === void 0) {
      throw Error('cannot find class "' + clazz + '"');
    }
    const length = targetClass.$ownMethods.length;
    for (let i2 = 0; i2 < length; i2++) {
      const method = targetClass.$ownMethods[i2];
      if (method.indexOf(name) >= 0) {
        const targetMethod = targetClass[method];
        $defineMethodProperties2(targetClass, targetMethod);
        $hookMethod2(targetMethod, impl);
      }
    }
  }
  function getEventImpl2(options) {
    const hookOpts = {};
    hookOpts.method = parseBoolean(options.method, true);
    hookOpts.thread = parseBoolean(options.thread, false);
    hookOpts.stack = parseBoolean(options.stack, false);
    hookOpts.symbol = parseBoolean(options.symbol, true);
    hookOpts.backtracer = options.backtracer || "accurate";
    hookOpts.args = parseBoolean(options.args, false);
    hookOpts.result = parseBoolean(options.result, hookOpts.args);
    hookOpts.error = parseBoolean(options.error, hookOpts.args);
    hookOpts.page = parseBoolean(options.page, false);
    hookOpts.extras = {};
    if (options.extras != null) {
      for (let i2 in options.extras) {
        hookOpts.extras[i2] = options.extras[i2];
      }
    }
    return function(obj, args) {
      const event2 = {};
      for (const key in hookOpts.extras) {
        event2[key] = hookOpts.extras[key];
      }
      if (hookOpts.method !== false) {
        event2["class_name"] = new ObjC.Object(obj).$className;
        event2["method_name"] = this.name;
        event2["method_simple_name"] = this.methodName;
      }
      if (hookOpts.thread !== false) {
        event2["thread_id"] = Process.getCurrentThreadId();
        event2["thread_name"] = ObjC.classes.NSThread.currentThread().name().toString();
      }
      if (hookOpts.args !== false) {
        const objectArgs = [];
        for (let i2 = 0; i2 < args.length; i2++) {
          objectArgs.push(convert2ObjcObject(args[i2]));
        }
        event2["args"] = pretty2Json(objectArgs);
        event2["result"] = null;
        event2["error"] = null;
      }
      if (hookOpts.result !== false) {
        event2["result"] = null;
      }
      if (hookOpts.error !== false) {
        event2["error"] = null;
      }
      if (hookOpts.page !== false) {
        const viewController = o2.currentViewController;
        event2["page"] = viewController ? viewController.$className : null;
      }
      try {
        const result = this(obj, args);
        if (hookOpts.result !== false) {
          event2["result"] = pretty2Json(convert2ObjcObject(result));
        }
        return result;
      } catch (e2) {
        if (hookOpts.error !== false) {
          event2["error"] = pretty2Json(e2);
        }
        throw e2;
      } finally {
        if (hookOpts.stack !== false) {
          const stack = event2["stack"] = [];
          const backtracer = hookOpts.backtracer === "accurate" ? Backtracer.ACCURATE : Backtracer.FUZZY;
          const elements = Thread.backtrace(this.context, backtracer);
          for (let i2 = 0; i2 < elements.length; i2++) {
            stack.push(getDescFromAddress(elements[i2], hookOpts.symbol !== false));
          }
        }
        event(event2);
      }
    };
  }
  function convert2ObjcObject(obj) {
    if (obj instanceof NativePointer) {
      return new ObjC.Object(obj);
    } else if (typeof obj === "object" && obj.hasOwnProperty("handle")) {
      return new ObjC.Object(obj);
    }
    return obj;
  }
  function bypassSslPinning2() {
    w("iOS Bypass ssl pinning");
    try {
      Module.ensureInitialized("libboringssl.dylib");
    } catch (err) {
      d("libboringssl.dylib module not loaded. Trying to manually load it.");
      Module.load("libboringssl.dylib");
    }
    const customVerifyCallback = new NativeCallback(function(ssl, out_alert) {
      d(`custom SSL context verify callback, returning SSL_VERIFY_NONE`);
      return 0;
    }, "int", ["pointer", "pointer"]);
    try {
      hookFunction("libboringssl.dylib", "SSL_set_custom_verify", "void", ["pointer", "int", "pointer"], function(args) {
        d(`SSL_set_custom_verify(), setting custom callback.`);
        args[2] = customVerifyCallback;
        return this(args);
      });
    } catch (e2) {
      hookFunction("libboringssl.dylib", "SSL_CTX_set_custom_verify", "void", ["pointer", "int", "pointer"], function(args) {
        d(`SSL_CTX_set_custom_verify(), setting custom callback.`);
        args[2] = customVerifyCallback;
        return this(args);
      });
    }
    hookFunction("libboringssl.dylib", "SSL_get_psk_identity", "pointer", ["pointer"], function(args) {
      d(`SSL_get_psk_identity(), returning "fakePSKidentity"`);
      return Memory.allocUtf8String("fakePSKidentity");
    });
  }
  function $defineMethodProperties2(clazz, method) {
    const implementation = method["origImplementation"] || method.implementation;
    const className = clazz.toString();
    const methodName = ObjC.selectorAsString(method.selector);
    const isClassMethod = ObjC.classes.NSThread.hasOwnProperty(methodName);
    Object.defineProperties(method, {
      className: {
        configurable: true,
        enumerable: true,
        get() {
          return className;
        }
      },
      methodName: {
        configurable: true,
        enumerable: true,
        get() {
          return methodName;
        }
      },
      name: {
        configurable: true,
        enumerable: true,
        get() {
          return (isClassMethod ? "+" : "-") + "[" + className + " " + methodName + "]";
        }
      },
      origImplementation: {
        configurable: true,
        enumerable: true,
        get() {
          return implementation;
        }
      },
      toString: {
        value: function() {
          return this.name;
        }
      }
    });
  }
  function $hookMethod2(method, impl = void 0) {
    if (impl != void 0) {
      const hookImpl = isFunction(impl) ? impl : getEventImpl2(impl);
      method.implementation = ObjC.implement(method, function() {
        const self = this;
        const args = Array.prototype.slice.call(arguments);
        const obj = args.shift();
        const sel = args.shift();
        const proxy = new Proxy(method, {
          get: function(target, p, receiver) {
            if (p in self) {
              return self[p];
            }
            return target[p];
          },
          apply: function(target, thisArg, argArray) {
            const obj2 = argArray[0];
            const args2 = argArray[1];
            return target["origImplementation"].apply(null, [].concat(obj2, sel, args2));
          }
        });
        return hookImpl.call(proxy, obj, args);
      });
      i("Hook method: " + method);
    } else {
      method.implementation = method["origImplementation"];
      i("Unhook method: " + pretty2String(method));
    }
  }

  // lib/c.ts
  var Objects3 = class {
    get dlopen() {
      return getExportFunction(null, "dlopen", "pointer", ["pointer", "int"]);
    }
  };
  var o3 = new Objects3();
  var $moduleMap = new ModuleMap();
  var $nativeFunctionCaches = {};
  var $debugSymbolAddressCaches = {};
  function getExportFunction(moduleName, exportName, retType, argTypes) {
    const key = (moduleName || "") + "|" + exportName;
    if (key in $nativeFunctionCaches) {
      return $nativeFunctionCaches[key];
    }
    var ptr = Module.findExportByName(moduleName, exportName);
    if (ptr === null) {
      throw Error("cannot find " + exportName);
    }
    const result = $nativeFunctionCaches[key] = new NativeFunction(ptr, retType, argTypes);
    return result;
  }
  function hookFunctionWithOptions(moduleName, exportName, options) {
    return hookFunctionWithCallbacks(moduleName, exportName, getEventImpl3(options));
  }
  function hookFunctionWithCallbacks(moduleName, exportName, callbacks) {
    const funcPtr = Module.findExportByName(moduleName, exportName);
    if (funcPtr === null) {
      throw Error("cannot find " + exportName);
    }
    const proxyHandler = {
      get: function(target, p, receiver) {
        switch (p) {
          case "name":
            return exportName;
          default:
            return target[p];
        }
      }
    };
    const cb = {};
    if ("onEnter" in callbacks) {
      cb["onEnter"] = function(args) {
        const fn = callbacks.onEnter;
        fn.call(new Proxy(this, proxyHandler), args);
      };
    }
    if ("onLeave" in callbacks) {
      cb["onLeave"] = function(ret) {
        const fn = callbacks.onLeave;
        fn.call(new Proxy(this, proxyHandler), ret);
      };
    }
    const result = Interceptor.attach(funcPtr, cb);
    i("Hook function: " + exportName + " (" + funcPtr + ")");
    return result;
  }
  function hookFunction(moduleName, exportName, retType, argTypes, impl) {
    const func = getExportFunction(moduleName, exportName, retType, argTypes);
    if (func === null) {
      throw Error("cannot find " + exportName);
    }
    const hookImpl = isFunction(impl) ? impl : getEventImpl3(impl);
    const callbackArgTypes = argTypes;
    Interceptor.replace(func, new NativeCallback(function() {
      const self = this;
      const targetArgs = [];
      for (let i2 = 0; i2 < argTypes.length; i2++) {
        targetArgs[i2] = arguments[i2];
      }
      const proxy = new Proxy(func, {
        get: function(target, p, receiver) {
          switch (p) {
            case "name":
              return exportName;
            case "argumentTypes":
              return argTypes;
            case "returnType":
              return retType;
            case "context":
              return self.context;
            default:
              target[p];
          }
          ;
        },
        apply: function(target, thisArg, argArray) {
          const f = target;
          return f.apply(null, argArray[0]);
        }
      });
      return hookImpl.call(proxy, targetArgs);
    }, retType, callbackArgTypes));
    i("Hook function: " + exportName + " (" + func + ")");
  }
  function getEventImpl3(options) {
    const hookOpts = {};
    hookOpts.method = parseBoolean(options.method, true);
    hookOpts.thread = parseBoolean(options.thread, false);
    hookOpts.stack = parseBoolean(options.stack, false);
    hookOpts.symbol = parseBoolean(options.symbol, true);
    hookOpts.backtracer = options.backtracer || "accurate";
    hookOpts.args = parseBoolean(options.args, false);
    hookOpts.result = parseBoolean(options.result, hookOpts.args);
    hookOpts.error = parseBoolean(options.error, hookOpts.args);
    hookOpts.page = parseBoolean(options.page, false);
    hookOpts.extras = {};
    if (options.extras != null) {
      for (let i2 in options.extras) {
        hookOpts.extras[i2] = options.extras[i2];
      }
    }
    const result = function(args) {
      const event2 = {};
      for (const key in hookOpts.extras) {
        event2[key] = hookOpts.extras[key];
      }
      if (hookOpts.method !== false) {
        event2["method_name"] = this.name;
      }
      if (hookOpts.thread !== false) {
        event2["thread_id"] = Process.getCurrentThreadId();
      }
      if (hookOpts.args !== false) {
        event2["args"] = pretty2Json(args);
      }
      if (hookOpts.result !== false) {
        event2["result"] = null;
      }
      if (hookOpts.error !== false) {
        event2["error"] = null;
      }
      if (hookOpts.page !== false) {
        event2["page"] = $getCurrentPage();
      }
      try {
        const result2 = this(args);
        if (hookOpts.result !== false) {
          event2["result"] = pretty2Json(result2);
        }
        return result2;
      } catch (e2) {
        if (hookOpts.error !== false) {
          event2["error"] = pretty2Json(e2);
        }
        throw e2;
      } finally {
        if (hookOpts.stack !== false) {
          const stack = event2["stack"] = [];
          const backtracer = hookOpts.backtracer === "accurate" ? Backtracer.ACCURATE : Backtracer.FUZZY;
          const elements = Thread.backtrace(this.context, backtracer);
          for (let i2 = 0; i2 < elements.length; i2++) {
            stack.push(getDescFromAddress(elements[i2], hookOpts.symbol !== false));
          }
        }
        event(event2);
      }
    };
    result["onLeave"] = function(ret) {
      const event2 = {};
      for (const key in hookOpts.extras) {
        event2[key] = hookOpts.extras[key];
      }
      if (hookOpts.method !== false) {
        event2["method_name"] = this.name;
      }
      if (hookOpts.thread !== false) {
        event2["thread_id"] = Process.getCurrentThreadId();
      }
      if (hookOpts.result !== false) {
        event2["result"] = pretty2Json(ret);
      }
      if (hookOpts.page !== false) {
        event2["page"] = $getCurrentPage();
      }
      if (hookOpts.stack !== false) {
        const stack = event2["stack"] = [];
        const backtracer = hookOpts.backtracer === "accurate" ? Backtracer.ACCURATE : Backtracer.FUZZY;
        const elements = Thread.backtrace(this.context, backtracer);
        for (let i2 = 0; i2 < elements.length; i2++) {
          stack.push(getDescFromAddress(elements[i2], hookOpts.symbol !== false));
        }
      }
      event(event2);
    };
    return result;
  }
  function getDebugSymbolFromAddress(pointer) {
    const key = pointer.toString();
    let result = $debugSymbolAddressCaches[key];
    if (result === void 0) {
      result = $debugSymbolAddressCaches[key] = DebugSymbol.fromAddress(pointer);
    }
    return result;
  }
  function getDescFromAddress(pointer, symbol) {
    if (symbol) {
      const debugSymbol = getDebugSymbolFromAddress(pointer);
      if (debugSymbol != null) {
        return debugSymbol.toString();
      }
    }
    const module = $moduleMap.find(pointer);
    if (module != null) {
      return `${pointer} ${module.name}!${pointer.sub(module.base)}`;
    }
    return `${pointer}`;
  }
  function $getCurrentPage() {
    let result = null;
    try {
      if (globalThis.Java && Java.available) {
        Java.perform(function() {
          const activity = o.currentActivity;
          result = activity ? activity.$className : null;
        });
      } else if (ObjC.available) {
        const viewController = o2.currentViewController;
        result = viewController ? viewController.$className : null;
      }
    } catch (e2) {
      result = null;
    }
    return result;
  }

  // index.ts
  var logWrapper = (fn) => {
    return function() {
      if (arguments.length > 0) {
        var message = pretty2String(arguments[0]);
        for (var i2 = 1; i2 < arguments.length; i2++) {
          message += " ";
          message += pretty2String(arguments[i2]);
        }
        fn(message);
      } else {
        fn("");
      }
    };
  };
  console.debug = logWrapper(d.bind(log_exports));
  console.info = logWrapper(i.bind(log_exports));
  console.warn = logWrapper(w.bind(log_exports));
  console.error = logWrapper(e.bind(log_exports));
  console.log = logWrapper(i.bind(log_exports));
  var global = globalThis;
  if (global._setUnhandledExceptionCallback != void 0) {
    global._setUnhandledExceptionCallback(function(error) {
      let stack = void 0;
      if (error instanceof Error) {
        const errorStack = error.stack;
        if (errorStack !== void 0) {
          stack = errorStack;
        }
      }
      if (globalThis.Java && Java.available) {
        const javaStack = getErrorStack(error);
        if (javaStack !== void 0) {
          if (stack !== void 0) {
            stack += `

Caused by: 
${javaStack}`;
          } else {
            stack = javaStack;
          }
        }
      }
      exception("" + error, stack);
    });
  }
  var ScriptLoader = class {
    load(scripts, parameters) {
      for (const script of scripts) {
        try {
          let name = script.filename;
          name = name.replace(/[\/\\]/g, "$");
          name = name.replace(/[^A-Za-z0-9_$]+/g, "_");
          name = `fn_${name}`.substring(0, 255);
          const func = (0, eval)(`(function ${name}(parameters) {${script.source}
})
//# sourceURL=${script.filename}`);
          func(parameters);
        } catch (e2) {
          let message = e2.hasOwnProperty("stack") ? e2.stack : e2;
          throw new Error(`Unable to load ${script.filename}: ${message}`);
        }
      }
    }
  };
  var scriptLoader = new ScriptLoader();
  rpc.exports = {
    loadScripts: scriptLoader.load.bind(scriptLoader)
  };
  Object.defineProperties(globalThis, {
    Log: {
      enumerable: true,
      value: log_exports
    },
    CHelper: {
      enumerable: true,
      value: c_exports
    },
    JavaHelper: {
      enumerable: true,
      value: java_exports
    },
    ObjCHelper: {
      enumerable: true,
      value: objc_exports
    },
    isFunction: {
      enumerable: false,
      value: function(obj) {
        return Object.prototype.toString.call(obj) === "[object Function]";
      }
    },
    ignoreError: {
      enumerable: false,
      value: function(fn, defaultValue = void 0) {
        try {
          return fn();
        } catch (e2) {
          d("Catch ignored error. " + e2);
          return defaultValue;
        }
      }
    },
    parseBoolean: {
      enumerable: false,
      value: function(value, defaultValue = false) {
        if (typeof value === "boolean") {
          return value;
        }
        if (typeof value === "string") {
          const lower = value.toLowerCase();
          if (lower === "true") {
            return true;
          } else if (lower === "false") {
            return false;
          }
        }
        return defaultValue;
      }
    },
    pretty2String: {
      enumerable: false,
      value: function(obj) {
        if (typeof obj !== "string") {
          obj = pretty2Json(obj);
        }
        return JSON.stringify(obj);
      }
    },
    pretty2Json: {
      enumerable: false,
      value: function(obj) {
        if (!(obj instanceof Object)) {
          return obj;
        }
        if (Array.isArray(obj)) {
          let result = [];
          for (let i2 = 0; i2 < obj.length; i2++) {
            result.push(pretty2Json(obj[i2]));
          }
          return result;
        }
        if (globalThis.Java && Java.available && isJavaObject(obj)) {
          return o.objectClass.toString.apply(obj);
        }
        return ignoreError(() => obj.toString());
      }
    }
  });
})();
