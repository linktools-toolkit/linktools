/**
 *  用于方便调用frida的ObjC方法
 *  https://github.com/frida/frida-objc-bridge/blob/main/index.js
 */

import * as Log from "./log"
import * as c from "./c"

type HookOpts = {
    method?: boolean;
    thread?: boolean;
    stack?: boolean;
    symbol?: boolean;
    backtracer?: "accurate" | "fuzzy";
    args?: boolean;
    result?: boolean;
    error?: boolean;
    page?: boolean;
    extras?: {
        [name: string]: any
    };
}
type HookImpl = (obj: any, args: any[]) => any;

class Objects {

    get currentViewController(): ObjC.Object | null {
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
        } catch (e) {
            return null;
        }
    }
}

export const o = new Objects();

/**
 * 获取hook实现，调用原方法并发送调用事件
 * @param clazz 类对象
 * @param method 方法对象
 * @param impl hook实现，如调用原函数： function(obj, args) { return this(obj, sel, args); }
 * @returns hook实现
 */
export function hookMethod(
    clazz: string | ObjC.Object,
    method: string | ObjC.ObjectMethod,
    impl: HookImpl | HookOpts = null
): void {
    var targetClass: any = clazz;
    if (typeof (targetClass) === "string") {
        targetClass = ObjC.classes[targetClass];
    }
    if (targetClass === void 0) {
        throw Error("cannot find class \"" + clazz + "\"");
    }
    var targetMethod: any = method;
    if (typeof (targetMethod) === "string") {
        targetMethod = targetClass[targetMethod];
    }
    if (targetMethod === void 0) {
        throw Error("cannot find method \"" + method + "\" in class \"" + targetClass + "\"");
    }
    $defineMethodProperties(targetClass, targetMethod);
    $hookMethod(targetMethod, impl);
}

/**
 * 获取hook实现，调用原方法并发送调用事件
 * @param clazz 类对象
 * @param name 方法名（模糊匹配）
 * @param impl hook实现，如调用原函数： function(obj, sel, args) { return this(obj, sel, args); }
 * @returns hook实现
 */
export function hookMethods(
    clazz: string | ObjC.Object,
    name: string,
    impl: HookImpl | HookOpts = null
): void {
    var targetClass: any = clazz;
    if (typeof (targetClass) === "string") {
        targetClass = ObjC.classes[targetClass];
    }
    if (targetClass === void 0) {
        throw Error("cannot find class \"" + clazz + "\"");
    }
    const length = targetClass.$ownMethods.length;
    for (let i = 0; i < length; i++) {
        const method = targetClass.$ownMethods[i];
        if (method.indexOf(name) >= 0) {
            const targetMethod = targetClass[method];
            $defineMethodProperties(targetClass, targetMethod);
            $hookMethod(targetMethod, impl);
        }
    }
}

/**
 * 获取hook实现，调用原方法并发送调用事件
 * @param options hook选项，如：{stack: true, args: true, thread: true}
 * @returns hook实现
 */
export function getEventImpl(options: HookOpts): HookImpl {
    const hookOpts: HookOpts = {};
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
            event["class_name"] = new ObjC.Object(obj).$className
            event["method_name"] = this.name;
            event["method_simple_name"] = this.methodName;
        }
        if (hookOpts.thread !== false) {
            event["thread_id"] = Process.getCurrentThreadId();
            event["thread_name"] = ObjC.classes.NSThread.currentThread().name().toString()
        }
        if (hookOpts.args !== false) {
            const objectArgs = []
            for (let i = 0; i < args.length; i++) {
                objectArgs.push(convert2ObjcObject(args[i]));
            }
            event["args"] = pretty2Json(objectArgs);
            event["result"] = null;
            event["error"] = null;
        }
        if (hookOpts.result !== false) {
            event["result"] = null;
        }
        if (hookOpts.error !== false) {
            event["error"] = null;
        }
        if (hookOpts.page !== false) {
            const viewController = o.currentViewController;
            event["page"] = viewController ? viewController.$className : null;
        }
        try {
            const result = this(obj, args);
            if (hookOpts.result !== false) {
                event["result"] = pretty2Json(convert2ObjcObject(result));
            }
            return result;
        } catch (e) {
            if (hookOpts.error !== false) {
                event["error"] = pretty2Json(e);
            }
            throw e;
        } finally {
            if (hookOpts.stack !== false) {
                const stack = event["stack"] = [];
                const backtracer = hookOpts.backtracer === "accurate" ? Backtracer.ACCURATE : Backtracer.FUZZY;
                const elements = Thread.backtrace(this.context, backtracer);
                for (let i = 0; i < elements.length; i++) {
                    stack.push(c.getDescFromAddress(elements[i], hookOpts.symbol !== false));
                }
            }
            Log.event(event);
        }
    };
}

/**
 * 将指针转化为objc对象
 * @param obj 指针
 * @returns objc对象
 */
export function convert2ObjcObject(obj: any) {
    if (obj instanceof NativePointer) {
        return new ObjC.Object(obj);
    } else if (typeof obj === 'object' && obj.hasOwnProperty('handle')) {
        return new ObjC.Object(obj);
    }
    return obj;
}

/**
 * 绕过ssl pinning
 * copy from https://github.com/sensepost/objection/blob/master/agent/src/ios/pinning.ts
 */
export function bypassSslPinning() {

    Log.w("iOS Bypass ssl pinning");

    try {
        Module.ensureInitialized("libboringssl.dylib");
    } catch(err) {
        Log.d("libboringssl.dylib module not loaded. Trying to manually load it.")
        Module.load("libboringssl.dylib");  
    }

    const customVerifyCallback = new NativeCallback(function (ssl, out_alert) {
        Log.d(`custom SSL context verify callback, returning SSL_VERIFY_NONE`);
        return 0;
    }, "int", ["pointer", "pointer"]);

    try {
        c.hookFunction("libboringssl.dylib", "SSL_set_custom_verify", "void", ["pointer", "int", "pointer"], function(args) {
            Log.d(`SSL_set_custom_verify(), setting custom callback.`);
            args[2] = customVerifyCallback;
            return this(args);
        });
    } catch (e) {
        c.hookFunction("libboringssl.dylib", "SSL_CTX_set_custom_verify", "void", ["pointer", "int", "pointer"], function(args) {
            Log.d(`SSL_CTX_set_custom_verify(), setting custom callback.`);
            args[2] = customVerifyCallback;
            return this(args);
        });
    }

    c.hookFunction("libboringssl.dylib", "SSL_get_psk_identity", "pointer", ["pointer"], function(args) {
        Log.d(`SSL_get_psk_identity(), returning "fakePSKidentity"`);
        return Memory.allocUtf8String("fakePSKidentity");
    });
}

/**
 * 为method添加properties
 * @param clazz 类对象
 * @param method 方法对象
 */
function $defineMethodProperties(clazz: ObjC.Object, method: ObjC.ObjectMethod): void {
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
            },
        },
        methodName: {
            configurable: true,
            enumerable: true,
            get() {
                return methodName;
            },
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
            value: function () {
                return this.name;
            }
        }
    });
}


/**
 * hook指定方法对象
 * @param method 方法对象
 * @param impl hook实现，如调用原函数： function(obj, sel, args) { return this(obj, sel, args); }
 */
function $hookMethod(method: ObjC.ObjectMethod, impl: HookImpl | HookOpts = null): void {
    if (impl != null) {
        const hookImpl = isFunction(impl) ? impl as HookImpl : getEventImpl(impl as HookOpts);
        method.implementation = ObjC.implement(method, function () {
            const self = this;
            const args = Array.prototype.slice.call(arguments);
            const obj = args.shift();
            const sel = args.shift();
            const proxy: ObjC.ObjectMethod = new Proxy(method, {
                get: function (target, p: string | symbol, receiver: any) {
                    if (p in self) {
                        return self[p];
                    }
                    return target[p];
                },
                apply: function (target, thisArg: any, argArray: any[]) {
                    const obj = argArray[0];
                    const args = argArray[1];
                    return target["origImplementation"].apply(null, [].concat(obj, sel, args));
                }
            });
            return hookImpl.call(proxy, obj, args);
        });
        Log.i("Hook method: " + method);
    } else {
        method.implementation = method["origImplementation"];
        Log.i("Unhook method: " + pretty2String(method));
    }
}
