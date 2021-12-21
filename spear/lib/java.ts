import { Base, Log } from "./base";

/**
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

/**
 *  用于方便调用frida的java方法
 */
export class JavaHelper extends Base {

    get javaClass(): Java.Wrapper {
        return Java.use("java.lang.Class");
    }

    get javaString(): Java.Wrapper {
        return Java.use("java.lang.String");
    }

    get javaThrowable(): Java.Wrapper {
        return Java.use("java.lang.Throwable");
    }

    /**
     * 获取类对象类名
     * @param clazz 类对象
     * @returns 类名
     */
    getClassName<T extends Java.Members<T> = {}>(clazz: Java.Wrapper<T>): string {
        return clazz.$classWrapper.__name__;
    }

    /**
     * 获取java类的类对象
     * @param className java类名
     * @param classloader java类所在的ClassLoader
     * @returns 类对象
     */
    findClass<T extends Java.Members<T> = {}>(className: string, classloader: Java.Wrapper = void 0): Java.Wrapper<T> {
        if (classloader !== void 0) {
            var originClassloader = Java.classFactory.loader;
            try {
                Reflect.set(Java.classFactory, "loader", classloader);
                return Java.use(className);
            } finally {
                Reflect.set(Java.classFactory, "loader", originClassloader);
            }
        } else {
            var error = null;
            var loaders = Java.enumerateClassLoadersSync();
            for (var i in loaders) {
                try {
                    var clazz = this.findClass<T>(className, loaders[i]);
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
     * 为method添加properties
     * @param method 方法对象
     */
    $fixMethod<T extends Java.Members<T> = {}>(method: Java.Method<T>): void {
        method.toString = function () {
            var ret = this.returnType.className;
            var name = (this.holder.$className || this.holder.__name__) + "." + this.methodName;
            var args = "";
            if (this.argumentTypes.length > 0) {
                args = this.argumentTypes[0].className;
                for (var i = 1; i < this.argumentTypes.length; i++) {
                    args = args + ", " + this.argumentTypes[i].className;
                }
            }
            return ret + " " + name + "(" + args + ")";
        };
    }

    /**
     * hook指定方法对象
     * @param method 方法对象
     * @param impl hook实现，如调用原函数： function(obj, args) { return this.apply(obj, args); }
     */
    $hookMethod<T extends Java.Members<T> = {}>(method: Java.Method<T>, impl: (obj: Java.Wrapper<T>, args: any[]) => any = null): void {
        if (impl != null) {
            method.implementation = function () {
                return impl.call(method, this, arguments);
            };
            this.$fixMethod(method);
            Log.i("Hook method: " + method);
        } else {
            method.implementation = null;
            this.$fixMethod(method);
            Log.i("Unhook method: " + method);
        }
    }

    /**
     * hook指定方法对象
     * @param clazz java类名/类对象
     * @param method java方法名/方法对象
     * @param signature java方法签名，为null表示不设置签名
     * @param impl hook实现，如调用原函数： function(obj, args) { return this.apply(obj, args); }
     */
    hookMethod<T extends Java.Members<T> = {}>(
        clazz: string | Java.Wrapper<T>,
        method: string | Java.Method<T>,
        signatures: (string | Java.Wrapper<T>)[],
        impl: (obj: Java.Wrapper<T>, args: any[]) => any = null
    ): void {
        var traget_method: any = method;
        if (typeof (traget_method) === "string") {
            var targetClass: any = clazz;
            if (typeof (targetClass) === "string") {
                targetClass = this.findClass(targetClass);
            }
            traget_method = targetClass[traget_method];
            if (signatures != null) {
                var targetSignatures: any[] = signatures;
                for (var i in targetSignatures) {
                    if (typeof (targetSignatures[i]) !== "string") {
                        targetSignatures[i] = this.getClassName(targetSignatures[i]);
                    }
                }
                traget_method = traget_method.overload.apply(traget_method, targetSignatures);
            }
        }
        this.$hookMethod(traget_method, impl);
    }

    /**
     * hook指定方法名的所有重载
     * @param clazz java类名/类对象
     * @param method java方法名
     * @param impl hook实现，如调用原函数： function(obj, args) { return this.apply(obj, args); }
     */
    hookMethods<T extends Java.Members<T> = {}>(
        clazz: string | Java.Wrapper<T>,
        methodName: string,
        impl: (obj: Java.Wrapper<T>, args: any[]) => any = null
    ): void {
        var targetClass: any = clazz;
        if (typeof (targetClass) === "string") {
            targetClass = this.findClass(targetClass);
        }
        var methods: Java.Method<T>[] = targetClass[methodName].overloads;
        for (var i = 0; i < methods.length; i++) {
            /* 过滤一些不存在的方法（拿不到返回值） */
            if (methods[i].returnType !== void 0 &&
                methods[i].returnType.className !== void 0) {
                this.$hookMethod(methods[i], impl);
            }
        }
    }

    /**
     * hook指定类的所有方法
     * @param clazz java类名/类对象
     * @param impl hook实现，如调用原函数： function(obj, args) { return this.apply(obj, args); }
     */
    hookClass<T extends Java.Members<T> = {}>(
        clazz: string | Java.Wrapper<T>,
        impl: (obj: Java.Wrapper<T>, args: any[]) => any = null
    ): void {
        var targetClass: any = clazz;
        if (typeof (targetClass) === "string") {
            targetClass = this.findClass(targetClass);
        }

        /* hook constructor*/
        this.hookMethods(targetClass, "$init", impl);

        /* hook method */
        var methodNames = [];
        var targetJavaClass = targetClass.class;
        while (targetJavaClass != null && targetJavaClass.getName() !== "java.lang.Object") {
            var methods = targetJavaClass.getDeclaredMethods();
            for (let i = 0; i < methods.length; i++) {
                const method = methods[i];
                var methodName = method.getName();
                if (methodNames.indexOf(methodName) < 0) {
                    methodNames.push(methodName);
                    this.hookMethods(targetClass, methodName, impl);
                }
            }
            targetJavaClass = Java.cast(targetJavaClass.getSuperclass(), this.javaClass);
        }
    }

    /**
     * 根据当前栈调用原java方法
     * @param obj java对象
     * @param args java参数
     * @returns java方法返回值
     */
    callMethod<T extends Java.Members<T> = {}>(obj: Java.Wrapper<T>, args: any[]): any {
        var methodName = this.getStackTrace()[0].getMethodName();
        if (methodName === "<init>") {
            methodName = "$init";
        }
        return Reflect.get(obj, methodName).apply(obj, args);
    }

    /**
     * 获取hook实现，调用愿方法并展示栈和返回值
     * @param options hook选项，如：{printStack: true, printArgs: true}
     * @returns hook实现
     */
    getHookImpl<T extends Java.Members<T> = {}>(options: any): (obj: Java.Wrapper<T>, args: any[]) => any {
        var helper = this;
        var printStack = options["printStack"] || false;
        var printArgs = options["printArgs"] || false;
        return function (obj, args) {
            var message = {};
            var ret = this.apply(obj, args);
            if (printStack !== false) {
                message = Object.assign(message, helper.$makeStackObject(this));
            }
            if (printArgs !== false) {
                message = Object.assign(message, helper.$makeArgsObject(args, ret, this));
            }
            if (Object.keys(message).length !== 0) {
                Log.i(message);
            }
            return ret;
        };
    }

    /**
     * java数组转为js数组
     * @param clazz java类名/类对象
     * @param array java数组
     * @returns js数组
     */
    fromJavaArray<T extends Java.Members<T> = {}>(
        clazz: string | Java.Wrapper<T>,
        array: Java.Wrapper<T>
    ): Java.Wrapper<T>[] {
        var targetClass: any = clazz;
        if (typeof (targetClass) === "string") {
            targetClass = this.findClass(targetClass);
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
    getEnumValue<T extends Java.Members<T> = {}>(
        clazz: string | Java.Wrapper<T>,
        name: string
    ): Java.Wrapper<T> {
        var targetClass: any = clazz;
        if (typeof (targetClass) === "string") {
            targetClass = this.findClass(targetClass);
        }
        var values = targetClass.class.getEnumConstants();
        if (!(values instanceof Array)) {
            values = this.fromJavaArray(targetClass, values);
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
     * @param printStack 是否展示栈，默认为true
     * @param printArgs 是否展示参数，默认为true
     * @returns java栈对象
     */
    getStackTrace<T extends Java.Members<T> = {}>(): Java.Wrapper<T>[] {
        return this.javaThrowable.$new().getStackTrace();
    }

    $makeStackObject<T extends Java.Members<T> = {}>(message: string, elements: Java.Wrapper<T>[] = void 0) {
        if (elements === void 0) {
            elements = this.getStackTrace()
        }
        var body = "Stack: " + message;
        for (var i = 0; i < elements.length; i++) {
            body += "\n    at " + this.toString(elements[i]);
        }
        return { "stack": body };
    }

    /**
     * 打印当前栈
     * @param message 回显的信息
     */
    printStack(message: any = void 0): void {
        var elements = this.getStackTrace();
        if (message == void 0) {
            message = elements[0];
        }
        Log.i(this.$makeStackObject(message, elements));
    }

    /**
     * 调用java对象的toString方法
     * @param obj java对象
     * @returns toString返回值
     */
    toString(obj: any): string {
        if (obj === void 0 || obj == null || !(obj instanceof Object)) {
            return obj;
        }
        if (Array.isArray(obj)) {
            var array = [];
            for (var i = 0; i < obj.length; i++) {
                array.push(this.toString(obj[i]));
            }
            return "[" + array.toString() + "]";
        }
        return this.ignoreError(function () {
            return obj.toString();
        }, void 0);
    }

    $makeArgsObject(args: any, ret: any, message: any) {
        var body = "Arguments: " + message;
        for (var i = 0; i < args.length; i++)
            body += "\n    Arguments[" + i + "]: " + this.toString(args[i]);
        if (ret !== void 0)
            body += "\n    Return: " + this.toString(ret);
        return { "arguments": body };
    }

    /**
     * 打印当前参数和返回值
     * @param args java方法参数
     * @param ret java方法返回值
     * @param message 回显的信息
     */
    printArguments(args: any, ret: any, message: any = void 0) {
        if (message === void 0) {
            message = this.getStackTrace()[0];
        }
        Log.i(this.$makeArgsObject(args, ret, message));
    }
}