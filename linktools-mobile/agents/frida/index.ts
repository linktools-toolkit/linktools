import * as log from "./lib/log"
import * as c from "./lib/c"
import * as java from "./lib/java";
import * as objc from "./lib/objc";

////////////////////////////////////////////////////////////////////////
// 处理 console相关方法 和 未处理exception的log
////////////////////////////////////////////////////////////////////////

const logWrapper = (fn: (msg: string) => any) => {
    return function () {
        if (arguments.length > 0) {
            var message = pretty2String(arguments[0]);
            for (var i = 1; i < arguments.length; i++) {
                message += " ";
                message += pretty2String(arguments[i]);
            }
            fn(message);
        } else {
            fn("");
        }
    }
};

console.debug = logWrapper(log.d.bind(log));
console.info = logWrapper(log.i.bind(log));
console.warn = logWrapper(log.w.bind(log));
console.error = logWrapper(log.e.bind(log));
console.log = logWrapper(log.i.bind(log));


const global: any = globalThis;
if (global._setUnhandledExceptionCallback != void 0) {
    global._setUnhandledExceptionCallback(function(error: any) {
        let stack: any = void 0;
        if (error instanceof Error) {
            const errorStack = error.stack;
            if (errorStack !== void 0) {
                stack = errorStack;
            }
        }
        if (globalThis.Java && Java.available) {
            const javaStack = java.getErrorStack(error);
            if (javaStack !== void 0) {
                if (stack !== void 0) {
                    stack += `\n\nCaused by: \n${javaStack}`;
                } else {
                    stack = javaStack;
                }
            }
        }
        log.exception("" + error, stack);
    });
}


////////////////////////////////////////////////////////////////////////
// 处理 script loader
////////////////////////////////////////////////////////////////////////

interface Parameters {
    [name: string]: any;
}

interface Script {
    filename: string;
    source: string;
}

export class ScriptLoader {

    load(scripts: Script[], parameters: Parameters) {
        for (const script of scripts) {
            try {
                let name = script.filename;
                name = name.replace(/[\/\\]/g, '$');
                name = name.replace(/[^A-Za-z0-9_$]+/g, "_");
                name = `fn_${name}`.substring(0, 255);
                const func = (0, eval)(
                    `(function ${name}(parameters) {${script.source}\n})\n` +
                    `//# sourceURL=${script.filename}`
                )
                func(parameters);
            } catch (e: any) {
                let message = e.hasOwnProperty("stack") ? e.stack : e;
                throw new Error(`Unable to load ${script.filename}: ${message}`);
            }
        }
    }
}

const scriptLoader = new ScriptLoader();

rpc.exports = {
    loadScripts: scriptLoader.load.bind(scriptLoader),
};

////////////////////////////////////////////////////////////////////////
// global variables
////////////////////////////////////////////////////////////////////////

declare global {
    function isFunction(obj: any): boolean;
    function ignoreError<T>(fn: () => T): T;
    function ignoreError<T>(fn: () => T, defaultValue: T): T;
    function parseBoolean(value: string | boolean | undefined): boolean;
    function parseBoolean(value: string | boolean | undefined, defaultValue: boolean): boolean;
    function pretty2String(obj: any): any;
    function pretty2Json(obj: any): any;
}


Object.defineProperties(globalThis, {
    Log: {
        enumerable: true,
        value: log
    },
    CHelper: {
        enumerable: true,
        value: c
    },
    JavaHelper: {
        enumerable: true,
        value: java
    },
    ObjCHelper: {
        enumerable: true,
        value: objc
    },
    isFunction: {
        enumerable: false,
        value: function (obj: any): boolean {
            return Object.prototype.toString.call(obj) === "[object Function]"
        }
    },
    ignoreError: {
        enumerable: false,
        value: function <T>(fn: () => T, defaultValue: T | undefined = void 0): T | undefined {
            try {
                return fn();
            } catch (e) {
                log.d("Catch ignored error. " + e);
                return defaultValue;
            }
        }
    },
    parseBoolean: {
        enumerable: false,
        value: function (value: string | boolean, defaultValue: boolean = false): boolean {
            if (typeof (value) === "boolean") {
                return value;
            }
            if (typeof (value) === "string") {
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
        value: function (obj: any): string {
            if (typeof obj !== "string") {
                obj = pretty2Json(obj);
            }
            return JSON.stringify(obj);
        }
    },
    pretty2Json: {
        enumerable: false,
        value: function (obj: any): any {
            if (!(obj instanceof Object)) {
                return obj;
            }
            if (Array.isArray(obj)) {
                let result: any[] = [];
                for (let i = 0; i < obj.length; i++) {
                    result.push(pretty2Json(obj[i]));
                }
                return result;
            }
            if (globalThis.Java && Java.available && java.isJavaObject(obj)) {
                return java.o.objectClass.toString.apply(obj);
            }
            return ignoreError(() => obj.toString());
        }
    },
});
