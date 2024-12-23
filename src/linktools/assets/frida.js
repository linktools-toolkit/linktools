(function(){function r(e,n,t){function o(i,f){if(!n[i]){if(!e[i]){var c="function"==typeof require&&require;if(!f&&c)return c(i,!0);if(u)return u(i,!0);var a=new Error("Cannot find module '"+i+"'");throw a.code="MODULE_NOT_FOUND",a}var p=n[i]={exports:{}};e[i][0].call(p.exports,function(r){var n=e[i][1][r];return o(n||r)},p,p.exports,r,e,n,t)}return n[i].exports}for(var u="function"==typeof require&&require,i=0;i<t.length;i++)o(t[i]);return o}return r})()({1:[function(require,module,exports){
(function (global){(function (){
"use strict";

Object.defineProperty(exports, "__esModule", {
  value: !0
}), exports.ScriptLoader = void 0;

var e = require("./lib/log"), r = require("./lib/c"), n = require("./lib/java"), t = require("./lib/objc"), o = function(e) {
  return function() {
    if (arguments.length > 0) {
      for (var r = pretty2String(arguments[0]), n = 1; n < arguments.length; n++) r += " ", 
      r += pretty2String(arguments[n]);
      e(r);
    } else e("");
  };
};

console.debug = o(e.d.bind(e)), console.info = o(e.i.bind(e)), console.warn = o(e.w.bind(e)), 
console.error = o(e.e.bind(e)), console.log = o(e.i.bind(e)), null != global._setUnhandledExceptionCallback && global._setUnhandledExceptionCallback((function(r) {
  var t = void 0;
  if (r instanceof Error) {
    var o = r.stack;
    void 0 !== o && (t = o);
  }
  if (Java.available) {
    var a = n.getErrorStack(r);
    void 0 !== a && (void 0 !== t ? t += "\n\nCaused by: \n".concat(a) : t = a);
  }
  e.exception("" + r, t);
}));

var a = function() {
  function e() {}
  return e.prototype.load = function(e, r) {
    for (var n = 0, t = e; n < t.length; n++) {
      var o = t[n];
      try {
        var a = o.filename;
        a = (a = a.replace(/[\/\\]/g, "$")).replace(/[^A-Za-z0-9_$]+/g, "_"), a = "fn_".concat(a).substring(0, 255), 
        (0, eval)("(function ".concat(a, "(parameters) {").concat(o.source, "\n})\n") + "//# sourceURL=".concat(o.filename))(r);
      } catch (e) {
        var i = e.hasOwnProperty("stack") ? e.stack : e;
        throw new Error("Unable to load ".concat(o.filename, ": ").concat(i));
      }
    }
  }, e;
}();

exports.ScriptLoader = a;

var i = new a;

rpc.exports = {
  loadScripts: i.load.bind(i)
}, Object.defineProperties(globalThis, {
  Log: {
    enumerable: !0,
    value: e
  },
  CHelper: {
    enumerable: !0,
    value: r
  },
  JavaHelper: {
    enumerable: !0,
    value: n
  },
  ObjCHelper: {
    enumerable: !0,
    value: t
  },
  isFunction: {
    enumerable: !1,
    value: function(e) {
      return "[object Function]" === Object.prototype.toString.call(e);
    }
  },
  ignoreError: {
    enumerable: !1,
    value: function(r, n) {
      void 0 === n && (n = void 0);
      try {
        return r();
      } catch (r) {
        return e.d("Catch ignored error. " + r), n;
      }
    }
  },
  parseBoolean: {
    enumerable: !1,
    value: function(e, r) {
      if (void 0 === r && (r = void 0), "boolean" == typeof e) return e;
      if ("string" == typeof e) {
        var n = e.toLowerCase();
        if ("true" === n) return !0;
        if ("false" === n) return !1;
      }
      return r;
    }
  },
  pretty2String: {
    enumerable: !1,
    value: function(e) {
      return "string" != typeof e && (e = pretty2Json(e)), JSON.stringify(e);
    }
  },
  pretty2Json: {
    enumerable: !1,
    value: function(e) {
      if (!(e instanceof Object)) return e;
      if (Array.isArray(e)) {
        for (var r = [], t = 0; t < e.length; t++) r.push(pretty2Json(e[t]));
        return r;
      }
      return Java.available && n.isJavaObject(e) ? n.o.objectClass.toString.apply(e) : ignoreError((function() {
        return e.toString();
      }));
    }
  }
});

}).call(this)}).call(this,typeof global !== "undefined" ? global : typeof self !== "undefined" ? self : typeof window !== "undefined" ? window : {})

},{"./lib/c":2,"./lib/java":3,"./lib/log":4,"./lib/objc":5}],2:[function(require,module,exports){
"use strict";

Object.defineProperty(exports, "__esModule", {
  value: !0
}), exports.getDescFromAddress = exports.getDebugSymbolFromAddress = exports.getEventImpl = exports.hookFunction = exports.hookFunctionWithCallbacks = exports.hookFunctionWithOptions = exports.getExportFunction = exports.o = void 0;

var r = require("./log"), t = function() {
  function r() {}
  return Object.defineProperty(r.prototype, "dlopen", {
    get: function() {
      return a(null, "dlopen", "pointer", [ "pointer", "int" ]);
    },
    enumerable: !1,
    configurable: !0
  }), r;
}();

exports.o = new t;

var e = new ModuleMap, n = {}, o = {};

function a(r, t, e, o) {
  var a = (r || "") + "|" + t;
  if (a in n) return n[a];
  var s = Module.findExportByName(r, t);
  if (null === s) throw Error("cannot find " + t);
  return n[a] = new NativeFunction(s, e, o), n[a];
}

function s(r, t, e) {
  return c(r, t, i(e));
}

function c(t, e, n) {
  var o = Module.findExportByName(t, e);
  if (null === o) throw Error("cannot find " + e);
  var a = {
    get: function(r, t, n) {
      return "name" === t ? e : r[t];
    }
  }, s = {};
  "onEnter" in n && (s.onEnter = function(r) {
    n.onEnter.call(new Proxy(this, a), r);
  }), "onLeave" in n && (s.onLeave = function(r) {
    n.onLeave.call(new Proxy(this, a), r);
  });
  var c = Interceptor.attach(o, s);
  return r.i("Hook function: " + e + " (" + o + ")"), c;
}

function u(t, e, n, o, s) {
  var c = a(t, e, n, o);
  if (null === c) throw Error("cannot find " + e);
  var u = isFunction(s) ? s : i(s), l = o;
  Interceptor.replace(c, new NativeCallback((function() {
    for (var r = this, t = [], a = 0; a < o.length; a++) t[a] = arguments[a];
    var s = new Proxy(c, {
      get: function(t, a, s) {
        switch (a) {
         case "name":
          return e;

         case "argumentTypes":
          return o;

         case "returnType":
          return n;

         case "context":
          return r.context;

         default:
          t[a];
        }
      },
      apply: function(r, t, e) {
        return r.apply(null, e[0]);
      }
    });
    return u.call(s, t);
  }), n, l)), r.i("Hook function: " + e + " (" + c + ")");
}

function i(t) {
  var e = {};
  if (e.method = parseBoolean(t.method, !0), e.thread = parseBoolean(t.thread, !1), 
  e.stack = parseBoolean(t.stack, !1), e.symbol = parseBoolean(t.symbol, !0), e.backtracer = t.backtracer || "accurate", 
  e.args = parseBoolean(t.args, !1), e.result = parseBoolean(t.result, e.args), e.extras = {}, 
  null != t.extras) for (var n in t.extras) e.extras[n] = t.extras[n];
  var o = function(t) {
    var n = {};
    for (var o in e.extras) n[o] = e.extras[o];
    !1 !== e.method && (n.method_name = this.name), !1 !== e.thread && (n.thread_id = Process.getCurrentThreadId()), 
    !1 !== e.args && (n.args = pretty2Json(t)), !1 !== e.result && (n.result = null), 
    !1 === e.args && !1 === e.result || (n.error = null);
    try {
      var a = this(t);
      return !1 !== e.result && (n.result = pretty2Json(a)), a;
    } catch (r) {
      throw !1 === e.args && !1 === e.result || (n.error = pretty2Json(r)), r;
    } finally {
      if (!1 !== e.stack) for (var s = n.stack = [], c = "accurate" === e.backtracer ? Backtracer.ACCURATE : Backtracer.FUZZY, u = Thread.backtrace(this.context, c), i = 0; i < u.length; i++) s.push(p(u[i], !1 !== e.symbol));
      r.event(n);
    }
  };
  return o.onLeave = function(t) {
    var n = {};
    for (var o in e.extras) n[o] = e.extras[o];
    if (!1 !== e.method && (n.method_name = this.name), !1 !== e.thread && (n.thread_id = Process.getCurrentThreadId()), 
    !1 !== e.result && (n.result = pretty2Json(t)), !1 !== e.stack) for (var a = n.stack = [], s = "accurate" === e.backtracer ? Backtracer.ACCURATE : Backtracer.FUZZY, c = Thread.backtrace(this.context, s), u = 0; u < c.length; u++) a.push(p(c[u], !1 !== e.symbol));
    r.event(n);
  }, o;
}

function l(r) {
  var t = r.toString();
  return void 0 === o[t] && (o[t] = DebugSymbol.fromAddress(r)), o[t];
}

function p(r, t) {
  if (t) {
    var n = l(r);
    if (null != n) return n.toString();
  }
  var o = e.find(r);
  return null != o ? "".concat(r, " ").concat(o.name, "!").concat(r.sub(o.base)) : "".concat(r);
}

exports.getExportFunction = a, exports.hookFunctionWithOptions = s, exports.hookFunctionWithCallbacks = c, 
exports.hookFunction = u, exports.getEventImpl = i, exports.getDebugSymbolFromAddress = l, 
exports.getDescFromAddress = p;

},{"./log":4}],3:[function(require,module,exports){
"use strict";

Object.defineProperty(exports, "__esModule", {
  value: !0
}), exports.getErrorStack = exports.runOnCreateApplication = exports.runOnCreateContext = exports.traceClasses = exports.chooseClassLoader = exports.bypassSslPinning = exports.setWebviewDebuggingEnabled = exports.use = exports.getStackTrace = exports.getJavaEnumValue = exports.fromJavaArray = exports.isJavaArray = exports.isJavaObject = exports.getEventImpl = exports.hookClass = exports.hookAllMethods = exports.hookAllConstructors = exports.hookMethods = exports.hookMethod = exports.findClass = exports.getClassMethod = exports.getClassName = exports.getObjectHandle = exports.isSameObject = exports.o = void 0;

var e = require("./log"), r = function() {
  function e() {
    this.excludeHookPackages = [ "java.", "javax.", "android.", "androidx." ];
  }
  return Object.defineProperty(e.prototype, "objectClass", {
    get: function() {
      return Java.use("java.lang.Object");
    },
    enumerable: !1,
    configurable: !0
  }), Object.defineProperty(e.prototype, "classClass", {
    get: function() {
      return Java.use("java.lang.Class");
    },
    enumerable: !1,
    configurable: !0
  }), Object.defineProperty(e.prototype, "classLoaderClass", {
    get: function() {
      return Java.use("java.lang.ClassLoader");
    },
    enumerable: !1,
    configurable: !0
  }), Object.defineProperty(e.prototype, "stringClass", {
    get: function() {
      return Java.use("java.lang.String");
    },
    enumerable: !1,
    configurable: !0
  }), Object.defineProperty(e.prototype, "threadClass", {
    get: function() {
      return Java.use("java.lang.Thread");
    },
    enumerable: !1,
    configurable: !0
  }), Object.defineProperty(e.prototype, "throwableClass", {
    get: function() {
      return Java.use("java.lang.Throwable");
    },
    enumerable: !1,
    configurable: !0
  }), Object.defineProperty(e.prototype, "uriClass", {
    get: function() {
      return Java.use("android.net.Uri");
    },
    enumerable: !1,
    configurable: !0
  }), Object.defineProperty(e.prototype, "urlClass", {
    get: function() {
      return Java.use("java.net.URL");
    },
    enumerable: !1,
    configurable: !0
  }), Object.defineProperty(e.prototype, "mapClass", {
    get: function() {
      return Java.use("java.util.Map");
    },
    enumerable: !1,
    configurable: !0
  }), Object.defineProperty(e.prototype, "hashSetClass", {
    get: function() {
      return Java.use("java.util.HashSet");
    },
    enumerable: !1,
    configurable: !0
  }), Object.defineProperty(e.prototype, "applicationContext", {
    get: function() {
      return Java.use("android.app.ActivityThread").currentApplication().getApplicationContext();
    },
    enumerable: !1,
    configurable: !0
  }), e;
}();

function t(e, r) {
  return e === r || null != e && null != r && (!!e.hasOwnProperty("$isSameObject") && e.$isSameObject(r));
}

function n(e) {
  return null == e ? null : e.hasOwnProperty("$h") ? e.$h : void 0;
}

function a(r) {
  var t = r.$className;
  if (null != t) return t;
  if (null != (t = r.__name__)) return t;
  if (null != r.$classWrapper) {
    if (null != (t = r.$classWrapper.$className)) return t;
    if (null != (t = r.$classWrapper.__name__)) return t;
  }
  e.e("Cannot get class name: " + r);
}

function o(e, r) {
  var t = e[r];
  return void 0 !== t || "$" == r[0] && void 0 !== (t = e["_" + r]) ? t : void 0;
}

function s(e, r) {
  if (void 0 === r && (r = void 0), void 0 !== r && null != r) return Java.ClassFactory.get(r).use(e);
  if (parseInt(Java.androidVersion) < 7) return Java.use(e);
  for (var t = null, n = 0, a = Java.enumerateClassLoadersSync(); n < a.length; n++) {
    var o = a[n];
    try {
      var i = s(e, o);
      if (null != i) return i;
    } catch (e) {
      null == t && (t = e);
    }
  }
  throw t;
}

function i(e, r, t, n) {
  void 0 === n && (n = null);
  var i = r;
  if ("string" == typeof i) {
    var l = i, c = e;
    "string" == typeof c && (c = s(c));
    var u = o(c, l);
    if (void 0 === u || void 0 === u.overloads) throw Error("Cannot find method: " + a(c) + "." + l);
    if (null != t) {
      var p = t;
      for (var d in p) "string" != typeof p[d] && (p[d] = a(p[d]));
      i = u.overload.apply(u, p);
    } else {
      if (1 != u.overloads.length) throw Error(a(c) + "." + l + " has too many overloads");
      i = u.overloads[0];
    }
  }
  P(i), E(i, n);
}

function l(e, r, t) {
  void 0 === t && (t = null);
  var n = e;
  "string" == typeof n && (n = s(n));
  var i = o(n, r);
  if (void 0 === i || void 0 === i.overloads) throw Error("Cannot find method: " + a(n) + "." + r);
  for (var l = 0; l < i.overloads.length; l++) {
    var c = i.overloads[l];
    void 0 !== c.returnType && void 0 !== c.returnType.className && (P(c), E(c, t));
  }
}

function c(e, r) {
  void 0 === r && (r = null);
  var t = e;
  "string" == typeof t && (t = s(t)), l(t, "$init", r);
}

function u(e, r) {
  void 0 === r && (r = null);
  var t = e;
  "string" == typeof t && (t = s(t));
  for (var n = [], a = null, o = t.class; null != o; ) {
    for (var i = o.getDeclaredMethods(), c = 0; c < i.length; c++) {
      var u = i[c].getName();
      n.indexOf(u) < 0 && (n.push(u), l(t, u, r));
    }
    if (a = o.getSuperclass(), o.$dispose(), null == a) break;
    if (L((o = Java.cast(a, exports.o.classClass)).getName())) break;
  }
}

function p(e, r) {
  void 0 === r && (r = null);
  var t = e;
  "string" == typeof t && (t = s(t)), c(t, r), u(t, r);
}

function d(r) {
  var t = {};
  if (t.method = parseBoolean(r.method, !0), t.thread = parseBoolean(r.thread, !1), 
  t.stack = parseBoolean(r.stack, !1), t.args = parseBoolean(r.args, !1), t.result = parseBoolean(r.result, t.args), 
  t.extras = {}, null != r.extras) for (var n in r.extras) t.extras[n] = r.extras[n];
  return function(r, n) {
    var a = {};
    for (var o in t.extras) a[o] = t.extras[o];
    !1 !== t.method && (a.class_name = r.$className, a.method_name = this.name, a.method_simple_name = this.methodName), 
    !1 !== t.thread && (a.thread_id = Process.getCurrentThreadId(), a.thread_name = exports.o.threadClass.currentThread().getName()), 
    !1 !== t.args && (a.args = pretty2Json(n)), !1 !== t.result && (a.result = null), 
    !1 === t.args && !1 === t.result || (a.error = null);
    try {
      var s = this(r, n);
      return !1 !== t.result && (a.result = pretty2Json(s)), s;
    } catch (e) {
      throw !1 === t.args && !1 === t.result || (a.error = pretty2Json(e)), e;
    } finally {
      !1 !== t.stack && (a.stack = pretty2Json(b())), e.event(a);
    }
  };
}

function f(e) {
  if (e instanceof Object && e.hasOwnProperty("class") && e.class instanceof Object) {
    var r = e.class;
    if (r.hasOwnProperty("getName") && r.hasOwnProperty("getDeclaredClasses") && r.hasOwnProperty("getDeclaredFields") && r.hasOwnProperty("getDeclaredMethods")) return !0;
  }
  return !1;
}

function g(e) {
  if (e instanceof Object && e.hasOwnProperty("class") && e.class instanceof Object) {
    var r = e.class;
    if (r.hasOwnProperty("isArray") && r.isArray()) return !0;
  }
  return !1;
}

function v(e, r) {
  var t = e;
  "string" == typeof t && (t = s(t));
  for (var n = [], a = Java.vm.getEnv(), o = 0; o < a.getArrayLength(r.$handle); o++) n.push(Java.cast(a.getObjectArrayElement(r.$handle, o), t));
  return n;
}

function h(e, r) {
  var t = e;
  "string" == typeof t && (t = s(t));
  var n = t.class.getEnumConstants();
  n instanceof Array || (n = v(t, n));
  for (var a = 0; a < n.length; a++) if (n[a].toString() === r) return n[a];
  throw new Error("Name of " + r + " does not match " + t);
}

function b(e) {
  void 0 === e && (e = void 0);
  for (var r = [], t = (e || exports.o.throwableClass.$new()).getStackTrace(), n = 0; n < t.length; n++) r.push(t[n]);
  return r;
}

exports.o = new r, exports.isSameObject = t, exports.getObjectHandle = n, exports.getClassName = a, 
exports.getClassMethod = o, exports.findClass = s, exports.hookMethod = i, exports.hookMethods = l, 
exports.hookAllConstructors = c, exports.hookAllMethods = u, exports.hookClass = p, 
exports.getEventImpl = d, exports.isJavaObject = f, exports.isJavaArray = g, exports.fromJavaArray = v, 
exports.getJavaEnumValue = h, exports.getStackTrace = b;

var y = null;

function m(r) {
  var t = exports.o.hashSetClass.$new(), n = function(t) {
    for (var n, a = r.entries(), o = function() {
      var a = n.value[0], o = n.value[1], i = null;
      try {
        i = s(a, t);
      } catch (e) {}
      null != i && (r.delete(a), o.forEach((function(r, t, n) {
        try {
          r(i);
        } catch (r) {
          e.w("Call JavaHelper.use callback error: " + r);
        }
      })));
    }; !(n = a.next()).done; ) o();
  }, a = exports.o.classClass, o = exports.o.classLoaderClass;
  i(a, "forName", [ "java.lang.String", "boolean", o ], (function(e, r) {
    var a = r[2];
    return null == a || t.contains(a) || (t.add(a), n(a)), this(e, r);
  })), i(o, "loadClass", [ "java.lang.String", "boolean" ], (function(e, r) {
    var a = e;
    return t.contains(a) || (t.add(a), n(a)), this(e, r);
  }));
}

function x(r, t) {
  var n = null;
  try {
    n = s(r);
  } catch (e) {
    var a;
    if (null == y && m(y = new Map), y.has(r)) void 0 !== (a = y.get(r)) && a.add(t); else (a = new Set).add(t), 
    y.set(r, a);
    return;
  }
  try {
    t(n);
  } catch (r) {
    e.w("Call JavaHelper.use callback error: " + r);
  }
}

function C() {
  e.w("Android Enable Webview Debugging"), ignoreError((function() {
    var r = s("android.webkit.WebView");
    l(r, "setWebContentsDebuggingEnabled", (function(t, n) {
      return e.d("".concat(r, ".setWebContentsDebuggingEnabled: ").concat(n[0])), n[0] = !0, 
      this(t, n);
    })), l(r, "loadUrl", (function(t, n) {
      return e.d("".concat(r, ".loadUrl: ").concat(n[0])), r.setWebContentsDebuggingEnabled(!0), 
      this(t, n);
    }));
  })), ignoreError((function() {
    var r = s("com.uc.webview.export.WebView");
    l(r, "setWebContentsDebuggingEnabled", (function(t, n) {
      return e.d("".concat(r, ".setWebContentsDebuggingEnabled: ").concat(n[0])), n[0] = !0, 
      this(t, n);
    })), l(r, "loadUrl", (function(t, n) {
      return e.d("".concat(r, ".loadUrl: ").concat(n[0])), r.setWebContentsDebuggingEnabled(!0), 
      this(t, n);
    }));
  }));
}

function k() {
  e.w("Android Bypass ssl pinning");
  var r = Java.use("java.util.Arrays");
  ignoreError((function() {
    return l("com.android.org.conscrypt.TrustManagerImpl", "checkServerTrusted", (function(t, n) {
      if (e.d("SSL bypassing " + this), "void" != this.returnType.type) return "pointer" == this.returnType.type && "java.util.List" == this.returnType.className ? r.asList(n[0]) : void 0;
    }));
  })), ignoreError((function() {
    return l("com.google.android.gms.org.conscrypt.Platform", "checkServerTrusted", (function(r, t) {
      e.d("SSL bypassing " + this);
    }));
  })), ignoreError((function() {
    return l("com.android.org.conscrypt.Platform", "checkServerTrusted", (function(r, t) {
      e.d("SSL bypassing " + this);
    }));
  })), ignoreError((function() {
    return l("okhttp3.CertificatePinner", "check", (function(r, t) {
      if (e.d("SSL bypassing " + this), "boolean" == this.returnType.type) return !0;
    }));
  })), ignoreError((function() {
    return l("okhttp3.CertificatePinner", "check$okhttp", (function(r, t) {
      e.d("SSL bypassing " + this);
    }));
  })), ignoreError((function() {
    return l("com.android.okhttp.CertificatePinner", "check", (function(r, t) {
      if (e.d("SSL bypassing " + this), "boolean" == this.returnType.type) return !0;
    }));
  })), ignoreError((function() {
    return l("com.android.okhttp.CertificatePinner", "check$okhttp", (function(r, t) {
      e.d("SSL bypassing " + this);
    }));
  })), ignoreError((function() {
    return l("com.android.org.conscrypt.TrustManagerImpl", "verifyChain", (function(r, t) {
      return e.d("SSL bypassing " + this), t[0];
    }));
  }));
}

function w(r) {
  e.w("choose classloder: " + r), Java.enumerateClassLoaders({
    onMatch: function(t) {
      try {
        null != t.findClass(r) && (e.i("choose classloader: " + t), Reflect.set(Java.classFactory, "loader", t));
      } catch (r) {
        e.e(pretty2Json(r));
      }
    },
    onComplete: function() {
      e.d("enumerate classLoaders complete");
    }
  });
}

function O(r, t, n) {
  void 0 === t && (t = void 0), void 0 === n && (n = void 0), r = null != r ? r.trim().toLowerCase() : "", 
  t = null != t ? t.trim().toLowerCase() : "", n = null != n ? n : {
    stack: !0,
    args: !0
  }, e.w("trace classes, include: " + r + ", exclude: " + t + ", options: " + JSON.stringify(n)), 
  Java.enumerateLoadedClasses({
    onMatch: function(e) {
      var a = e.toString().toLowerCase();
      a.indexOf(r) >= 0 && ("" == t || a.indexOf(t) < 0) && u(e, d(n));
    },
    onComplete: function() {
      e.d("enumerate classLoaders complete");
    }
  });
}

function S(e) {
  l("android.app.ContextImpl", "createAppContext", (function(r, t) {
    var n = this(r, t);
    return e(n), n;
  }));
}

function j(e) {
  l("android.app.LoadedApk", "makeApplication", (function(r, t) {
    var n = this(r, t);
    return e(n), n;
  }));
}

function J(e) {
  if (e.startsWith("[L") && e.endsWith(";")) return "".concat(e.substring(2, e.length - 1), "[]");
  if (e.startsWith("[")) switch (e.substring(1, 2)) {
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
  return e;
}

function P(e) {
  Object.defineProperties(e, {
    className: {
      configurable: !0,
      enumerable: !0,
      writable: !1,
      value: a(e.holder)
    },
    name: {
      configurable: !0,
      enumerable: !0,
      get: function() {
        var e = J(this.returnType.className), r = J(this.className) + "." + this.methodName, t = "";
        if (this.argumentTypes.length > 0) {
          t = J(this.argumentTypes[0].className);
          for (var n = 1; n < this.argumentTypes.length; n++) t = t + ", " + J(this.argumentTypes[n].className);
        }
        return e + " " + r + "(" + t + ")";
      }
    },
    toString: {
      configurable: !0,
      value: function() {
        return this.name;
      }
    }
  });
}

function E(r, t) {
  if (void 0 === t && (t = null), null != t) {
    var n = new Proxy(r, {
      apply: function(e, r, t) {
        var n = t[0], a = t[1];
        return e.apply(n, a);
      }
    }), a = isFunction(t) ? t : d(t);
    r.implementation = function() {
      return a.call(n, this, Array.prototype.slice.call(arguments));
    }, e.i("Hook method: " + r);
  } else r.implementation = null, e.i("Unhook method: " + r);
}

function L(e) {
  for (var r in exports.o.excludeHookPackages) if (0 == e.indexOf(exports.o.excludeHookPackages[r])) return !0;
  return !1;
}

function A(r) {
  try {
    var t = n(r);
    if (void 0 !== t) {
      for (var a = Java.cast(t, exports.o.throwableClass), o = [], s = 0, i = b(a); s < i.length; s++) {
        var l = i[s];
        o.push("    at ".concat(l));
      }
      return o.length > 0 ? "".concat(a, "\n").concat(o.join("\n")) : "".concat(a);
    }
  } catch (r) {
    e.d("getErrorStack error: ".concat(r));
  }
}

exports.use = x, exports.setWebviewDebuggingEnabled = C, exports.bypassSslPinning = k, 
exports.chooseClassLoader = w, exports.traceClasses = O, exports.runOnCreateContext = S, 
exports.runOnCreateApplication = j, exports.getErrorStack = A;

},{"./log":4}],4:[function(require,module,exports){
"use strict";

Object.defineProperty(exports, "__esModule", {
  value: !0
}), exports.exception = exports.event = exports.e = exports.w = exports.i = exports.d = exports.setLevel = exports.getLevel = exports.ERROR = exports.WARNING = exports.INFO = exports.DEBUG = void 0, 
exports.DEBUG = 1, exports.INFO = 2, exports.WARNING = 3, exports.ERROR = 4;

var e = exports.INFO, t = [], o = null;

function s() {
  return e;
}

function r(t) {
  e = t, n("Set log level: " + t);
}

function n(t, o) {
  e <= exports.DEBUG && v("log", {
    level: "debug",
    message: t
  }, o);
}

function p(t, o) {
  e <= exports.INFO && v("log", {
    level: "info",
    message: t
  }, o);
}

function l(t, o) {
  e <= exports.WARNING && v("log", {
    level: "warning",
    message: t
  }, o);
}

function x(t, o) {
  e <= exports.ERROR && v("log", {
    level: "error",
    message: t
  }, o);
}

function i(e, t) {
  v("msg", e, t);
}

function u(e, t) {
  v("error", {
    description: e,
    stack: t
  });
}

function v(e, s, r) {
  var n = {};
  n[e] = s, null == r ? (t.push(n), t.length >= 50 ? c() : null === o && (o = setTimeout(c, 50))) : (c(), 
  send({
    $events: [ n ]
  }, r));
}

function c() {
  if (null !== o && (clearTimeout(o), o = null), 0 !== t.length) {
    var e = t;
    t = [], send({
      $events: e
    });
  }
}

exports.getLevel = s, exports.setLevel = r, exports.d = n, exports.i = p, exports.w = l, 
exports.e = x, exports.event = i, exports.exception = u;

},{}],5:[function(require,module,exports){
"use strict";

Object.defineProperty(exports, "__esModule", {
  value: !0
}), exports.bypassSslPinning = exports.convert2ObjcObject = exports.getEventImpl = exports.hookMethods = exports.hookMethod = void 0;

var e = require("./log"), t = require("./c");

function r(e, t, r) {
  void 0 === r && (r = null);
  var n = e;
  if ("string" == typeof n && (n = ObjC.classes[n]), void 0 === n) throw Error('cannot find class "' + e + '"');
  var o = t;
  if ("string" == typeof o && (o = n[o]), void 0 === o) throw Error('cannot find method "' + t + '" in class "' + n + '"');
  s(n, o), l(o, r);
}

function n(e, t, r) {
  void 0 === r && (r = null);
  var n = e;
  if ("string" == typeof n && (n = ObjC.classes[n]), void 0 === n) throw Error('cannot find class "' + e + '"');
  for (var o = n.$ownMethods.length, i = 0; i < o; i++) {
    var a = n.$ownMethods[i];
    if (a.indexOf(t) >= 0) {
      var c = n[a];
      s(n, c), l(c, r);
    }
  }
}

function o(r) {
  var n = {};
  if (n.method = parseBoolean(r.method, !0), n.thread = parseBoolean(r.thread, !1), 
  n.stack = parseBoolean(r.stack, !1), n.symbol = parseBoolean(r.symbol, !0), n.backtracer = r.backtracer || "accurate", 
  n.args = parseBoolean(r.args, !1), n.result = parseBoolean(r.result, n.args), n.extras = {}, 
  null != r.extras) for (var o in r.extras) n.extras[o] = r.extras[o];
  return function(r, o) {
    var a = {};
    for (var s in n.extras) a[s] = n.extras[s];
    if (!1 !== n.method && (a.class_name = new ObjC.Object(r).$className, a.method_name = this.name, 
    a.method_simple_name = this.methodName), !1 !== n.thread && (a.thread_id = Process.getCurrentThreadId(), 
    a.thread_name = ObjC.classes.NSThread.currentThread().name().toString()), !1 !== n.args) {
      for (var l = [], c = 0; c < o.length; c++) l.push(i(o[c]));
      a.args = pretty2Json(l), a.result = null, a.error = null;
    }
    !1 !== n.result && (a.result = null), !1 === n.args && !1 === n.result || (a.error = null);
    try {
      var u = this(r, o);
      return !1 !== n.result && (a.result = pretty2Json(i(u))), u;
    } catch (e) {
      throw !1 === n.args && !1 === n.result || (a.error = pretty2Json(e)), e;
    } finally {
      if (!1 !== n.stack) {
        var d = a.stack = [], p = "accurate" === n.backtracer ? Backtracer.ACCURATE : Backtracer.FUZZY, f = Thread.backtrace(this.context, p);
        for (c = 0; c < f.length; c++) d.push(t.getDescFromAddress(f[c], !1 !== n.symbol));
      }
      e.event(a);
    }
  };
}

function i(e) {
  return e instanceof NativePointer || "object" == typeof e && e.hasOwnProperty("handle") ? new ObjC.Object(e) : e;
}

function a() {
  e.w("iOS Bypass ssl pinning");
  try {
    Module.ensureInitialized("libboringssl.dylib");
  } catch (t) {
    e.d("libboringssl.dylib module not loaded. Trying to manually load it."), Module.load("libboringssl.dylib");
  }
  var r = new NativeCallback((function(t, r) {
    return e.d("custom SSL context verify callback, returning SSL_VERIFY_NONE"), 0;
  }), "int", [ "pointer", "pointer" ]);
  try {
    t.hookFunction("libboringssl.dylib", "SSL_set_custom_verify", "void", [ "pointer", "int", "pointer" ], (function(t) {
      return e.d("SSL_set_custom_verify(), setting custom callback."), t[2] = r, this(t);
    }));
  } catch (n) {
    t.hookFunction("libboringssl.dylib", "SSL_CTX_set_custom_verify", "void", [ "pointer", "int", "pointer" ], (function(t) {
      return e.d("SSL_CTX_set_custom_verify(), setting custom callback."), t[2] = r, this(t);
    }));
  }
  t.hookFunction("libboringssl.dylib", "SSL_get_psk_identity", "pointer", [ "pointer" ], (function(t) {
    return e.d('SSL_get_psk_identity(), returning "fakePSKidentity"'), Memory.allocUtf8String("fakePSKidentity");
  }));
}

function s(e, t) {
  var r = t.origImplementation || t.implementation, n = e.toString(), o = ObjC.selectorAsString(t.selector), i = ObjC.classes.NSThread.hasOwnProperty(o);
  Object.defineProperties(t, {
    className: {
      configurable: !0,
      enumerable: !0,
      get: function() {
        return n;
      }
    },
    methodName: {
      configurable: !0,
      enumerable: !0,
      get: function() {
        return o;
      }
    },
    name: {
      configurable: !0,
      enumerable: !0,
      get: function() {
        return (i ? "+" : "-") + "[" + n + " " + o + "]";
      }
    },
    origImplementation: {
      configurable: !0,
      enumerable: !0,
      get: function() {
        return r;
      }
    },
    toString: {
      value: function() {
        return this.name;
      }
    }
  });
}

function l(t, r) {
  if (void 0 === r && (r = null), null != r) {
    var n = isFunction(r) ? r : o(r);
    t.implementation = ObjC.implement(t, (function() {
      var e = this, r = Array.prototype.slice.call(arguments), o = r.shift(), i = r.shift(), a = new Proxy(t, {
        get: function(t, r, n) {
          return r in e ? e[r] : t[r];
        },
        apply: function(e, t, r) {
          var n = r[0], o = r[1];
          return e.origImplementation.apply(null, [].concat(n, i, o));
        }
      });
      return n.call(a, o, r);
    })), e.i("Hook method: " + t);
  } else t.implementation = t.origImplementation, e.i("Unhook method: " + pretty2String(t));
}

exports.hookMethod = r, exports.hookMethods = n, exports.getEventImpl = o, exports.convert2ObjcObject = i, 
exports.bypassSslPinning = a;

},{"./c":2,"./log":4}]},{},[1])
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIm5vZGVfbW9kdWxlcy9icm93c2VyLXBhY2svX3ByZWx1ZGUuanMiLCJpbmRleC50cyIsImxpYi9jLnRzIiwibGliL2phdmEudHMiLCJsaWIvbG9nLnRzIiwibGliL29iamMudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7O0FDQUEsSUFBQSxJQUFBLFFBQUEsY0FDQSxJQUFBLFFBQUEsWUFDQSxJQUFBLFFBQUEsZUFDQSxJQUFBLFFBQUEsZUFNTSxJQUFhLFNBQUM7RUFDaEIsT0FBTztJQUNILElBQUksVUFBVSxTQUFTLEdBQUc7TUFFdEIsS0FEQSxJQUFJLElBQVUsY0FBYyxVQUFVLEtBQzdCLElBQUksR0FBRyxJQUFJLFVBQVUsUUFBUSxLQUNsQyxLQUFXO01BQ1gsS0FBVyxjQUFjLFVBQVU7TUFFdkMsRUFBRztXQUVILEVBQUc7QUFFWDtBQUNKOztBQUVBLFFBQVEsUUFBUSxFQUFXLEVBQUksRUFBRSxLQUFLLEtBQ3RDLFFBQVEsT0FBTyxFQUFXLEVBQUksRUFBRSxLQUFLLEtBQ3JDLFFBQVEsT0FBTyxFQUFXLEVBQUksRUFBRSxLQUFLO0FBQ3JDLFFBQVEsUUFBUSxFQUFXLEVBQUksRUFBRSxLQUFLLEtBQ3RDLFFBQVEsTUFBTSxFQUFXLEVBQUksRUFBRSxLQUFLLEtBR1MsUUFBekMsT0FBTyxrQ0FDUCxPQUFPLGdDQUErQixTQUFBO0VBQ2xDLElBQUksU0FBUTtFQUNaLElBQUksYUFBaUIsT0FBTztJQUN4QixJQUFNLElBQWEsRUFBTTtTQUNOLE1BQWYsTUFDQSxJQUFROztFQUdoQixJQUFJLEtBQUssV0FBVztJQUNoQixJQUFNLElBQVksRUFBSyxjQUFjO1NBQ25CLE1BQWQsV0FDYyxNQUFWLElBQ0EsS0FBUyxvQkFBQSxPQUFvQixLQUU3QixJQUFROztFQUlwQixFQUFJLFVBQVUsS0FBSyxHQUFPO0FBQzlCOztBQWlCSixJQUFBLElBQUE7RUFBQSxTQUFBLEtBb0JBO0VBQUEsT0FsQkksRUFBQSxVQUFBLE9BQUEsU0FBSyxH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QTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUNuSUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtB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
