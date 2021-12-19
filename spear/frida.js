(function(){function r(e,n,t){function o(i,f){if(!n[i]){if(!e[i]){var c="function"==typeof require&&require;if(!f&&c)return c(i,!0);if(u)return u(i,!0);var a=new Error("Cannot find module '"+i+"'");throw a.code="MODULE_NOT_FOUND",a}var p=n[i]={exports:{}};e[i][0].call(p.exports,function(r){var n=e[i][1][r];return o(n||r)},p,p.exports,r,e,n,t)}return n[i].exports}for(var u="function"==typeof require&&require,i=0;i<t.length;i++)o(t[i]);return o}return r})()({1:[function(require,module,exports){
"use strict";

Object.defineProperty(exports, "__esModule", {
  value: !0
});

var e = require("./lib/java"), r = require("./lib/objc");

globalThis.JavaHelper = new e.JavaHelper, globalThis.ObjCHelper = new r.ObjCHelper;

},{"./lib/java":3,"./lib/objc":4}],2:[function(require,module,exports){
"use strict";

Object.defineProperty(exports, "__esModule", {
  value: !0
}), exports.Base = void 0;

var t = function() {
  function t() {}
  return t.prototype.addMethod = function(t, r) {
    this[t + "_$_$_" + r.length] = r, this[t] = function() {
      var r = t + "_$_$_" + arguments.length;
      if (this.hasOwnProperty(r)) return this[r].apply(this, arguments);
      throw new Error("Argument count of " + arguments.length + " does not match " + t);
    };
  }, t.prototype.ignoreError = function(t, r) {
    void 0 === r && (r = void 0);
    try {
      return t();
    } catch (t) {
      return r;
    }
  }, t;
}();

exports.Base = t;

},{}],3:[function(require,module,exports){
"use strict";

var t = this && this.__extends || function() {
  var t = function(e, r) {
    return t = Object.setPrototypeOf || {
      __proto__: []
    } instanceof Array && function(t, e) {
      t.__proto__ = e;
    } || function(t, e) {
      for (var r in e) Object.prototype.hasOwnProperty.call(e, r) && (t[r] = e[r]);
    }, t(e, r);
  };
  return function(e, r) {
    if ("function" != typeof r && null !== r) throw new TypeError("Class extends value " + String(r) + " is not a constructor or null");
    function o() {
      this.constructor = e;
    }
    t(e, r), e.prototype = null === r ? Object.create(r) : (o.prototype = r.prototype, 
    new o);
  };
}();

Object.defineProperty(exports, "__esModule", {
  value: !0
}), exports.JavaHelper = void 0;

var e = require("./base"), r = function(e) {
  function r() {
    return null !== e && e.apply(this, arguments) || this;
  }
  return t(r, e), Object.defineProperty(r.prototype, "javaClass", {
    get: function() {
      return Java.use("java.lang.Class");
    },
    enumerable: !1,
    configurable: !0
  }), Object.defineProperty(r.prototype, "javaString", {
    get: function() {
      return Java.use("java.lang.String");
    },
    enumerable: !1,
    configurable: !0
  }), Object.defineProperty(r.prototype, "javaThrowable", {
    get: function() {
      return Java.use("java.lang.Throwable");
    },
    enumerable: !1,
    configurable: !0
  }), r.prototype.getClassName = function(t) {
    return t.$classWrapper.__name__;
  }, r.prototype.findClass = function(t, e) {
    if (void 0 === e && (e = void 0), void 0 === e) {
      var r = null, o = Java.enumerateClassLoadersSync();
      for (var n in o) try {
        var a = this.findClass(t, o[n]);
        if (null != a) return a;
      } catch (t) {
        null == r && (r = t);
      }
      throw r;
    }
    var i = Java.classFactory.loader;
    try {
      return Reflect.set(Java.classFactory, "loader", e), Java.use(t);
    } finally {
      Reflect.set(Java.classFactory, "loader", i);
    }
  }, r.prototype.$fixMethod = function(t) {
    t.toString = function() {
      var t = this.returnType.className, e = (this.holder.$className || this.holder.__name__) + "." + this.methodName, r = "";
      if (this.argumentTypes.length > 0) {
        r = this.argumentTypes[0].className;
        for (var o = 1; o < this.argumentTypes.length; o++) r = r + ", " + this.argumentTypes[o].className;
      }
      return t + " " + e + "(" + r + ")";
    };
  }, r.prototype.$hookMethod = function(t, e) {
    void 0 === e && (e = null), null != e ? (t.implementation = function() {
      return e.call(t, this, arguments);
    }, this.$fixMethod(t), send("Hook method: " + t)) : (t.implementation = null, this.$fixMethod(t), 
    send("Unhook method: " + t));
  }, r.prototype.hookMethod = function(t, e, r, o) {
    void 0 === o && (o = null);
    var n = e;
    if ("string" == typeof n) {
      var a = t;
      if ("string" == typeof a && (a = this.findClass(a)), n = a[n], null != r) {
        var i = r;
        for (var s in i) "string" != typeof i[s] && (i[s] = this.getClassName(i[s]));
        n = n.overload.apply(n, i);
      }
    }
    this.$hookMethod(n, o);
  }, r.prototype.hookMethods = function(t, e, r) {
    void 0 === r && (r = null);
    var o = t;
    "string" == typeof o && (o = this.findClass(o));
    for (var n = o[e].overloads, a = 0; a < n.length; a++) void 0 !== n[a].returnType && void 0 !== n[a].returnType.className && this.$hookMethod(n[a], r);
  }, r.prototype.hookClass = function(t, e) {
    void 0 === e && (e = null);
    var r = t;
    "string" == typeof r && (r = this.findClass(r)), this.hookMethods(r, "$init", e);
    for (var o = [], n = r.class; null != n && "java.lang.Object" !== n.getName(); ) {
      for (var a = n.getDeclaredMethods(), i = 0; i < a.length; i++) {
        var s = a[i].getName();
        o.indexOf(s) < 0 && (o.push(s), this.hookMethods(r, s, e));
      }
      n = Java.cast(n.getSuperclass(), this.javaClass);
    }
  }, r.prototype.callMethod = function(t, e) {
    var r = this.getStackTrace()[0].getMethodName();
    return "<init>" === r && (r = "$init"), Reflect.get(t, r).apply(t, e);
  }, r.prototype.getHookImpl = function(t) {
    var e = this, r = t.printStack || !1, o = t.printArgs || !1;
    return function(t, n) {
      var a = {}, i = this.apply(t, n);
      return !1 !== r && (a = Object.assign(a, e.$makeStackObject(this))), !1 !== o && (a = Object.assign(a, e.$makeArgsObject(n, i, this))), 
      0 !== Object.keys(a).length && send(a), i;
    };
  }, r.prototype.fromJavaArray = function(t, e) {
    var r = t;
    "string" == typeof r && (r = this.findClass(r));
    for (var o = [], n = Java.vm.getEnv(), a = 0; a < n.getArrayLength(e.$handle); a++) o.push(Java.cast(n.getObjectArrayElement(e.$handle, a), r));
    return o;
  }, r.prototype.getEnumValue = function(t, e) {
    var r = t;
    "string" == typeof r && (r = this.findClass(r));
    var o = r.class.getEnumConstants();
    o instanceof Array || (o = this.fromJavaArray(r, o));
    for (var n = 0; n < o.length; n++) if (o[n].toString() === e) return o[n];
    throw new Error("Name of " + e + " does not match " + r);
  }, r.prototype.getStackTrace = function() {
    return this.javaThrowable.$new().getStackTrace();
  }, r.prototype.$makeStackObject = function(t, e) {
    void 0 === e && (e = void 0), void 0 === e && (e = this.getStackTrace());
    for (var r = "Stack: " + t, o = 0; o < e.length; o++) r += "\n    at " + this.toString(e[o]);
    return {
      stack: r
    };
  }, r.prototype.printStack = function(t) {
    void 0 === t && (t = void 0);
    var e = this.getStackTrace();
    null == t && (t = e[0]), send(this.$makeStackObject(t, e));
  }, r.prototype.toString = function(t) {
    if (void 0 === t || null == t || !(t instanceof Object)) return t;
    if (Array.isArray(t)) {
      for (var e = [], r = 0; r < t.length; r++) e.push(this.toString(t[r]));
      return "[" + e.toString() + "]";
    }
    return this.ignoreError((function() {
      return t.toString();
    }), void 0);
  }, r.prototype.$makeArgsObject = function(t, e, r) {
    for (var o = "Arguments: " + r, n = 0; n < t.length; n++) o += "\n    Arguments[" + n + "]: " + this.toString(t[n]);
    return void 0 !== e && (o += "\n    Return: " + this.toString(e)), {
      arguments: o
    };
  }, r.prototype.printArguments = function(t, e, r) {
    void 0 === r && (r = void 0), void 0 === r && (r = this.getStackTrace()[0]), send(this.$makeArgsObject(t, e, r));
  }, r;
}(e.Base);

exports.JavaHelper = r;

},{"./base":2}],4:[function(require,module,exports){
"use strict";

var t = this && this.__extends || function() {
  var t = function(e, r) {
    return t = Object.setPrototypeOf || {
      __proto__: []
    } instanceof Array && function(t, e) {
      t.__proto__ = e;
    } || function(t, e) {
      for (var r in e) Object.prototype.hasOwnProperty.call(e, r) && (t[r] = e[r]);
    }, t(e, r);
  };
  return function(e, r) {
    if ("function" != typeof r && null !== r) throw new TypeError("Class extends value " + String(r) + " is not a constructor or null");
    function o() {
      this.constructor = e;
    }
    t(e, r), e.prototype = null === r ? Object.create(r) : (o.prototype = r.prototype, 
    new o);
  };
}();

Object.defineProperty(exports, "__esModule", {
  value: !0
}), exports.ObjCHelper = void 0;

var e = require("./base"), r = function(e) {
  function r() {
    return e.call(this) || this;
  }
  return t(r, e), r;
}(e.Base);

exports.ObjCHelper = r;

},{"./base":2}]},{},[1])
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIm5vZGVfbW9kdWxlcy9icm93c2VyLXBhY2svX3ByZWx1ZGUuanMiLCJpbmRleC50cyIsImxpYi9iYXNlLnRzIiwibGliL2phdmEudHMiLCJsaWIvb2JqYy50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7OztBQ0FBLElBQUEsSUFBQSxRQUFBLGVBQ0EsSUFBQSxRQUFBOztBQUVBLFdBQVcsYUFBYSxJQUFJLEVBQUEsWUFDNUIsV0FBVyxhQUFhLElBQUksRUFBQTs7Ozs7Ozs7O0FDSjVCLElBQUEsSUFBQTtFQUFBLFNBQUE7RUFrQ0EsT0EzQkksRUFBQSxVQUFBLFlBQUEsU0FBVSxHQUFjO0lBQ3BCLEtBQUssSUFBTyxVQUFVLEVBQUcsVUFBVSxHQUNuQyxLQUFLLEtBQVE7TUFDVCxJQUFJLElBQU8sSUFBTyxVQUFVLFVBQVU7TUFDdEMsSUFBSSxLQUFLLGVBQWUsSUFDcEIsT0FBTyxLQUFLLEdBQU0sTUFBTSxNQUFNO01BRTlCLE1BQU0sSUFBSSxNQUFNLHVCQUF1QixVQUFVLFNBQVMscUJBQXFCOztLQVczRixFQUFBLFVBQUEsY0FBQSxTQUFZLEdBQTZCO1NBQUEsTUFBQSxNQUFBLFNBQUE7SUFDckM7TUFFSSxPQUFPO01BQ1QsT0FBTztNQUNMLE9BQU87O0tBSW5CO0NBbENBOztBQUFhLFFBQUEsT0FBQTs7O0FDQWI7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQ25LQSxJQUFBLElBQUEsUUFBQSxXQUVBLElBQUEsU0FBQTtFQUVJLFNBQUE7V0FDSSxFQUFBLEtBQUEsU0FBTzs7RUFFZixPQUxnQyxFQUFBLEdBQUEsSUFLaEM7Q0FMQSxDQUFnQyxFQUFBOztBQUFuQixRQUFBLGFBQUEiLCJmaWxlIjoiZ2VuZXJhdGVkLmpzIiwic291cmNlUm9vdCI6IiJ9
