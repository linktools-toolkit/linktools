(()=>{var x=Object.defineProperty;var T=(e,t)=>{for(var r in t)x(e,r,{get:t[r],enumerable:!0})};var s=[],l=[],U="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";for(let e=0,t=U.length;e<t;++e)s[e]=U[e],l[U.charCodeAt(e)]=e;l[45]=62;l[95]=63;function O(e){let t=e.length;if(t%4>0)throw new Error("Invalid string. Length must be a multiple of 4");let r=e.indexOf("=");r===-1&&(r=t);let n=r===t?0:4-r%4;return[r,n]}function F(e,t,r){return(t+r)*3/4-r}function D(t){let e=O(t),r=e[0],n=e[1],i=new Uint8Array(F(t,r,n)),a=0,o=n>0?r-4:r,s;for(s=0;s<o;s+=4){let e=l[t.charCodeAt(s)]<<18|l[t.charCodeAt(s+1)]<<12|l[t.charCodeAt(s+2)]<<6|l[t.charCodeAt(s+3)];i[a++]=e>>16&255,i[a++]=e>>8&255,i[a++]=e&255}if(n===2){let e=l[t.charCodeAt(s)]<<2|l[t.charCodeAt(s+1)]>>4;i[a++]=e&255}if(n===1){let e=l[t.charCodeAt(s)]<<10|l[t.charCodeAt(s+1)]<<4|l[t.charCodeAt(s+2)]>>2;i[a++]=e>>8&255,i[a++]=e&255}return i}function z(e){return s[e>>18&63]+s[e>>12&63]+s[e>>6&63]+s[e&63]}function $(r,e,n){let i=[];for(let t=e;t<n;t+=3){let e=(r[t]<<16&16711680)+(r[t+1]<<8&65280)+(r[t+2]&255);i.push(z(e))}return i.join("")}function V(r){let n=r.length,i=n%3,a=[],o=16383;for(let e=0,t=n-i;e<t;e+=o)a.push($(r,e,e+o>t?t:e+o));if(i===1){let e=r[n-1];a.push(s[e>>2]+s[e<<4&63]+"==")}else if(i===2){let e=(r[n-2]<<8)+r[n-1];a.push(s[e>>10]+s[e>>4&63]+s[e<<2&63]+"=")}return a.join("")}function J(e,t,r,n,i){let a,o,s=i*8-n-1,l=(1<<s)-1,d=l>>1,c=-7,u=r?i-1:0,h=r?-1:1,p=e[t+u];for(u+=h,a=p&(1<<-c)-1,p>>=-c,c+=s;c>0;)a=a*256+e[t+u],u+=h,c-=8;for(o=a&(1<<-c)-1,a>>=-c,c+=n;c>0;)o=o*256+e[t+u],u+=h,c-=8;if(a===0)a=1-d;else{if(a===l)return o?NaN:(p?-1:1)*(1/0);o=o+Math.pow(2,n),a=a-d}return(p?-1:1)*o*Math.pow(2,a-n)}function B(e,t,r,n,i,a){let o,s,l,d=a*8-i-1,c=(1<<d)-1,u=c>>1,h=i===23?Math.pow(2,-24)-Math.pow(2,-77):0,p=n?0:a-1,f=n?1:-1,_=t<0||t===0&&1/t<0?1:0;for(t=Math.abs(t),isNaN(t)||t===1/0?(s=isNaN(t)?1:0,o=c):(o=Math.floor(Math.log(t)/Math.LN2),t*(l=Math.pow(2,-o))<1&&(o--,l*=2),o+u>=1?t+=h/l:t+=h*Math.pow(2,1-u),t*l>=2&&(o++,l/=2),o+u>=c?(s=0,o=c):o+u>=1?(s=(t*l-1)*Math.pow(2,i),o=o+u):(s=t*Math.pow(2,u-1)*Math.pow(2,i),o=0));i>=8;)e[r+p]=s&255,p+=f,s/=256,i-=8;for(o=o<<i|s,d+=i;d>0;)e[r+p]=o&255,p+=f,o/=256,d-=8;e[r+p-f]|=_*128}var G={INSPECT_MAX_BYTES:50},Z=2147483647;Q.TYPED_ARRAY_SUPPORT=!0;Object.defineProperty(Q.prototype,"parent",{enumerable:!0,get:function(){if(Q.isBuffer(this))return this.buffer}});Object.defineProperty(Q.prototype,"offset",{enumerable:!0,get:function(){if(Q.isBuffer(this))return this.byteOffset}});function a(e){if(e>Z)throw new RangeError('The value "'+e+'" is invalid for option "size"');let t=new Uint8Array(e);return Object.setPrototypeOf(t,Q.prototype),t}function Q(e,t,r){if(typeof e=="number"){if(typeof t=="string")throw new TypeError('The "string" argument must be of type string. Received type number');return K(e)}return H(e,t,r)}Q.poolSize=8192;function H(e,t,r){if(typeof e=="string")return Y(e,t);if(ArrayBuffer.isView(e))return te(e);if(e==null)throw new TypeError("The first argument must be one of type string, Buffer, ArrayBuffer, Array, or Array-like Object. Received type "+typeof e);if(e instanceof ArrayBuffer||e&&e.buffer instanceof ArrayBuffer||e instanceof SharedArrayBuffer||e&&e.buffer instanceof SharedArrayBuffer)return re(e,t,r);if(typeof e=="number")throw new TypeError('The "value" argument must not be of type number. Received type number');let n=e.valueOf&&e.valueOf();if(n!=null&&n!==e)return Q.from(n,t,r);let i=ne(e);if(i)return i;if(typeof Symbol<"u"&&Symbol.toPrimitive!=null&&typeof e[Symbol.toPrimitive]=="function")return Q.from(e[Symbol.toPrimitive]("string"),t,r);throw new TypeError("The first argument must be one of type string, Buffer, ArrayBuffer, Array, or Array-like Object. Received type "+typeof e)}Q.from=function(e,t,r){return H(e,t,r)};Object.setPrototypeOf(Q.prototype,Uint8Array.prototype);Object.setPrototypeOf(Q,Uint8Array);function q(e){if(typeof e!="number")throw new TypeError('"size" argument must be of type number');if(e<0)throw new RangeError('The value "'+e+'" is invalid for option "size"')}function W(e,t,r){return q(e),e<=0?a(e):t!==void 0?typeof r=="string"?a(e).fill(t,r):a(e).fill(t):a(e)}Q.alloc=function(e,t,r){return W(e,t,r)};function K(e){return q(e),a(e<0?0:ie(e)|0)}Q.allocUnsafe=function(e){return K(e)};Q.allocUnsafeSlow=function(e){return K(e)};function Y(e,t){if((typeof t!="string"||t==="")&&(t="utf8"),!Q.isEncoding(t))throw new TypeError("Unknown encoding: "+t);let r=ae(e,t)|0,n=a(r),i=n.write(e,t);return i!==r&&(n=n.slice(0,i)),n}function ee(t){let r=t.length<0?0:ie(t.length)|0,n=a(r);for(let e=0;e<r;e+=1)n[e]=t[e]&255;return n}function te(t){if(t instanceof Uint8Array){let e=new Uint8Array(t);return re(e.buffer,e.byteOffset,e.byteLength)}return ee(t)}function re(e,t,r){if(t<0||e.byteLength<t)throw new RangeError('"offset" is outside of buffer bounds');if(e.byteLength<t+(r||0))throw new RangeError('"length" is outside of buffer bounds');let n;return t===void 0&&r===void 0?n=new Uint8Array(e):r===void 0?n=new Uint8Array(e,t):n=new Uint8Array(e,t,r),Object.setPrototypeOf(n,Q.prototype),n}function ne(r){if(Q.isBuffer(r)){let e=ie(r.length)|0,t=a(e);return t.length===0||r.copy(t,0,0,e),t}if(r.length!==void 0)return typeof r.length!="number"||Number.isNaN(r.length)?a(0):ee(r);if(r.type==="Buffer"&&Array.isArray(r.data))return ee(r.data)}function ie(e){if(e>=Z)throw new RangeError("Attempt to allocate Buffer larger than maximum size: 0x"+Z.toString(16)+" bytes");return e|0}Q.isBuffer=function(e){return e!=null&&e._isBuffer===!0&&e!==Q.prototype};Q.compare=function(r,n){if(r instanceof Uint8Array&&(r=Q.from(r,r.offset,r.byteLength)),n instanceof Uint8Array&&(n=Q.from(n,n.offset,n.byteLength)),!Q.isBuffer(r)||!Q.isBuffer(n))throw new TypeError('The "buf1", "buf2" arguments must be one of type Buffer or Uint8Array');if(r===n)return 0;let i=r.length,a=n.length;for(let e=0,t=Math.min(i,a);e<t;++e)if(r[e]!==n[e]){i=r[e],a=n[e];break}return i<a?-1:a<i?1:0};Q.isEncoding=function(e){switch(String(e).toLowerCase()){case"hex":case"utf8":case"utf-8":case"ascii":case"latin1":case"binary":case"base64":case"ucs2":case"ucs-2":case"utf16le":case"utf-16le":return!0;default:return!1}};Q.concat=function(t,e){if(!Array.isArray(t))throw new TypeError('"list" argument must be an Array of Buffers');if(t.length===0)return Q.alloc(0);let r;if(e===void 0)for(e=0,r=0;r<t.length;++r)e+=t[r].length;let n=Q.allocUnsafe(e),i=0;for(r=0;r<t.length;++r){let e=t[r];if(e instanceof Uint8Array)i+e.length>n.length?(Q.isBuffer(e)||(e=Q.from(e.buffer,e.byteOffset,e.byteLength)),e.copy(n,i)):Uint8Array.prototype.set.call(n,e,i);else if(Q.isBuffer(e))e.copy(n,i);else throw new TypeError('"list" argument must be an Array of Buffers');i+=e.length}return n};function ae(e,t){if(Q.isBuffer(e))return e.length;if(ArrayBuffer.isView(e)||e instanceof ArrayBuffer)return e.byteLength;if(typeof e!="string")throw new TypeError('The "string" argument must be one of type string, Buffer, or ArrayBuffer. Received type '+typeof e);let r=e.length,n=arguments.length>2&&arguments[2]===!0;if(!n&&r===0)return 0;let i=!1;for(;;)switch(t){case"ascii":case"latin1":case"binary":return r;case"utf8":case"utf-8":return Ue(e).length;case"ucs2":case"ucs-2":case"utf16le":case"utf-16le":return r*2;case"hex":return r>>>1;case"base64":return De(e).length;default:if(i)return n?-1:Ue(e).length;t=(""+t).toLowerCase(),i=!0}}Q.byteLength=ae;function oe(e,t,r){let n=!1;if((t===void 0||t<0)&&(t=0),t>this.length||((r===void 0||r>this.length)&&(r=this.length),r<=0)||(r>>>=0,t>>>=0,r<=t))return"";for(e||(e="utf8");;)switch(e){case"hex":return ye(this,t,r);case"utf8":case"utf-8":return _e(this,t,r);case"ascii":return be(this,t,r);case"latin1":case"binary":return ve(this,t,r);case"base64":return fe(this,t,r);case"ucs2":case"ucs-2":case"utf16le":case"utf-16le":return we(this,t,r);default:if(n)throw new TypeError("Unknown encoding: "+e);e=(e+"").toLowerCase(),n=!0}}Q.prototype._isBuffer=!0;function r(e,t,r){let n=e[t];e[t]=e[r],e[r]=n}Q.prototype.swap16=function(){let t=this.length;if(t%2!==0)throw new RangeError("Buffer size must be a multiple of 16-bits");for(let e=0;e<t;e+=2)r(this,e,e+1);return this};Q.prototype.swap32=function(){let t=this.length;if(t%4!==0)throw new RangeError("Buffer size must be a multiple of 32-bits");for(let e=0;e<t;e+=4)r(this,e,e+3),r(this,e+1,e+2);return this};Q.prototype.swap64=function(){let t=this.length;if(t%8!==0)throw new RangeError("Buffer size must be a multiple of 64-bits");for(let e=0;e<t;e+=8)r(this,e,e+7),r(this,e+1,e+6),r(this,e+2,e+5),r(this,e+3,e+4);return this};Q.prototype.toString=function(){let e=this.length;return e===0?"":arguments.length===0?_e(this,0,e):oe.apply(this,arguments)};Q.prototype.toLocaleString=Q.prototype.toString;Q.prototype.equals=function(e){if(!Q.isBuffer(e))throw new TypeError("Argument must be a Buffer");return this===e?!0:Q.compare(this,e)===0};Q.prototype.inspect=function(){let e="",t=G.INSPECT_MAX_BYTES;return e=this.toString("hex",0,t).replace(/(.{2})/g,"$1 ").trim(),this.length>t&&(e+=" ... "),"<Buffer "+e+">"};Q.prototype[Symbol.for("nodejs.util.inspect.custom")]=Q.prototype.inspect;Q.prototype.compare=function(e,t,r,n,i){if(e instanceof Uint8Array&&(e=Q.from(e,e.offset,e.byteLength)),!Q.isBuffer(e))throw new TypeError('The "target" argument must be one of type Buffer or Uint8Array. Received type '+typeof e);if(t===void 0&&(t=0),r===void 0&&(r=e?e.length:0),n===void 0&&(n=0),i===void 0&&(i=this.length),t<0||r>e.length||n<0||i>this.length)throw new RangeError("out of range index");if(n>=i&&t>=r)return 0;if(n>=i)return-1;if(t>=r)return 1;if(t>>>=0,r>>>=0,n>>>=0,i>>>=0,this===e)return 0;let a=i-n,o=r-t,s=Math.min(a,o),l=this.slice(n,i),d=e.slice(t,r);for(let e=0;e<s;++e)if(l[e]!==d[e]){a=l[e],o=d[e];break}return a<o?-1:o<a?1:0};function se(e,t,r,n,i){if(e.length===0)return-1;if(typeof r=="string"?(n=r,r=0):r>2147483647?r=2147483647:r<-2147483648&&(r=-2147483648),r=+r,Number.isNaN(r)&&(r=i?0:e.length-1),r<0&&(r=e.length+r),r>=e.length){if(i)return-1;r=e.length-1}else if(r<0)if(i)r=0;else return-1;if(typeof t=="string"&&(t=Q.from(t,n)),Q.isBuffer(t))return t.length===0?-1:le(e,t,r,n,i);if(typeof t=="number")return t=t&255,typeof Uint8Array.prototype.indexOf=="function"?i?Uint8Array.prototype.indexOf.call(e,t,r):Uint8Array.prototype.lastIndexOf.call(e,t,r):le(e,[t],r,n,i);throw new TypeError("val must be string, number or Buffer")}function le(r,n,t,e,i){let a=1,o=r.length,s=n.length;if(e!==void 0&&(e=String(e).toLowerCase(),e==="ucs2"||e==="ucs-2"||e==="utf16le"||e==="utf-16le")){if(r.length<2||n.length<2)return-1;a=2,o/=2,s/=2,t/=2}function l(e,t){return a===1?e[t]:e.readUInt16BE(t*a)}let d;if(i){let e=-1;for(d=t;d<o;d++)if(l(r,d)===l(n,e===-1?0:d-e)){if(e===-1&&(e=d),d-e+1===s)return e*a}else e!==-1&&(d-=d-e),e=-1}else for(t+s>o&&(t=o-s),d=t;d>=0;d--){let t=!0;for(let e=0;e<s;e++)if(l(r,d+e)!==l(n,e)){t=!1;break}if(t)return d}return-1}Q.prototype.includes=function(e,t,r){return this.indexOf(e,t,r)!==-1};Q.prototype.indexOf=function(e,t,r){return se(this,e,t,r,!0)};Q.prototype.lastIndexOf=function(e,t,r){return se(this,e,t,r,!1)};function de(t,r,n,e){n=Number(n)||0;let i=t.length-n;e?(e=Number(e),e>i&&(e=i)):e=i;let a=r.length;e>a/2&&(e=a/2);let o;for(o=0;o<e;++o){let e=parseInt(r.substr(o*2,2),16);if(Number.isNaN(e))return o;t[n+o]=e}return o}function ce(e,t,r,n){return ze(Ue(t,e.length-r),e,r,n)}function ue(e,t,r,n){return ze(Oe(t),e,r,n)}function he(e,t,r,n){return ze(De(t),e,r,n)}function pe(e,t,r,n){return ze(Fe(t,e.length-r),e,r,n)}Q.prototype.write=function(e,t,r,n){if(t===void 0)n="utf8",r=this.length,t=0;else if(r===void 0&&typeof t=="string")n=t,r=this.length,t=0;else if(isFinite(t))t=t>>>0,isFinite(r)?(r=r>>>0,n===void 0&&(n="utf8")):(n=r,r=void 0);else throw new Error("Buffer.write(string, encoding, offset[, length]) is no longer supported");let i=this.length-t;if((r===void 0||r>i)&&(r=i),e.length>0&&(r<0||t<0)||t>this.length)throw new RangeError("Attempt to write outside buffer bounds");n||(n="utf8");let a=!1;for(;;)switch(n){case"hex":return de(this,e,t,r);case"utf8":case"utf-8":return ce(this,e,t,r);case"ascii":case"latin1":case"binary":return ue(this,e,t,r);case"base64":return he(this,e,t,r);case"ucs2":case"ucs-2":case"utf16le":case"utf-16le":return pe(this,e,t,r);default:if(a)throw new TypeError("Unknown encoding: "+n);n=(""+n).toLowerCase(),a=!0}};Q.prototype.toJSON=function(){return{type:"Buffer",data:Array.prototype.slice.call(this._arr||this,0)}};function fe(e,t,r){return t===0&&r===e.length?V(e):V(e.slice(t,r))}function _e(s,e,t){t=Math.min(s.length,t);let r=[],l=e;for(;l<t;){let i=s[l],a=null,o=i>239?4:i>223?3:i>191?2:1;if(l+o<=t){let e,t,r,n;switch(o){case 1:i<128&&(a=i);break;case 2:e=s[l+1],(e&192)===128&&(n=(i&31)<<6|e&63,n>127&&(a=n));break;case 3:e=s[l+1],t=s[l+2],(e&192)===128&&(t&192)===128&&(n=(i&15)<<12|(e&63)<<6|t&63,n>2047&&(n<55296||n>57343)&&(a=n));break;case 4:e=s[l+1],t=s[l+2],r=s[l+3],(e&192)===128&&(t&192)===128&&(r&192)===128&&(n=(i&15)<<18|(e&63)<<12|(t&63)<<6|r&63,n>65535&&n<1114112&&(a=n))}}a===null?(a=65533,o=1):a>65535&&(a-=65536,r.push(a>>>10&1023|55296),a=56320|a&1023),r.push(a),l+=o}return ge(r)}var me=4096;function ge(e){let t=e.length;if(t<=me)return String.fromCharCode.apply(String,e);let r="",n=0;for(;n<t;)r+=String.fromCharCode.apply(String,e.slice(n,n+=me));return r}function be(t,r,n){let i="";n=Math.min(t.length,n);for(let e=r;e<n;++e)i+=String.fromCharCode(t[e]&127);return i}function ve(t,r,n){let i="";n=Math.min(t.length,n);for(let e=r;e<n;++e)i+=String.fromCharCode(t[e]);return i}function ye(t,r,n){let e=t.length;(!r||r<0)&&(r=0),(!n||n<0||n>e)&&(n=e);let i="";for(let e=r;e<n;++e)i+=$e[t[e]];return i}function we(e,t,r){let n=e.slice(t,r),i="";for(let e=0;e<n.length-1;e+=2)i+=String.fromCharCode(n[e]+n[e+1]*256);return i}Q.prototype.slice=function(e,t){let r=this.length;e=~~e,t=t===void 0?r:~~t,e<0?(e+=r,e<0&&(e=0)):e>r&&(e=r),t<0?(t+=r,t<0&&(t=0)):t>r&&(t=r),t<e&&(t=e);let n=this.subarray(e,t);return Object.setPrototypeOf(n,Q.prototype),n};function o(e,t,r){if(e%1!==0||e<0)throw new RangeError("offset is not uint");if(e+t>r)throw new RangeError("Trying to access beyond buffer length")}Q.prototype.readUintLE=Q.prototype.readUIntLE=function(e,t,r){e=e>>>0,t=t>>>0,r||o(e,t,this.length);let n=this[e],i=1,a=0;for(;++a<t&&(i*=256);)n+=this[e+a]*i;return n};Q.prototype.readUintBE=Q.prototype.readUIntBE=function(e,t,r){e=e>>>0,t=t>>>0,r||o(e,t,this.length);let n=this[e+--t],i=1;for(;t>0&&(i*=256);)n+=this[e+--t]*i;return n};Q.prototype.readUint8=Q.prototype.readUInt8=function(e,t){return e=e>>>0,t||o(e,1,this.length),this[e]};Q.prototype.readUint16LE=Q.prototype.readUInt16LE=function(e,t){return e=e>>>0,t||o(e,2,this.length),this[e]|this[e+1]<<8};Q.prototype.readUint16BE=Q.prototype.readUInt16BE=function(e,t){return e=e>>>0,t||o(e,2,this.length),this[e]<<8|this[e+1]};Q.prototype.readUint32LE=Q.prototype.readUInt32LE=function(e,t){return e=e>>>0,t||o(e,4,this.length),(this[e]|this[e+1]<<8|this[e+2]<<16)+this[e+3]*16777216};Q.prototype.readUint32BE=Q.prototype.readUInt32BE=function(e,t){return e=e>>>0,t||o(e,4,this.length),this[e]*16777216+(this[e+1]<<16|this[e+2]<<8|this[e+3])};Q.prototype.readBigUInt64LE=function(e){e=e>>>0,Re(e,"offset");let t=this[e],r=this[e+7];(t===void 0||r===void 0)&&Ae(e,this.length-8);let n=t+this[++e]*2**8+this[++e]*2**16+this[++e]*2**24,i=this[++e]+this[++e]*2**8+this[++e]*2**16+r*2**24;return BigInt(n)+(BigInt(i)<<BigInt(32))};Q.prototype.readBigUInt64BE=function(e){e=e>>>0,Re(e,"offset");let t=this[e],r=this[e+7];(t===void 0||r===void 0)&&Ae(e,this.length-8);let n=t*2**24+this[++e]*2**16+this[++e]*2**8+this[++e],i=this[++e]*2**24+this[++e]*2**16+this[++e]*2**8+r;return(BigInt(n)<<BigInt(32))+BigInt(i)};Q.prototype.readIntLE=function(e,t,r){e=e>>>0,t=t>>>0,r||o(e,t,this.length);let n=this[e],i=1,a=0;for(;++a<t&&(i*=256);)n+=this[e+a]*i;return i*=128,n>=i&&(n-=Math.pow(2,8*t)),n};Q.prototype.readIntBE=function(e,t,r){e=e>>>0,t=t>>>0,r||o(e,t,this.length);let n=t,i=1,a=this[e+--n];for(;n>0&&(i*=256);)a+=this[e+--n]*i;return i*=128,a>=i&&(a-=Math.pow(2,8*t)),a};Q.prototype.readInt8=function(e,t){return e=e>>>0,t||o(e,1,this.length),this[e]&128?(255-this[e]+1)*-1:this[e]};Q.prototype.readInt16LE=function(e,t){e=e>>>0,t||o(e,2,this.length);let r=this[e]|this[e+1]<<8;return r&32768?r|4294901760:r};Q.prototype.readInt16BE=function(e,t){e=e>>>0,t||o(e,2,this.length);let r=this[e+1]|this[e]<<8;return r&32768?r|4294901760:r};Q.prototype.readInt32LE=function(e,t){return e=e>>>0,t||o(e,4,this.length),this[e]|this[e+1]<<8|this[e+2]<<16|this[e+3]<<24};Q.prototype.readInt32BE=function(e,t){return e=e>>>0,t||o(e,4,this.length),this[e]<<24|this[e+1]<<16|this[e+2]<<8|this[e+3]};Q.prototype.readBigInt64LE=function(e){e=e>>>0,Re(e,"offset");let t=this[e],r=this[e+7];(t===void 0||r===void 0)&&Ae(e,this.length-8);let n=this[e+4]+this[e+5]*2**8+this[e+6]*2**16+(r<<24);return(BigInt(n)<<BigInt(32))+BigInt(t+this[++e]*2**8+this[++e]*2**16+this[++e]*2**24)};Q.prototype.readBigInt64BE=function(e){e=e>>>0,Re(e,"offset");let t=this[e],r=this[e+7];(t===void 0||r===void 0)&&Ae(e,this.length-8);let n=(t<<24)+this[++e]*2**16+this[++e]*2**8+this[++e];return(BigInt(n)<<BigInt(32))+BigInt(this[++e]*2**24+this[++e]*2**16+this[++e]*2**8+r)};Q.prototype.readFloatLE=function(e,t){return e=e>>>0,t||o(e,4,this.length),J(this,e,!0,23,4)};Q.prototype.readFloatBE=function(e,t){return e=e>>>0,t||o(e,4,this.length),J(this,e,!1,23,4)};Q.prototype.readDoubleLE=function(e,t){return e=e>>>0,t||o(e,8,this.length),J(this,e,!0,52,8)};Q.prototype.readDoubleBE=function(e,t){return e=e>>>0,t||o(e,8,this.length),J(this,e,!1,52,8)};function d(e,t,r,n,i,a){if(!Q.isBuffer(e))throw new TypeError('"buffer" argument must be a Buffer instance');if(t>i||t<a)throw new RangeError('"value" argument is out of bounds');if(r+n>e.length)throw new RangeError("Index out of range")}Q.prototype.writeUintLE=Q.prototype.writeUIntLE=function(t,r,n,e){if(t=+t,r=r>>>0,n=n>>>0,!e){let e=Math.pow(2,8*n)-1;d(this,t,r,n,e,0)}let i=1,a=0;for(this[r]=t&255;++a<n&&(i*=256);)this[r+a]=t/i&255;return r+n};Q.prototype.writeUintBE=Q.prototype.writeUIntBE=function(t,r,n,e){if(t=+t,r=r>>>0,n=n>>>0,!e){let e=Math.pow(2,8*n)-1;d(this,t,r,n,e,0)}let i=n-1,a=1;for(this[r+i]=t&255;--i>=0&&(a*=256);)this[r+i]=t/a&255;return r+n};Q.prototype.writeUint8=Q.prototype.writeUInt8=function(e,t,r){return e=+e,t=t>>>0,r||d(this,e,t,1,255,0),this[t]=e&255,t+1};Q.prototype.writeUint16LE=Q.prototype.writeUInt16LE=function(e,t,r){return e=+e,t=t>>>0,r||d(this,e,t,2,65535,0),this[t]=e&255,this[t+1]=e>>>8,t+2};Q.prototype.writeUint16BE=Q.prototype.writeUInt16BE=function(e,t,r){return e=+e,t=t>>>0,r||d(this,e,t,2,65535,0),this[t]=e>>>8,this[t+1]=e&255,t+2};Q.prototype.writeUint32LE=Q.prototype.writeUInt32LE=function(e,t,r){return e=+e,t=t>>>0,r||d(this,e,t,4,4294967295,0),this[t+3]=e>>>24,this[t+2]=e>>>16,this[t+1]=e>>>8,this[t]=e&255,t+4};Q.prototype.writeUint32BE=Q.prototype.writeUInt32BE=function(e,t,r){return e=+e,t=t>>>0,r||d(this,e,t,4,4294967295,0),this[t]=e>>>24,this[t+1]=e>>>16,this[t+2]=e>>>8,this[t+3]=e&255,t+4};function Ee(e,t,r,n,i){Pe(t,n,i,e,r,7);let a=Number(t&BigInt(4294967295));e[r++]=a,a=a>>8,e[r++]=a,a=a>>8,e[r++]=a,a=a>>8,e[r++]=a;let o=Number(t>>BigInt(32)&BigInt(4294967295));return e[r++]=o,o=o>>8,e[r++]=o,o=o>>8,e[r++]=o,o=o>>8,e[r++]=o,r}function Se(e,t,r,n,i){Pe(t,n,i,e,r,7);let a=Number(t&BigInt(4294967295));e[r+7]=a,a=a>>8,e[r+6]=a,a=a>>8,e[r+5]=a,a=a>>8,e[r+4]=a;let o=Number(t>>BigInt(32)&BigInt(4294967295));return e[r+3]=o,o=o>>8,e[r+2]=o,o=o>>8,e[r+1]=o,o=o>>8,e[r]=o,r+8}Q.prototype.writeBigUInt64LE=function(e,t=0){return Ee(this,e,t,BigInt(0),BigInt("0xffffffffffffffff"))};Q.prototype.writeBigUInt64BE=function(e,t=0){return Se(this,e,t,BigInt(0),BigInt("0xffffffffffffffff"))};Q.prototype.writeIntLE=function(t,r,n,e){if(t=+t,r=r>>>0,!e){let e=Math.pow(2,8*n-1);d(this,t,r,n,e-1,-e)}let i=0,a=1,o=0;for(this[r]=t&255;++i<n&&(a*=256);)t<0&&o===0&&this[r+i-1]!==0&&(o=1),this[r+i]=(t/a>>0)-o&255;return r+n};Q.prototype.writeIntBE=function(t,r,n,e){if(t=+t,r=r>>>0,!e){let e=Math.pow(2,8*n-1);d(this,t,r,n,e-1,-e)}let i=n-1,a=1,o=0;for(this[r+i]=t&255;--i>=0&&(a*=256);)t<0&&o===0&&this[r+i+1]!==0&&(o=1),this[r+i]=(t/a>>0)-o&255;return r+n};Q.prototype.writeInt8=function(e,t,r){return e=+e,t=t>>>0,r||d(this,e,t,1,127,-128),e<0&&(e=255+e+1),this[t]=e&255,t+1};Q.prototype.writeInt16LE=function(e,t,r){return e=+e,t=t>>>0,r||d(this,e,t,2,32767,-32768),this[t]=e&255,this[t+1]=e>>>8,t+2};Q.prototype.writeInt16BE=function(e,t,r){return e=+e,t=t>>>0,r||d(this,e,t,2,32767,-32768),this[t]=e>>>8,this[t+1]=e&255,t+2};Q.prototype.writeInt32LE=function(e,t,r){return e=+e,t=t>>>0,r||d(this,e,t,4,2147483647,-2147483648),this[t]=e&255,this[t+1]=e>>>8,this[t+2]=e>>>16,this[t+3]=e>>>24,t+4};Q.prototype.writeInt32BE=function(e,t,r){return e=+e,t=t>>>0,r||d(this,e,t,4,2147483647,-2147483648),e<0&&(e=4294967295+e+1),this[t]=e>>>24,this[t+1]=e>>>16,this[t+2]=e>>>8,this[t+3]=e&255,t+4};Q.prototype.writeBigInt64LE=function(e,t=0){return Ee(this,e,t,-BigInt("0x8000000000000000"),BigInt("0x7fffffffffffffff"))};Q.prototype.writeBigInt64BE=function(e,t=0){return Se(this,e,t,-BigInt("0x8000000000000000"),BigInt("0x7fffffffffffffff"))};function Ne(e,t,r,n,i,a){if(r+n>e.length)throw new RangeError("Index out of range");if(r<0)throw new RangeError("Index out of range")}function Le(e,t,r,n,i){return t=+t,r=r>>>0,i||Ne(e,t,r,4,34028234663852886e22,-34028234663852886e22),B(e,t,r,n,23,4),r+4}Q.prototype.writeFloatLE=function(e,t,r){return Le(this,e,t,!0,r)};Q.prototype.writeFloatBE=function(e,t,r){return Le(this,e,t,!1,r)};function ke(e,t,r,n,i){return t=+t,r=r>>>0,i||Ne(e,t,r,8,17976931348623157e292,-17976931348623157e292),B(e,t,r,n,52,8),r+8}Q.prototype.writeDoubleLE=function(e,t,r){return ke(this,e,t,!0,r)};Q.prototype.writeDoubleBE=function(e,t,r){return ke(this,e,t,!1,r)};Q.prototype.copy=function(e,t,r,n){if(!Q.isBuffer(e))throw new TypeError("argument should be a Buffer");if(r||(r=0),!n&&n!==0&&(n=this.length),t>=e.length&&(t=e.length),t||(t=0),n>0&&n<r&&(n=r),n===r||e.length===0||this.length===0)return 0;if(t<0)throw new RangeError("targetStart out of bounds");if(r<0||r>=this.length)throw new RangeError("Index out of range");if(n<0)throw new RangeError("sourceEnd out of bounds");n>this.length&&(n=this.length),e.length-t<n-r&&(n=e.length-t+r);let i=n-r;return this===e?this.copyWithin(t,r,n):Uint8Array.prototype.set.call(e,this.subarray(r,n),t),i};Q.prototype.fill=function(r,n,i,a){if(typeof r=="string"){if(typeof n=="string"?(a=n,n=0,i=this.length):typeof i=="string"&&(a=i,i=this.length),a!==void 0&&typeof a!="string")throw new TypeError("encoding must be a string");if(typeof a=="string"&&!Q.isEncoding(a))throw new TypeError("Unknown encoding: "+a);if(r.length===1){let e=r.charCodeAt(0);(a==="utf8"&&e<128||a==="latin1")&&(r=e)}}else typeof r=="number"?r=r&255:typeof r=="boolean"&&(r=Number(r));if(n<0||this.length<n||this.length<i)throw new RangeError("Out of range index");if(i<=n)return this;n=n>>>0,i=i===void 0?this.length:i>>>0,r||(r=0);let o;if(typeof r=="number")for(o=n;o<i;++o)this[o]=r;else{let e=Q.isBuffer(r)?r:Q.from(r,a),t=e.length;if(t===0)throw new TypeError('The value "'+r+'" is invalid for argument "value"');for(o=0;o<i-n;++o)this[o+n]=e[o%t]}return this};var Ce={};function je(e,t,r){Ce[e]=class extends r{constructor(){super(),Object.defineProperty(this,"message",{value:t.apply(this,arguments),writable:!0,configurable:!0}),this.name=`${this.name} [${e}]`,this.stack,delete this.name}get code(){return e}set code(e){Object.defineProperty(this,"code",{configurable:!0,enumerable:!0,value:e,writable:!0})}toString(){return`${this.name} [${e}]: ${this.message}`}}}je("ERR_BUFFER_OUT_OF_BOUNDS",function(e){return e?`${e} is outside of buffer bounds`:"Attempt to access memory outside buffer bounds"},RangeError);je("ERR_INVALID_ARG_TYPE",function(e,t){return`The "${e}" argument must be of type number. Received type ${typeof t}`},TypeError);je("ERR_OUT_OF_RANGE",function(e,t,r){let n=`The value of "${e}" is out of range.`,i=r;return Number.isInteger(r)&&Math.abs(r)>2**32?i=Me(String(r)):typeof r=="bigint"&&(i=String(r),(r>BigInt(2)**BigInt(32)||r<-(BigInt(2)**BigInt(32)))&&(i=Me(i)),i+="n"),n+=` It must be ${t}. Received ${i}`,n},RangeError);function Me(e){let t="",r=e.length,n=e[0]==="-"?1:0;for(;r>=n+4;r-=3)t=`_${e.slice(r-3,r)}${t}`;return`${e.slice(0,r)}${t}`}function Ie(e,t,r){Re(t,"offset"),(e[t]===void 0||e[t+r]===void 0)&&Ae(t,e.length-(r+1))}function Pe(r,n,i,e,t,a){if(r>i||r<n){let e=typeof n=="bigint"?"n":"",t;throw a>3?n===0||n===BigInt(0)?t=`>= 0${e} and < 2${e} ** ${(a+1)*8}${e}`:t=`>= -(2${e} ** ${(a+1)*8-1}${e}) and < 2 ** ${(a+1)*8-1}${e}`:t=`>= ${n}${e} and <= ${i}${e}`,new Ce.ERR_OUT_OF_RANGE("value",t,r)}Ie(e,t,a)}function Re(e,t){if(typeof e!="number")throw new Ce.ERR_INVALID_ARG_TYPE(t,"number",e)}function Ae(e,t,r){throw Math.floor(e)!==e?(Re(e,r),new Ce.ERR_OUT_OF_RANGE(r||"offset","an integer",e)):t<0?new Ce.ERR_BUFFER_OUT_OF_BOUNDS:new Ce.ERR_OUT_OF_RANGE(r||"offset",`>= ${r?1:0} and <= ${t}`,e)}var xe=/[^+/0-9A-Za-z-_]/g;function Te(e){if(e=e.split("=")[0],e=e.trim().replace(xe,""),e.length<2)return"";for(;e.length%4!==0;)e=e+"=";return e}function Ue(t,r){r=r||1/0;let n,i=t.length,a=null,o=[];for(let e=0;e<i;++e){if(n=t.charCodeAt(e),n>55295&&n<57344){if(!a){if(n>56319){(r-=3)>-1&&o.push(239,191,189);continue}else if(e+1===i){(r-=3)>-1&&o.push(239,191,189);continue}a=n;continue}if(n<56320){(r-=3)>-1&&o.push(239,191,189),a=n;continue}n=(a-55296<<10|n-56320)+65536}else a&&(r-=3)>-1&&o.push(239,191,189);if(a=null,n<128){if((r-=1)<0)break;o.push(n)}else if(n<2048){if((r-=2)<0)break;o.push(n>>6|192,n&63|128)}else if(n<65536){if((r-=3)<0)break;o.push(n>>12|224,n>>6&63|128,n&63|128)}else if(n<1114112){if((r-=4)<0)break;o.push(n>>18|240,n>>12&63|128,n>>6&63|128,n&63|128)}else throw new Error("Invalid code point")}return o}function Oe(t){let r=[];for(let e=0;e<t.length;++e)r.push(t.charCodeAt(e)&255);return r}function Fe(t,r){let n,i,a,o=[];for(let e=0;e<t.length&&!((r-=2)<0);++e)n=t.charCodeAt(e),i=n>>8,a=n%256,o.push(a),o.push(i);return o}function De(e){return D(Te(e))}function ze(e,t,r,n){let i;for(i=0;i<n&&!(i+r>=t.length||i>=e.length);++i)t[i+r]=e[i];return i}var $e=function(){let n="0123456789abcdef",i=new Array(256);for(let r=0;r<16;++r){let t=r*16;for(let e=0;e<16;++e)i[t+e]=n[r]+n[e]}return i}();var Ve={};T(Ve,{ArtMethod:()=>_a,ArtStackVisitor:()=>fa,DVM_JNI_ENV_OFFSET_SELF:()=>wn,HandleVector:()=>xo,VariableSizedHandleScope:()=>Jo,backtrace:()=>Da,deoptimizeBootImage:()=>fo,deoptimizeEverything:()=>po,deoptimizeMethod:()=>ho,ensureClassInitialized:()=>yi,getAndroidApiLevel:()=>m,getAndroidVersion:()=>Xn,getApi:()=>S,getArtApexVersion:()=>ei,getArtClassSpec:()=>$i,getArtFieldSpec:()=>Ji,getArtMethodSpec:()=>E,getArtThreadFromEnv:()=>Qi,getArtThreadSpec:()=>Wn,makeArtClassLoaderVisitor:()=>ha,makeArtClassVisitor:()=>ca,makeMethodMangler:()=>Oa,makeObjectVisitorPredicate:()=>Zo,revertGlobalPatches:()=>Va,translateMethod:()=>Fa,withAllArtThreadsSuspended:()=>la,withRunnableArtThread:()=>w});var{pageSize:Je,pointerSize:Be}=Process,Ge=class{constructor(e){this.sliceSize=e,this.slicesPerPage=Je/e,this.pages=[],this.free=[]}allocateSlice(o,t){let s=o.near===void 0,l=t===1;if(s&&l){let e=this.free.pop();if(e!==void 0)return e}else if(t<Je){let{free:i}=this,e=i.length,a=l?null:ptr(t-1);for(let n=0;n!==e;n++){let e=i[n],t=s||this._isSliceNear(e,o),r=l||e.and(a).isNull();if(t&&r)return i.splice(n,1)[0]}}return this._allocatePage(o)}_allocatePage(e){let r=Memory.alloc(Je,e),{sliceSize:n,slicesPerPage:i}=this;for(let t=1;t!==i;t++){let e=r.add(t*n);this.free.push(e)}return this.pages.push(r),r}_isSliceNear(e,t){let r=e.add(this.sliceSize),{near:n,maxDistance:i}=t,a=Ze(n.sub(e)),o=Ze(n.sub(r));return a.compare(i)<=0&&o.compare(i)<=0}freeSlice(e){this.free.push(e)}};function Ze(e){let t=Be===4?31:63,r=ptr(1).shl(t).not();return e.and(r)}function He(e){return new Ge(e)}function h(e,t){if(t!==0)throw new Error(e+" failed: "+t)}var qe={v1_0:805371904,v1_2:805372416},We={canTagObjects:1},{pointerSize:Ke}=Process,Qe={exceptions:"propagate"};function c(e,t){this.handle=e,this.vm=t,this.vtable=e.readPointer()}c.prototype.deallocate=Xe(47,"int32",["pointer","pointer"],function(e,t){return e(this.handle,t)});c.prototype.getLoadedClasses=Xe(78,"int32",["pointer","pointer","pointer"],function(e,t,r){let n=e(this.handle,t,r);h("EnvJvmti::getLoadedClasses",n)});c.prototype.iterateOverInstancesOfClass=Xe(112,"int32",["pointer","pointer","int","pointer","pointer"],function(e,t,r,n,i){let a=e(this.handle,t,r,n,i);h("EnvJvmti::iterateOverInstancesOfClass",a)});c.prototype.getObjectsWithTags=Xe(114,"int32",["pointer","int","pointer","pointer","pointer","pointer"],function(e,t,r,n,i,a){let o=e(this.handle,t,r,n,i,a);h("EnvJvmti::getObjectsWithTags",o)});c.prototype.addCapabilities=Xe(142,"int32",["pointer","pointer"],function(e,t){return e(this.handle,t)});function Xe(t,r,n,i){let a=null;return function(){a===null&&(a=new NativeFunction(this.vtable.add((t-1)*Ke).readPointer(),r,n,Qe));let e=[a];return e=e.concat.apply(e,arguments),i.apply(this,e)}}function p(e,r,{limit:t}){let n=e,i=null;for(let e=0;e!==t;e++){let e=Instruction.parse(n),t=r(e,i);if(t!==null)return t;n=e.next,i=e}return null}function e(t){let r=null,n=!1;return function(...e){return n||(r=t(...e),n=!0),r}}function g(e,t){this.handle=e,this.vm=t}var Ye=Process.pointerSize,n=2,et=28,tt=34,rt=37,nt=40,it=43,at=46,ot=49,st=52,lt=55,dt=58,ct=61,ut=64,ht=67,pt=70,ft=73,_t=76,mt=79,gt=82,bt=85,vt=88,yt=91,wt=114,Et=117,St=120,Nt=123,Lt=126,kt=129,Ct=132,jt=135,Mt=138,It=141,Pt=95,Rt=96,At=97,xt=98,Tt=99,Ut=100,Ot=101,Ft=102,Dt=103,zt=104,$t=105,Vt=106,Jt=107,Bt=108,Gt=109,Zt=110,Ht=111,qt=112,Wt=145,Kt=146,Qt=147,Xt=148,Yt=149,er=150,tr=151,rr=152,nr=153,ir=154,ar=155,or=156,sr=157,lr=158,dr=159,cr=160,ur=161,hr=162,pr={pointer:tt,uint8:rt,int8:nt,uint16:it,int16:at,int32:ot,int64:st,float:lt,double:dt,void:ct},fr={pointer:ut,uint8:ht,int8:pt,uint16:ft,int16:_t,int32:mt,int64:gt,float:bt,double:vt,void:yt},_r={pointer:wt,uint8:Et,int8:St,uint16:Nt,int16:Lt,int32:kt,int64:Ct,float:jt,double:Mt,void:It},mr={pointer:Pt,uint8:Rt,int8:At,uint16:xt,int16:Tt,int32:Ut,int64:Ot,float:Ft,double:Dt},gr={pointer:zt,uint8:$t,int8:Vt,uint16:Jt,int16:Bt,int32:Gt,int64:Zt,float:Ht,double:qt},br={pointer:Wt,uint8:Kt,int8:Qt,uint16:Xt,int16:Yt,int32:er,int64:tr,float:rr,double:nr},vr={pointer:ir,uint8:ar,int8:or,uint16:sr,int16:lr,int32:dr,int64:cr,float:ur,double:hr},yr={exceptions:"propagate"},wr=null,Er=[];g.dispose=function(e){Er.forEach(e.deleteGlobalRef,e),Er=[]};function i(e){return Er.push(e),e}function Sr(e){return wr===null&&(wr=e.handle.readPointer()),wr}function t(t,r,n,i){let a=null;return function(){a===null&&(a=new NativeFunction(Sr(this).add(t*Ye).readPointer(),r,n,yr));let e=[a];return e=e.concat.apply(e,arguments),i.apply(this,e)}}g.prototype.getVersion=t(4,"int32",["pointer"],function(e){return e(this.handle)});g.prototype.findClass=t(6,"pointer",["pointer","pointer"],function(e,t){let r=e(this.handle,Memory.allocUtf8String(t));return this.throwIfExceptionPending(),r});g.prototype.throwIfExceptionPending=function(){let e=this.exceptionOccurred();if(e.isNull())return;this.exceptionClear();let t=this.newGlobalRef(e);this.deleteLocalRef(e);let r=this.vaMethod("pointer",[])(this.handle,t,this.javaLangObject().toString),n=this.stringFromJni(r);this.deleteLocalRef(r);let i=new Error(n);throw i.$h=t,Script.bindWeak(i,Nr(this.vm,t)),i};function Nr(e,t){return function(){e.perform(e=>{e.deleteGlobalRef(t)})}}g.prototype.fromReflectedMethod=t(7,"pointer",["pointer","pointer"],function(e,t){return e(this.handle,t)});g.prototype.fromReflectedField=t(8,"pointer",["pointer","pointer"],function(e,t){return e(this.handle,t)});g.prototype.toReflectedMethod=t(9,"pointer",["pointer","pointer","pointer","uint8"],function(e,t,r,n){return e(this.handle,t,r,n)});g.prototype.getSuperclass=t(10,"pointer",["pointer","pointer"],function(e,t){return e(this.handle,t)});g.prototype.isAssignableFrom=t(11,"uint8",["pointer","pointer","pointer"],function(e,t,r){return!!e(this.handle,t,r)});g.prototype.toReflectedField=t(12,"pointer",["pointer","pointer","pointer","uint8"],function(e,t,r,n){return e(this.handle,t,r,n)});g.prototype.throw=t(13,"int32",["pointer","pointer"],function(e,t){return e(this.handle,t)});g.prototype.exceptionOccurred=t(15,"pointer",["pointer"],function(e){return e(this.handle)});g.prototype.exceptionDescribe=t(16,"void",["pointer"],function(e){e(this.handle)});g.prototype.exceptionClear=t(17,"void",["pointer"],function(e){e(this.handle)});g.prototype.pushLocalFrame=t(19,"int32",["pointer","int32"],function(e,t){return e(this.handle,t)});g.prototype.popLocalFrame=t(20,"pointer",["pointer","pointer"],function(e,t){return e(this.handle,t)});g.prototype.newGlobalRef=t(21,"pointer",["pointer","pointer"],function(e,t){return e(this.handle,t)});g.prototype.deleteGlobalRef=t(22,"void",["pointer","pointer"],function(e,t){e(this.handle,t)});g.prototype.deleteLocalRef=t(23,"void",["pointer","pointer"],function(e,t){e(this.handle,t)});g.prototype.isSameObject=t(24,"uint8",["pointer","pointer","pointer"],function(e,t,r){return!!e(this.handle,t,r)});g.prototype.newLocalRef=t(25,"pointer",["pointer","pointer"],function(e,t){return e(this.handle,t)});g.prototype.allocObject=t(27,"pointer",["pointer","pointer"],function(e,t){return e(this.handle,t)});g.prototype.getObjectClass=t(31,"pointer",["pointer","pointer"],function(e,t){return e(this.handle,t)});g.prototype.isInstanceOf=t(32,"uint8",["pointer","pointer","pointer"],function(e,t,r){return!!e(this.handle,t,r)});g.prototype.getMethodId=t(33,"pointer",["pointer","pointer","pointer","pointer"],function(e,t,r,n){return e(this.handle,t,Memory.allocUtf8String(r),Memory.allocUtf8String(n))});g.prototype.getFieldId=t(94,"pointer",["pointer","pointer","pointer","pointer"],function(e,t,r,n){return e(this.handle,t,Memory.allocUtf8String(r),Memory.allocUtf8String(n))});g.prototype.getIntField=t(100,"int32",["pointer","pointer","pointer"],function(e,t,r){return e(this.handle,t,r)});g.prototype.getStaticMethodId=t(113,"pointer",["pointer","pointer","pointer","pointer"],function(e,t,r,n){return e(this.handle,t,Memory.allocUtf8String(r),Memory.allocUtf8String(n))});g.prototype.getStaticFieldId=t(144,"pointer",["pointer","pointer","pointer","pointer"],function(e,t,r,n){return e(this.handle,t,Memory.allocUtf8String(r),Memory.allocUtf8String(n))});g.prototype.getStaticIntField=t(150,"int32",["pointer","pointer","pointer"],function(e,t,r){return e(this.handle,t,r)});g.prototype.getStringLength=t(164,"int32",["pointer","pointer"],function(e,t){return e(this.handle,t)});g.prototype.getStringChars=t(165,"pointer",["pointer","pointer","pointer"],function(e,t){return e(this.handle,t,NULL)});g.prototype.releaseStringChars=t(166,"void",["pointer","pointer","pointer"],function(e,t,r){e(this.handle,t,r)});g.prototype.newStringUtf=t(167,"pointer",["pointer","pointer"],function(e,t){let r=Memory.allocUtf8String(t);return e(this.handle,r)});g.prototype.getStringUtfChars=t(169,"pointer",["pointer","pointer","pointer"],function(e,t){return e(this.handle,t,NULL)});g.prototype.releaseStringUtfChars=t(170,"void",["pointer","pointer","pointer"],function(e,t,r){e(this.handle,t,r)});g.prototype.getArrayLength=t(171,"int32",["pointer","pointer"],function(e,t){return e(this.handle,t)});g.prototype.newObjectArray=t(172,"pointer",["pointer","int32","pointer","pointer"],function(e,t,r,n){return e(this.handle,t,r,n)});g.prototype.getObjectArrayElement=t(173,"pointer",["pointer","pointer","int32"],function(e,t,r){return e(this.handle,t,r)});g.prototype.setObjectArrayElement=t(174,"void",["pointer","pointer","int32","pointer"],function(e,t,r,n){e(this.handle,t,r,n)});g.prototype.newBooleanArray=t(175,"pointer",["pointer","int32"],function(e,t){return e(this.handle,t)});g.prototype.newByteArray=t(176,"pointer",["pointer","int32"],function(e,t){return e(this.handle,t)});g.prototype.newCharArray=t(177,"pointer",["pointer","int32"],function(e,t){return e(this.handle,t)});g.prototype.newShortArray=t(178,"pointer",["pointer","int32"],function(e,t){return e(this.handle,t)});g.prototype.newIntArray=t(179,"pointer",["pointer","int32"],function(e,t){return e(this.handle,t)});g.prototype.newLongArray=t(180,"pointer",["pointer","int32"],function(e,t){return e(this.handle,t)});g.prototype.newFloatArray=t(181,"pointer",["pointer","int32"],function(e,t){return e(this.handle,t)});g.prototype.newDoubleArray=t(182,"pointer",["pointer","int32"],function(e,t){return e(this.handle,t)});g.prototype.getBooleanArrayElements=t(183,"pointer",["pointer","pointer","pointer"],function(e,t){return e(this.handle,t,NULL)});g.prototype.getByteArrayElements=t(184,"pointer",["pointer","pointer","pointer"],function(e,t){return e(this.handle,t,NULL)});g.prototype.getCharArrayElements=t(185,"pointer",["pointer","pointer","pointer"],function(e,t){return e(this.handle,t,NULL)});g.prototype.getShortArrayElements=t(186,"pointer",["pointer","pointer","pointer"],function(e,t){return e(this.handle,t,NULL)});g.prototype.getIntArrayElements=t(187,"pointer",["pointer","pointer","pointer"],function(e,t){return e(this.handle,t,NULL)});g.prototype.getLongArrayElements=t(188,"pointer",["pointer","pointer","pointer"],function(e,t){return e(this.handle,t,NULL)});g.prototype.getFloatArrayElements=t(189,"pointer",["pointer","pointer","pointer"],function(e,t){return e(this.handle,t,NULL)});g.prototype.getDoubleArrayElements=t(190,"pointer",["pointer","pointer","pointer"],function(e,t){return e(this.handle,t,NULL)});g.prototype.releaseBooleanArrayElements=t(191,"pointer",["pointer","pointer","pointer","int32"],function(e,t,r){e(this.handle,t,r,n)});g.prototype.releaseByteArrayElements=t(192,"pointer",["pointer","pointer","pointer","int32"],function(e,t,r){e(this.handle,t,r,n)});g.prototype.releaseCharArrayElements=t(193,"pointer",["pointer","pointer","pointer","int32"],function(e,t,r){e(this.handle,t,r,n)});g.prototype.releaseShortArrayElements=t(194,"pointer",["pointer","pointer","pointer","int32"],function(e,t,r){e(this.handle,t,r,n)});g.prototype.releaseIntArrayElements=t(195,"pointer",["pointer","pointer","pointer","int32"],function(e,t,r){e(this.handle,t,r,n)});g.prototype.releaseLongArrayElements=t(196,"pointer",["pointer","pointer","pointer","int32"],function(e,t,r){e(this.handle,t,r,n)});g.prototype.releaseFloatArrayElements=t(197,"pointer",["pointer","pointer","pointer","int32"],function(e,t,r){e(this.handle,t,r,n)});g.prototype.releaseDoubleArrayElements=t(198,"pointer",["pointer","pointer","pointer","int32"],function(e,t,r){e(this.handle,t,r,n)});g.prototype.getByteArrayRegion=t(200,"void",["pointer","pointer","int","int","pointer"],function(e,t,r,n,i){e(this.handle,t,r,n,i)});g.prototype.setBooleanArrayRegion=t(207,"void",["pointer","pointer","int32","int32","pointer"],function(e,t,r,n,i){e(this.handle,t,r,n,i)});g.prototype.setByteArrayRegion=t(208,"void",["pointer","pointer","int32","int32","pointer"],function(e,t,r,n,i){e(this.handle,t,r,n,i)});g.prototype.setCharArrayRegion=t(209,"void",["pointer","pointer","int32","int32","pointer"],function(e,t,r,n,i){e(this.handle,t,r,n,i)});g.prototype.setShortArrayRegion=t(210,"void",["pointer","pointer","int32","int32","pointer"],function(e,t,r,n,i){e(this.handle,t,r,n,i)});g.prototype.setIntArrayRegion=t(211,"void",["pointer","pointer","int32","int32","pointer"],function(e,t,r,n,i){e(this.handle,t,r,n,i)});g.prototype.setLongArrayRegion=t(212,"void",["pointer","pointer","int32","int32","pointer"],function(e,t,r,n,i){e(this.handle,t,r,n,i)});g.prototype.setFloatArrayRegion=t(213,"void",["pointer","pointer","int32","int32","pointer"],function(e,t,r,n,i){e(this.handle,t,r,n,i)});g.prototype.setDoubleArrayRegion=t(214,"void",["pointer","pointer","int32","int32","pointer"],function(e,t,r,n,i){e(this.handle,t,r,n,i)});g.prototype.registerNatives=t(215,"int32",["pointer","pointer","pointer","int32"],function(e,t,r,n){return e(this.handle,t,r,n)});g.prototype.monitorEnter=t(217,"int32",["pointer","pointer"],function(e,t){return e(this.handle,t)});g.prototype.monitorExit=t(218,"int32",["pointer","pointer"],function(e,t){return e(this.handle,t)});g.prototype.getDirectBufferAddress=t(230,"pointer",["pointer","pointer"],function(e,t){return e(this.handle,t)});g.prototype.getObjectRefType=t(232,"int32",["pointer","pointer"],function(e,t){return e(this.handle,t)});var Lr=new Map;function kr(e,t,r,n){return Mr(this,"p",Ir,e,t,r,n)}function Cr(e,t,r,n){return Mr(this,"v",Pr,e,t,r,n)}function jr(e,t,r,n){return Mr(this,"n",Rr,e,t,r,n)}function Mr(e,t,r,n,i,a,o){if(o!==void 0)return r(e,n,i,a,o);let s=[n,t,i].concat(a).join("|"),l=Lr.get(s);return l===void 0&&(l=r(e,n,i,a,yr),Lr.set(s,l)),l}function Ir(e,t,r,n,i){return new NativeFunction(Sr(e).add(t*Ye).readPointer(),r,["pointer","pointer","pointer"].concat(n),i)}function Pr(e,t,r,n,i){return new NativeFunction(Sr(e).add(t*Ye).readPointer(),r,["pointer","pointer","pointer","..."].concat(n),i)}function Rr(e,t,r,n,i){return new NativeFunction(Sr(e).add(t*Ye).readPointer(),r,["pointer","pointer","pointer","pointer","..."].concat(n),i)}g.prototype.constructor=function(e,t){return Cr.call(this,et,"pointer",e,t)};g.prototype.vaMethod=function(e,t,r){let n=pr[e];if(n===void 0)throw new Error("Unsupported type: "+e);return Cr.call(this,n,e,t,r)};g.prototype.nonvirtualVaMethod=function(e,t,r){let n=fr[e];if(n===void 0)throw new Error("Unsupported type: "+e);return jr.call(this,n,e,t,r)};g.prototype.staticVaMethod=function(e,t,r){let n=_r[e];if(n===void 0)throw new Error("Unsupported type: "+e);return Cr.call(this,n,e,t,r)};g.prototype.getField=function(e){let t=mr[e];if(t===void 0)throw new Error("Unsupported type: "+e);return kr.call(this,t,e,[])};g.prototype.getStaticField=function(e){let t=br[e];if(t===void 0)throw new Error("Unsupported type: "+e);return kr.call(this,t,e,[])};g.prototype.setField=function(e){let t=gr[e];if(t===void 0)throw new Error("Unsupported type: "+e);return kr.call(this,t,"void",[e])};g.prototype.setStaticField=function(e){let t=vr[e];if(t===void 0)throw new Error("Unsupported type: "+e);return kr.call(this,t,"void",[e])};var Ar=null;g.prototype.javaLangClass=function(){if(Ar===null){let t=this.findClass("java/lang/Class");try{let e=this.getMethodId.bind(this,t);Ar={handle:i(this.newGlobalRef(t)),getName:e("getName","()Ljava/lang/String;"),getSimpleName:e("getSimpleName","()Ljava/lang/String;"),getGenericSuperclass:e("getGenericSuperclass","()Ljava/lang/reflect/Type;"),getDeclaredConstructors:e("getDeclaredConstructors","()[Ljava/lang/reflect/Constructor;"),getDeclaredMethods:e("getDeclaredMethods","()[Ljava/lang/reflect/Method;"),getDeclaredFields:e("getDeclaredFields","()[Ljava/lang/reflect/Field;"),isArray:e("isArray","()Z"),isPrimitive:e("isPrimitive","()Z"),isInterface:e("isInterface","()Z"),getComponentType:e("getComponentType","()Ljava/lang/Class;")}}finally{this.deleteLocalRef(t)}}return Ar};var xr=null;g.prototype.javaLangObject=function(){if(xr===null){let t=this.findClass("java/lang/Object");try{let e=this.getMethodId.bind(this,t);xr={handle:i(this.newGlobalRef(t)),toString:e("toString","()Ljava/lang/String;"),getClass:e("getClass","()Ljava/lang/Class;")}}finally{this.deleteLocalRef(t)}}return xr};var Tr=null;g.prototype.javaLangReflectConstructor=function(){if(Tr===null){let e=this.findClass("java/lang/reflect/Constructor");try{Tr={getGenericParameterTypes:this.getMethodId(e,"getGenericParameterTypes","()[Ljava/lang/reflect/Type;")}}finally{this.deleteLocalRef(e)}}return Tr};var Ur=null;g.prototype.javaLangReflectMethod=function(){if(Ur===null){let t=this.findClass("java/lang/reflect/Method");try{let e=this.getMethodId.bind(this,t);Ur={getName:e("getName","()Ljava/lang/String;"),getGenericParameterTypes:e("getGenericParameterTypes","()[Ljava/lang/reflect/Type;"),getParameterTypes:e("getParameterTypes","()[Ljava/lang/Class;"),getGenericReturnType:e("getGenericReturnType","()Ljava/lang/reflect/Type;"),getGenericExceptionTypes:e("getGenericExceptionTypes","()[Ljava/lang/reflect/Type;"),getModifiers:e("getModifiers","()I"),isVarArgs:e("isVarArgs","()Z")}}finally{this.deleteLocalRef(t)}}return Ur};var Or=null;g.prototype.javaLangReflectField=function(){if(Or===null){let t=this.findClass("java/lang/reflect/Field");try{let e=this.getMethodId.bind(this,t);Or={getName:e("getName","()Ljava/lang/String;"),getType:e("getType","()Ljava/lang/Class;"),getGenericType:e("getGenericType","()Ljava/lang/reflect/Type;"),getModifiers:e("getModifiers","()I"),toString:e("toString","()Ljava/lang/String;")}}finally{this.deleteLocalRef(t)}}return Or};var Fr=null;g.prototype.javaLangReflectTypeVariable=function(){if(Fr===null){let t=this.findClass("java/lang/reflect/TypeVariable");try{let e=this.getMethodId.bind(this,t);Fr={handle:i(this.newGlobalRef(t)),getName:e("getName","()Ljava/lang/String;"),getBounds:e("getBounds","()[Ljava/lang/reflect/Type;"),getGenericDeclaration:e("getGenericDeclaration","()Ljava/lang/reflect/GenericDeclaration;")}}finally{this.deleteLocalRef(t)}}return Fr};var Dr=null;g.prototype.javaLangReflectWildcardType=function(){if(Dr===null){let t=this.findClass("java/lang/reflect/WildcardType");try{let e=this.getMethodId.bind(this,t);Dr={handle:i(this.newGlobalRef(t)),getLowerBounds:e("getLowerBounds","()[Ljava/lang/reflect/Type;"),getUpperBounds:e("getUpperBounds","()[Ljava/lang/reflect/Type;")}}finally{this.deleteLocalRef(t)}}return Dr};var zr=null;g.prototype.javaLangReflectGenericArrayType=function(){if(zr===null){let e=this.findClass("java/lang/reflect/GenericArrayType");try{zr={handle:i(this.newGlobalRef(e)),getGenericComponentType:this.getMethodId(e,"getGenericComponentType","()Ljava/lang/reflect/Type;")}}finally{this.deleteLocalRef(e)}}return zr};var $r=null;g.prototype.javaLangReflectParameterizedType=function(){if($r===null){let t=this.findClass("java/lang/reflect/ParameterizedType");try{let e=this.getMethodId.bind(this,t);$r={handle:i(this.newGlobalRef(t)),getActualTypeArguments:e("getActualTypeArguments","()[Ljava/lang/reflect/Type;"),getRawType:e("getRawType","()Ljava/lang/reflect/Type;"),getOwnerType:e("getOwnerType","()Ljava/lang/reflect/Type;")}}finally{this.deleteLocalRef(t)}}return $r};var Vr=null;g.prototype.javaLangString=function(){if(Vr===null){let e=this.findClass("java/lang/String");try{Vr={handle:i(this.newGlobalRef(e))}}finally{this.deleteLocalRef(e)}}return Vr};g.prototype.getClassName=function(e){let t=this.vaMethod("pointer",[])(this.handle,e,this.javaLangClass().getName);try{return this.stringFromJni(t)}finally{this.deleteLocalRef(t)}};g.prototype.getObjectClassName=function(e){let t=this.getObjectClass(e);try{return this.getClassName(t)}finally{this.deleteLocalRef(t)}};g.prototype.getActualTypeArgument=function(e){let t=this.vaMethod("pointer",[])(this.handle,e,this.javaLangReflectParameterizedType().getActualTypeArguments);if(this.throwIfExceptionPending(),!t.isNull())try{return this.getTypeNameFromFirstTypeElement(t)}finally{this.deleteLocalRef(t)}};g.prototype.getTypeNameFromFirstTypeElement=function(t){if(this.getArrayLength(t)>0){let e=this.getObjectArrayElement(t,0);try{return this.getTypeName(e)}finally{this.deleteLocalRef(e)}}else return"java.lang.Object"};g.prototype.getTypeName=function(r,n){let i=this.vaMethod("pointer",[]);if(this.isInstanceOf(r,this.javaLangClass().handle))return this.getClassName(r);if(this.isInstanceOf(r,this.javaLangReflectGenericArrayType().handle))return this.getArrayTypeName(r);if(this.isInstanceOf(r,this.javaLangReflectParameterizedType().handle)){let e=i(this.handle,r,this.javaLangReflectParameterizedType().getRawType);this.throwIfExceptionPending();let t;try{t=this.getTypeName(e)}finally{this.deleteLocalRef(e)}return n&&(t+="<"+this.getActualTypeArgument(r)+">"),t}else return this.isInstanceOf(r,this.javaLangReflectTypeVariable().handle)||this.isInstanceOf(r,this.javaLangReflectWildcardType().handle),"java.lang.Object"};g.prototype.getArrayTypeName=function(t){let r=this.vaMethod("pointer",[]);if(this.isInstanceOf(t,this.javaLangClass().handle))return this.getClassName(t);if(this.isInstanceOf(t,this.javaLangReflectGenericArrayType().handle)){let e=r(this.handle,t,this.javaLangReflectGenericArrayType().getGenericComponentType);this.throwIfExceptionPending();try{return"[L"+this.getTypeName(e)+";"}finally{this.deleteLocalRef(e)}}else return"[Ljava.lang.Object;"};g.prototype.stringFromJni=function(t){let r=this.getStringChars(t);if(r.isNull())throw new Error("Unable to access string");try{let e=this.getStringLength(t);return r.readUtf16String(e)}finally{this.releaseStringChars(t,r)}};var Jr=65542,Br=Process.pointerSize,Gr=Process.getCurrentThreadId(),u=new Map,Zr=new Map;function Hr(e){let n=e.vm,r=null,i=null,a=null;function t(){let e=n.readPointer(),t={exceptions:"propagate"};r=new NativeFunction(e.add(4*Br).readPointer(),"int32",["pointer","pointer","pointer"],t),i=new NativeFunction(e.add(5*Br).readPointer(),"int32",["pointer"],t),a=new NativeFunction(e.add(6*Br).readPointer(),"int32",["pointer","pointer","int32"],t)}this.handle=n,this.perform=function(e){let t=Process.getCurrentThreadId(),r=o(t);if(r!==null)return e(r);let n=this._tryGetEnv(),i=n!==null;i||(n=this.attachCurrentThread(),u.set(t,!0)),this.link(t,n);try{return e(n)}finally{let e=t===Gr;if(e||this.unlink(t),!i&&!e){let e=u.get(t);u.delete(t),e&&this.detachCurrentThread()}}},this.attachCurrentThread=function(){let e=Memory.alloc(Br);return h("VM::AttachCurrentThread",r(n,e,NULL)),new g(e.readPointer(),this)},this.detachCurrentThread=function(){h("VM::DetachCurrentThread",i(n))},this.preventDetachDueToClassLoader=function(){let e=Process.getCurrentThreadId();u.has(e)&&u.set(e,!1)},this.getEnv=function(){let e=o(Process.getCurrentThreadId());if(e!==null)return e;let t=Memory.alloc(Br),r=a(n,t,Jr);if(r===-2)throw new Error("Current thread is not attached to the Java VM; please move this code inside a Java.perform() callback");return h("VM::GetEnv",r),new g(t.readPointer(),this)},this.tryGetEnv=function(){let e=o(Process.getCurrentThreadId());return e!==null?e:this._tryGetEnv()},this._tryGetEnv=function(){let e=this.tryGetEnvHandle(Jr);return e===null?null:new g(e,this)},this.tryGetEnvHandle=function(e){let t=Memory.alloc(Br);return a(n,t,e)!==0?null:t.readPointer()},this.makeHandleDestructor=function(t){return()=>{this.perform(e=>{e.deleteGlobalRef(t)})}},this.link=function(e,t){let r=Zr.get(e);r===void 0?Zr.set(e,[t,1]):r[1]++},this.unlink=function(e){let t=Zr.get(e);t[1]===1?Zr.delete(e):t[1]--};function o(e){let t=Zr.get(e);return t===void 0?null:t[0]}t.call(this)}Hr.dispose=function(e){u.get(Gr)===!0&&(u.delete(Gr),e.detachCurrentThread())};var qr=4,v=Process.pointerSize,{readU32:Wr,readPointer:Kr,writeU32:Qr,writePointer:Xr}=NativePointer.prototype,Yr=1,en=8,tn=16,rn=256,nn=524288,an=2097152,on=1073741824,sn=524288,ln=134217728,dn=1048576,cn=2097152,un=268435456,hn=268435456,pn=0,fn=3,_n=5,mn=ptr(1).not(),gn=2147467263,bn=4294963200,vn=17*v,yn=18*v,wn=12,En=112,Sn=116,Nn=0,Ln=56,kn=4,Cn=8,jn=10,Mn=12,In=14,Pn=28,Rn=36,An=0,xn=1,Tn=2,Un=3,On=4,Fn=5,Dn=6,zn=7,$n=2147483648,Vn=28,Jn=3*v,Bn=3*v,Gn=1,Zn=1,Hn=e(Ei),qn=e(Fi),E=e(Vi),Wn=e(Bi),Kn=e(Gi),Qn=e(oa),Xn=e(Xi),Yn=e(Yi),m=e(ea),ei=e(ta),ti=e(ga),ri=Process.arch==="ia32"?Io:Mo,y={exceptions:"propagate"},ni={},ii=null,ai=null,oi=null,_=null,si=[],li=new Map,di=[],ci=null,ui=0,hi=!1,pi=!1,fi=null,_i=[],mi=null,gi=null;function S(){return ii===null&&(ii=bi()),ii}function bi(){let e=Process.enumerateModules().filter(e=>/^lib(art|dvm).so$/.test(e.name)).filter(e=>!/\/system\/fake-libs/.test(e.path));if(e.length===0)return null;let t=e[0],r=t.name.indexOf("art")!==-1?"art":"dalvik",n=r==="art",f={module:t,find(e){let{module:t}=this,r=t.findExportByName(e);return r===null&&(r=t.findSymbolByName(e)),r},flavor:r,addLocalReference:null};f.isApiLevel34OrApexEquivalent=n&&(f.find("_ZN3art7AppInfo29GetPrimaryApkReferenceProfileEv")!==null||f.find("_ZN3art6Thread15RunFlipFunctionEPS0_")!==null);let i=n?{functions:{JNI_GetCreatedJavaVMs:["JNI_GetCreatedJavaVMs","int",["pointer","int","pointer"]],artInterpreterToCompiledCodeBridge:function(e){this.artInterpreterToCompiledCodeBridge=e},_ZN3art9JavaVMExt12AddGlobalRefEPNS_6ThreadENS_6ObjPtrINS_6mirror6ObjectEEE:["art::JavaVMExt::AddGlobalRef","pointer",["pointer","pointer","pointer"]],_ZN3art9JavaVMExt12AddGlobalRefEPNS_6ThreadEPNS_6mirror6ObjectE:["art::JavaVMExt::AddGlobalRef","pointer",["pointer","pointer","pointer"]],_ZN3art17ReaderWriterMutex13ExclusiveLockEPNS_6ThreadE:["art::ReaderWriterMutex::ExclusiveLock","void",["pointer","pointer"]],_ZN3art17ReaderWriterMutex15ExclusiveUnlockEPNS_6ThreadE:["art::ReaderWriterMutex::ExclusiveUnlock","void",["pointer","pointer"]],_ZN3art22IndirectReferenceTable3AddEjPNS_6mirror6ObjectE:function(e){this["art::IndirectReferenceTable::Add"]=new NativeFunction(e,"pointer",["pointer","uint","pointer"],y)},_ZN3art22IndirectReferenceTable3AddENS_15IRTSegmentStateENS_6ObjPtrINS_6mirror6ObjectEEE:function(e){this["art::IndirectReferenceTable::Add"]=new NativeFunction(e,"pointer",["pointer","uint","pointer"],y)},_ZN3art9JavaVMExt12DecodeGlobalEPv:function(e){let n;m()>=26?n=ri(e,["pointer","pointer"]):n=new NativeFunction(e,"pointer",["pointer","pointer"],y),this["art::JavaVMExt::DecodeGlobal"]=function(e,t,r){return n(e,r)}},_ZN3art9JavaVMExt12DecodeGlobalEPNS_6ThreadEPv:["art::JavaVMExt::DecodeGlobal","pointer",["pointer","pointer","pointer"]],_ZNK3art6Thread19DecodeGlobalJObjectEP8_jobject:["art::Thread::DecodeJObject","pointer",["pointer","pointer"]],_ZNK3art6Thread13DecodeJObjectEP8_jobject:["art::Thread::DecodeJObject","pointer",["pointer","pointer"]],_ZN3art10ThreadList10SuspendAllEPKcb:["art::ThreadList::SuspendAll","void",["pointer","pointer","bool"]],_ZN3art10ThreadList10SuspendAllEv:function(e){let n=new NativeFunction(e,"void",["pointer"],y);this["art::ThreadList::SuspendAll"]=function(e,t,r){return n(e)}},_ZN3art10ThreadList9ResumeAllEv:["art::ThreadList::ResumeAll","void",["pointer"]],_ZN3art11ClassLinker12VisitClassesEPNS_12ClassVisitorE:["art::ClassLinker::VisitClasses","void",["pointer","pointer"]],_ZN3art11ClassLinker12VisitClassesEPFbPNS_6mirror5ClassEPvES4_:function(e){let r=new NativeFunction(e,"void",["pointer","pointer","pointer"],y);this["art::ClassLinker::VisitClasses"]=function(e,t){r(e,t,NULL)}},_ZNK3art11ClassLinker17VisitClassLoadersEPNS_18ClassLoaderVisitorE:["art::ClassLinker::VisitClassLoaders","void",["pointer","pointer"]],_ZN3art2gc4Heap12VisitObjectsEPFvPNS_6mirror6ObjectEPvES5_:["art::gc::Heap::VisitObjects","void",["pointer","pointer","pointer"]],_ZN3art2gc4Heap12GetInstancesERNS_24VariableSizedHandleScopeENS_6HandleINS_6mirror5ClassEEEiRNSt3__16vectorINS4_INS5_6ObjectEEENS8_9allocatorISB_EEEE:["art::gc::Heap::GetInstances","void",["pointer","pointer","pointer","int","pointer"]],_ZN3art2gc4Heap12GetInstancesERNS_24VariableSizedHandleScopeENS_6HandleINS_6mirror5ClassEEEbiRNSt3__16vectorINS4_INS5_6ObjectEEENS8_9allocatorISB_EEEE:function(e){let a=new NativeFunction(e,"void",["pointer","pointer","pointer","bool","int","pointer"],y);this["art::gc::Heap::GetInstances"]=function(e,t,r,n,i){a(e,t,r,0,n,i)}},_ZN3art12StackVisitorC2EPNS_6ThreadEPNS_7ContextENS0_13StackWalkKindEjb:["art::StackVisitor::StackVisitor","void",["pointer","pointer","pointer","uint","uint","bool"]],_ZN3art12StackVisitorC2EPNS_6ThreadEPNS_7ContextENS0_13StackWalkKindEmb:["art::StackVisitor::StackVisitor","void",["pointer","pointer","pointer","uint","size_t","bool"]],_ZN3art12StackVisitor9WalkStackILNS0_16CountTransitionsE0EEEvb:["art::StackVisitor::WalkStack","void",["pointer","bool"]],_ZNK3art12StackVisitor9GetMethodEv:["art::StackVisitor::GetMethod","pointer",["pointer"]],_ZNK3art12StackVisitor16DescribeLocationEv:function(e){this["art::StackVisitor::DescribeLocation"]=Po(e,["pointer"])},_ZNK3art12StackVisitor24GetCurrentQuickFrameInfoEv:function(e){this["art::StackVisitor::GetCurrentQuickFrameInfo"]=ma(e)},_ZN3art7Context6CreateEv:["art::Context::Create","pointer",[]],_ZN3art6Thread18GetLongJumpContextEv:["art::Thread::GetLongJumpContext","pointer",["pointer"]],_ZN3art6mirror5Class13GetDescriptorEPNSt3__112basic_stringIcNS2_11char_traitsIcEENS2_9allocatorIcEEEE:function(e){this["art::mirror::Class::GetDescriptor"]=e},_ZN3art6mirror5Class11GetLocationEv:function(e){this["art::mirror::Class::GetLocation"]=Po(e,["pointer"])},_ZN3art9ArtMethod12PrettyMethodEb:function(e){this["art::ArtMethod::PrettyMethod"]=Po(e,["pointer","bool"])},_ZN3art12PrettyMethodEPNS_9ArtMethodEb:function(e){this["art::ArtMethod::PrettyMethodNullSafe"]=Po(e,["pointer","bool"])},_ZN3art6Thread14CurrentFromGdbEv:["art::Thread::CurrentFromGdb","pointer",[]],_ZN3art6mirror6Object5CloneEPNS_6ThreadE:function(e){this["art::mirror::Object::Clone"]=new NativeFunction(e,"pointer",["pointer","pointer"],y)},_ZN3art6mirror6Object5CloneEPNS_6ThreadEm:function(e){let n=new NativeFunction(e,"pointer",["pointer","pointer","pointer"],y);this["art::mirror::Object::Clone"]=function(e,t){let r=NULL;return n(e,t,r)}},_ZN3art6mirror6Object5CloneEPNS_6ThreadEj:function(e){let r=new NativeFunction(e,"pointer",["pointer","pointer","uint"],y);this["art::mirror::Object::Clone"]=function(e,t){return r(e,t,0)}},_ZN3art3Dbg14SetJdwpAllowedEb:["art::Dbg::SetJdwpAllowed","void",["bool"]],_ZN3art3Dbg13ConfigureJdwpERKNS_4JDWP11JdwpOptionsE:["art::Dbg::ConfigureJdwp","void",["pointer"]],_ZN3art31InternalDebuggerControlCallback13StartDebuggerEv:["art::InternalDebuggerControlCallback::StartDebugger","void",["pointer"]],_ZN3art3Dbg9StartJdwpEv:["art::Dbg::StartJdwp","void",[]],_ZN3art3Dbg8GoActiveEv:["art::Dbg::GoActive","void",[]],_ZN3art3Dbg21RequestDeoptimizationERKNS_21DeoptimizationRequestE:["art::Dbg::RequestDeoptimization","void",["pointer"]],_ZN3art3Dbg20ManageDeoptimizationEv:["art::Dbg::ManageDeoptimization","void",[]],_ZN3art15instrumentation15Instrumentation20EnableDeoptimizationEv:["art::Instrumentation::EnableDeoptimization","void",["pointer"]],_ZN3art15instrumentation15Instrumentation20DeoptimizeEverythingEPKc:["art::Instrumentation::DeoptimizeEverything","void",["pointer","pointer"]],_ZN3art15instrumentation15Instrumentation20DeoptimizeEverythingEv:function(e){let r=new NativeFunction(e,"void",["pointer"],y);this["art::Instrumentation::DeoptimizeEverything"]=function(e,t){r(e)}},_ZN3art7Runtime19DeoptimizeBootImageEv:["art::Runtime::DeoptimizeBootImage","void",["pointer"]],_ZN3art15instrumentation15Instrumentation10DeoptimizeEPNS_9ArtMethodE:["art::Instrumentation::Deoptimize","void",["pointer","pointer"]],_ZN3art3jni12JniIdManager14DecodeMethodIdEP10_jmethodID:["art::jni::JniIdManager::DecodeMethodId","pointer",["pointer","pointer"]],_ZN3art3jni12JniIdManager13DecodeFieldIdEP9_jfieldID:["art::jni::JniIdManager::DecodeFieldId","pointer",["pointer","pointer"]],_ZN3art11interpreter18GetNterpEntryPointEv:["art::interpreter::GetNterpEntryPoint","pointer",[]],_ZN3art7Monitor17TranslateLocationEPNS_9ArtMethodEjPPKcPi:["art::Monitor::TranslateLocation","void",["pointer","uint32","pointer","pointer"]]},variables:{_ZN3art3Dbg9gRegistryE:function(e){this.isJdwpStarted=()=>!e.readPointer().isNull()},_ZN3art3Dbg15gDebuggerActiveE:function(e){this.isDebuggerActive=()=>!!e.readU8()}},optionals:new Set(["artInterpreterToCompiledCodeBridge","_ZN3art9JavaVMExt12AddGlobalRefEPNS_6ThreadENS_6ObjPtrINS_6mirror6ObjectEEE","_ZN3art9JavaVMExt12AddGlobalRefEPNS_6ThreadEPNS_6mirror6ObjectE","_ZN3art9JavaVMExt12DecodeGlobalEPv","_ZN3art9JavaVMExt12DecodeGlobalEPNS_6ThreadEPv","_ZNK3art6Thread19DecodeGlobalJObjectEP8_jobject","_ZNK3art6Thread13DecodeJObjectEP8_jobject","_ZN3art10ThreadList10SuspendAllEPKcb","_ZN3art10ThreadList10SuspendAllEv","_ZN3art11ClassLinker12VisitClassesEPNS_12ClassVisitorE","_ZN3art11ClassLinker12VisitClassesEPFbPNS_6mirror5ClassEPvES4_","_ZNK3art11ClassLinker17VisitClassLoadersEPNS_18ClassLoaderVisitorE","_ZN3art6mirror6Object5CloneEPNS_6ThreadE","_ZN3art6mirror6Object5CloneEPNS_6ThreadEm","_ZN3art6mirror6Object5CloneEPNS_6ThreadEj","_ZN3art22IndirectReferenceTable3AddEjPNS_6mirror6ObjectE","_ZN3art22IndirectReferenceTable3AddENS_15IRTSegmentStateENS_6ObjPtrINS_6mirror6ObjectEEE","_ZN3art2gc4Heap12VisitObjectsEPFvPNS_6mirror6ObjectEPvES5_","_ZN3art2gc4Heap12GetInstancesERNS_24VariableSizedHandleScopeENS_6HandleINS_6mirror5ClassEEEiRNSt3__16vectorINS4_INS5_6ObjectEEENS8_9allocatorISB_EEEE","_ZN3art2gc4Heap12GetInstancesERNS_24VariableSizedHandleScopeENS_6HandleINS_6mirror5ClassEEEbiRNSt3__16vectorINS4_INS5_6ObjectEEENS8_9allocatorISB_EEEE","_ZN3art12StackVisitorC2EPNS_6ThreadEPNS_7ContextENS0_13StackWalkKindEjb","_ZN3art12StackVisitorC2EPNS_6ThreadEPNS_7ContextENS0_13StackWalkKindEmb","_ZN3art12StackVisitor9WalkStackILNS0_16CountTransitionsE0EEEvb","_ZNK3art12StackVisitor9GetMethodEv","_ZNK3art12StackVisitor16DescribeLocationEv","_ZNK3art12StackVisitor24GetCurrentQuickFrameInfoEv","_ZN3art7Context6CreateEv","_ZN3art6Thread18GetLongJumpContextEv","_ZN3art6mirror5Class13GetDescriptorEPNSt3__112basic_stringIcNS2_11char_traitsIcEENS2_9allocatorIcEEEE","_ZN3art6mirror5Class11GetLocationEv","_ZN3art9ArtMethod12PrettyMethodEb","_ZN3art12PrettyMethodEPNS_9ArtMethodEb","_ZN3art3Dbg13ConfigureJdwpERKNS_4JDWP11JdwpOptionsE","_ZN3art31InternalDebuggerControlCallback13StartDebuggerEv","_ZN3art3Dbg15gDebuggerActiveE","_ZN3art15instrumentation15Instrumentation20EnableDeoptimizationEv","_ZN3art15instrumentation15Instrumentation20DeoptimizeEverythingEPKc","_ZN3art15instrumentation15Instrumentation20DeoptimizeEverythingEv","_ZN3art7Runtime19DeoptimizeBootImageEv","_ZN3art15instrumentation15Instrumentation10DeoptimizeEPNS_9ArtMethodE","_ZN3art3Dbg9StartJdwpEv","_ZN3art3Dbg8GoActiveEv","_ZN3art3Dbg21RequestDeoptimizationERKNS_21DeoptimizationRequestE","_ZN3art3Dbg20ManageDeoptimizationEv","_ZN3art3Dbg9gRegistryE","_ZN3art3jni12JniIdManager14DecodeMethodIdEP10_jmethodID","_ZN3art3jni12JniIdManager13DecodeFieldIdEP9_jfieldID","_ZN3art11interpreter18GetNterpEntryPointEv","_ZN3art7Monitor17TranslateLocationEPNS_9ArtMethodEjPPKcPi"])}:{functions:{_Z20dvmDecodeIndirectRefP6ThreadP8_jobject:["dvmDecodeIndirectRef","pointer",["pointer","pointer"]],_Z15dvmUseJNIBridgeP6MethodPv:["dvmUseJNIBridge","void",["pointer","pointer"]],_Z20dvmHeapSourceGetBasev:["dvmHeapSourceGetBase","pointer",[]],_Z21dvmHeapSourceGetLimitv:["dvmHeapSourceGetLimit","pointer",[]],_Z16dvmIsValidObjectPK6Object:["dvmIsValidObject","uint8",["pointer"]],JNI_GetCreatedJavaVMs:["JNI_GetCreatedJavaVMs","int",["pointer","int","pointer"]]},variables:{gDvmJni:function(e){this.gDvmJni=e},gDvm:function(e){this.gDvm=e}}},{functions:a={},variables:o={},optionals:s=new Set}=i,l=[];for(let[t,r]of Object.entries(a)){let e=f.find(t);e!==null?typeof r=="function"?r.call(f,e):f[r[0]]=new NativeFunction(e,r[1],r[2],y):s.has(t)||l.push(t)}for(let[t,r]of Object.entries(o)){let e=f.find(t);e!==null?r.call(f,e):s.has(t)||l.push(t)}if(l.length>0)throw new Error("Java API only partially available; please file a bug. Missing: "+l.join(", "));let d=Memory.alloc(v),c=Memory.alloc(qr);if(h("JNI_GetCreatedJavaVMs",f.JNI_GetCreatedJavaVMs(d,1,c)),c.readInt()===0)return null;if(f.vm=d.readPointer(),n){let e=m(),t;e>=27?t=33554432:e>=24?t=16777216:t=0,f.kAccCompileDontBother=t;let r=f.vm.add(v).readPointer();f.artRuntime=r;let n=Hn(f),i=n.offset,a=i.instrumentation;f.artInstrumentation=a!==null?r.add(a):null,ei()>=36e7&&f.artInstrumentation!=null&&(f.artInstrumentation=f.artInstrumentation.readPointer()),f.artHeap=r.add(i.heap).readPointer(),f.artThreadList=r.add(i.threadList).readPointer();let o=r.add(i.classLinker).readPointer(),s=Di(r,n).offset,l=o.add(s.quickResolutionTrampoline).readPointer(),d=o.add(s.quickImtConflictTrampoline).readPointer(),c=o.add(s.quickGenericJniTrampoline).readPointer(),u=o.add(s.quickToInterpreterBridgeTrampoline).readPointer();f.artClassLinker={address:o,quickResolutionTrampoline:l,quickImtConflictTrampoline:d,quickGenericJniTrampoline:c,quickToInterpreterBridgeTrampoline:u};let h=new Hr(f);f.artQuickGenericJniTrampoline=Hi(c,h),f.artQuickToInterpreterBridge=Hi(u,h),f.artQuickResolutionTrampoline=Hi(l,h),f["art::JavaVMExt::AddGlobalRef"]===void 0&&(f["art::JavaVMExt::AddGlobalRef"]=yo(f)),f["art::JavaVMExt::DecodeGlobal"]===void 0&&(f["art::JavaVMExt::DecodeGlobal"]=wo(f)),f["art::ArtMethod::PrettyMethod"]===void 0&&(f["art::ArtMethod::PrettyMethod"]=f["art::ArtMethod::PrettyMethodNullSafe"]),f["art::interpreter::GetNterpEntryPoint"]!==void 0?f.artNterpEntryPoint=f["art::interpreter::GetNterpEntryPoint"]():f.artNterpEntryPoint=f.find("ExecuteNterpImpl"),_=Ea(f,h),jo(f);let p=null;Object.defineProperty(f,"jvmti",{get(){return p===null&&(p=[vi(h,this.artRuntime)]),p[0]}})}let u=t.enumerateImports().filter(e=>e.name.indexOf("_Z")===0).reduce((e,t)=>(e[t.name]=t.address,e),{});return f.$new=new NativeFunction(u._Znwm||u._Znwj,"pointer",["ulong"],y),f.$delete=new NativeFunction(u._ZdlPv,"void",["pointer"],y),oi=n?io:lo,f}function vi(o,s){let l=null;return o.perform(()=>{let e=S().find("_ZN3art7Runtime18EnsurePluginLoadedEPKcPNSt3__112basic_stringIcNS3_11char_traitsIcEENS3_9allocatorIcEEEE");if(e===null)return;let t=new NativeFunction(e,"bool",["pointer","pointer","pointer"]),r=Memory.alloc(v);if(!t(s,Memory.allocUtf8String("libopenjdkjvmti.so"),r))return;let n=qe.v1_2|1073741824,i=o.tryGetEnvHandle(n);if(i===null)return;l=new c(i,o);let a=Memory.alloc(8);a.writeU64(We.canTagObjects),l.addCapabilities(a)!==0&&(l=null)}),l}function yi(e,t){S().flavor==="art"&&e.getClassName(t)}function wi(e){return{offset:v===4?{globalsLock:32,globals:72}:{globalsLock:64,globals:112}}}function Ei(e){let r=e.vm,o=e.artRuntime,n=v===4?200:384,i=n+100*v,s=m(),l=Yn(),{isApiLevel34OrApexEquivalent:d}=e,c=null;for(let t=n;t!==i;t+=v)if(o.add(t).readPointer().equals(r)){let e,a=null;s>=33||l==="Tiramisu"||d?(e=[t-4*v],a=t-v):s>=30||l==="R"?(e=[t-3*v,t-4*v],a=t-v):s>=29?e=[t-2*v]:s>=27?e=[t-Jn-3*v]:e=[t-Jn-2*v];for(let i of e){let e=i-v,t=e-v,r;d?r=t-9*v:s>=24?r=t-8*v:s>=23?r=t-7*v:r=t-4*v;let n={offset:{heap:r,threadList:t,internTable:e,classLinker:i,jniIdManager:a}};if(zi(o,n)!==null){c=n;break}}break}if(c===null)throw new Error("Unable to determine Runtime field offsets");let t=ei()>=36e7;return c.offset.instrumentation=t?Mi(e):Ni(e),c.offset.jniIdsIndirection=xi(e),c}var Si={ia32:Li,x64:Li,arm:ki,arm64:Ci};function Ni(e){let t=e["art::Runtime::DeoptimizeBootImage"];return t===void 0?null:p(t,Si[Process.arch],{limit:30})}function Li(e){if(e.mnemonic!=="lea")return null;let t=e.operands[1].value.disp;return t<256||t>1024?null:t}function ki(e){if(e.mnemonic!=="add.w")return null;let t=e.operands;if(t.length!==3)return null;let r=t[2];return r.type!=="imm"?null:r.value}function Ci(e){if(e.mnemonic!=="add")return null;let t=e.operands;if(t.length!==3||t[0].value==="sp"||t[1].value==="sp")return null;let r=t[2];if(r.type!=="imm")return null;let n=r.value.valueOf();return n<256||n>1024?null:n}var ji={ia32:Ii,x64:Ii,arm:Pi,arm64:Ri};function Mi(e){let t=e["art::Runtime::DeoptimizeBootImage"];return t===void 0?null:p(t,ji[Process.arch],{limit:30})}function Ii(e){if(e.mnemonic!=="mov")return null;let t=e.operands;if(t[0].value!=="rax")return null;let r=t[1];if(r.type!=="mem")return null;let n=r.value;if(n.base!=="rdi")return null;let i=n.disp;return i<256||i>1024?null:i}function Pi(e){return null}function Ri(e){if(e.mnemonic!=="ldr")return null;let t=e.operands;if(t[0].value==="x0")return null;let r=t[1].value;if(r.base!=="x0")return null;let n=r.disp;return n<256||n>1024?null:n}var Ai={ia32:Ti,x64:Ti,arm:Ui,arm64:Oi};function xi(e){let t=e.find("_ZN3art7Runtime12SetJniIdTypeENS_9JniIdTypeE");if(t===null)return null;let r=p(t,Ai[Process.arch],{limit:20});if(r===null)throw new Error("Unable to determine Runtime.jni_ids_indirection_ offset");return r}function Ti(e){return e.mnemonic==="cmp"?e.operands[0].value.disp:null}function Ui(e){return e.mnemonic==="ldr.w"?e.operands[1].value.disp:null}function Oi(e,t){if(t===null)return null;let{mnemonic:r}=e,{mnemonic:n}=t;return r==="cmp"&&n==="ldr"||r==="bl"&&n==="str"?t.operands[1].value.disp:null}function Fi(){let e={"4-21":136,"4-22":136,"4-23":172,"4-24":196,"4-25":196,"4-26":196,"4-27":196,"4-28":212,"4-29":172,"4-30":180,"4-31":180,"8-21":224,"8-22":224,"8-23":296,"8-24":344,"8-25":344,"8-26":352,"8-27":352,"8-28":392,"8-29":328,"8-30":336,"8-31":336}[`${v}-${m()}`];if(e===void 0)throw new Error("Unable to determine Instrumentation field offsets");return{offset:{forcedInterpretOnly:4,deoptimizationEnabled:e}}}function Di(e,t){let r=zi(e,t);if(r===null)throw new Error("Unable to determine ClassLinker field offsets");return r}function zi(e,t){if(ai!==null)return ai;let{classLinker:r,internTable:n}=t.offset,i=e.add(r).readPointer(),a=e.add(n).readPointer(),o=v===4?100:200,s=o+100*v,l=m(),d=null;for(let n=o;n!==s;n+=v)if(i.add(n).readPointer().equals(a)){let e;l>=30||Yn()==="R"?e=6:l>=29?e=4:l>=23?e=3:e=5;let t=n+e*v,r;l>=23?r=t-2*v:r=t-3*v,d={offset:{quickResolutionTrampoline:r,quickImtConflictTrampoline:t-v,quickGenericJniTrampoline:t,quickToInterpreterBridgeTrampoline:t+v}};break}return d!==null&&(ai=d),d}function $i(g){let b=null;return g.perform(u=>{let e=Ji(g),t=E(g),h={artArrayLengthSize:4,artArrayEntrySize:e.size,artArrayMax:50},p={artArrayLengthSize:v,artArrayEntrySize:t.size,artArrayMax:100},f=(e,t,r)=>{let n=e.add(t).readPointer();if(n.isNull())return null;let i=r===4?n.readU32():n.readU64().valueOf();return i<=0?null:{length:i,data:n.add(r)}},_=(e,n,i,a)=>{try{let t=f(e,n,a.artArrayLengthSize);if(t===null)return!1;let r=Math.min(t.length,a.artArrayMax);for(let e=0;e!==r;e++)if(t.data.add(e*a.artArrayEntrySize).equals(i))return!0}catch{}return!1},r=u.findClass("java/lang/Thread"),m=u.newGlobalRef(r);try{let t;w(g,u,e=>{t=S()["art::JavaVMExt::DecodeGlobal"](g,e,m)});let r=Ba(u.getFieldId(m,"name","Ljava/lang/String;")),n=Ba(u.getStaticFieldId(m,"MAX_PRIORITY","I")),i=-1,a=-1;for(let e=0;e!==256;e+=4)i===-1&&_(t,e,n,h)&&(i=e),a===-1&&_(t,e,r,h)&&(a=e);if(a===-1||i===-1)throw new Error("Unable to find fields in java/lang/Thread; please file a bug");let e=a!==i?i:0,o=a,s=-1,l=Ja(u.getMethodId(m,"getName","()Ljava/lang/String;"));for(let e=0;e!==256;e+=4)s===-1&&_(t,e,l,p)&&(s=e);if(s===-1)throw new Error("Unable to find methods in java/lang/Thread; please file a bug");let d=-1,c=f(t,s,p.artArrayLengthSize).length;for(let e=s;e!==256;e+=4)if(t.add(e).readU16()===c){d=e;break}if(d===-1)throw new Error("Unable to find copied methods in java/lang/Thread; please file a bug");b={offset:{ifields:o,methods:s,sfields:e,copiedMethodsOffset:d}}}finally{u.deleteLocalRef(r),u.deleteGlobalRef(m)}}),b}function Vi(e){let f=S(),_;return e.perform(e=>{let t=e.findClass("android/os/Process"),n=Ja(e.getStaticMethodId(t,"getElapsedCpuTime","()J"));e.deleteLocalRef(t);let r=Process.getModuleByName("libandroid_runtime.so"),i=r.base,a=i.add(r.size),o=m(),s=o<=21?8:v,l=Yr|en|tn|rn,d=~(on|un|cn)>>>0,c=null,u=null,h=2;for(let r=0;r!==64&&h!==0;r+=4){let t=n.add(r);if(c===null){let e=t.readPointer();e.compare(i)>=0&&e.compare(a)<0&&(c=r,h--)}u===null&&(t.readU32()&d)===l&&(u=r,h--)}if(h!==0)throw new Error("Unable to determine ArtMethod field offsets");let p=c+s;_={size:o<=21?p+32:p+v,offset:{jniCode:c,quickCode:p,accessFlags:u}},"artInterpreterToCompiledCodeBridge"in f&&(_.offset.interpreterCode=c-s)}),_}function Ji(e){let t=m();return t>=23?{size:16,offset:{accessFlags:4}}:t>=21?{size:24,offset:{accessFlags:12}}:null}function Bi(e){let d=m(),c;return e.perform(e=>{let t=Qi(e),r=e.handle,n=null,i=null,a=null,o=null,s=null,l=null;for(let e=144;e!==256;e+=v)if(t.add(e).readPointer().equals(r)){i=e-6*v,s=e-4*v,l=e+2*v,d<=22&&(i-=v,n=i-v-9*8-3*4,a=e+6*v,s-=v,l-=v),o=e+9*v,d<=22&&(o+=2*v+4,v===8&&(o+=4)),d>=23&&(o+=v);break}if(o===null)throw new Error("Unable to determine ArtThread field offsets");c={offset:{isExceptionReportedToInstrumentation:n,exception:i,throwLocation:a,topHandleScope:o,managedStack:s,self:l}}}),c}function Gi(){return m()>=23?{offset:{topQuickFrame:0,link:v}}:{offset:{topQuickFrame:2*v,link:0}}}var Zi={ia32:qi,x64:qi,arm:Wi,arm64:Ki};function Hi(a,e){let o;return e.perform(e=>{let t=Qi(e),r=Zi[Process.arch],n=Instruction.parse(a),i=r(n);i!==null?o=t.add(i).readPointer():o=a}),o}function qi(e){return e.mnemonic==="jmp"?e.operands[0].value.disp:null}function Wi(e){return e.mnemonic==="ldr.w"?e.operands[1].value.disp:null}function Ki(e){return e.mnemonic==="ldr"?e.operands[1].value.disp:null}function Qi(e){return e.handle.add(v).readPointer()}function Xi(){return aa("ro.build.version.release")}function Yi(){return aa("ro.build.version.codename")}function ea(){return parseInt(aa("ro.build.version.sdk"),10)}function ta(){try{let e=File.readAllText("/proc/self/mountinfo"),i=null,a=new Map;for(let n of e.trimEnd().split(`
`)){let e=n.split(" "),t=e[4];if(!t.startsWith("/apex/com.android.art"))continue;let r=e[10];t.includes("@")?a.set(r,t.split("@")[1]):i=r}let t=a.get(i);return t!==void 0?parseInt(t):ra()}catch{return ra()}}function ra(){return m()*1e7}var na=null,ia=92;function aa(e){na===null&&(na=new NativeFunction(Process.getModuleByName("libc.so").getExportByName("__system_property_get"),"int",["pointer","pointer"],y));let t=Memory.alloc(ia);return na(Memory.allocUtf8String(e),t),t.readUtf8String()}function w(e,t,r){let n=Qn(e,t),i=Qi(t).toString();if(ni[i]=r,n(t.handle),ni[i]!==void 0)throw delete ni[i],new Error("Unable to perform state transition; please file a bug")}function oa(e,t){let r=new NativeCallback(sa,"void",["pointer"]);return So(e,t,r)}function sa(e){let t=e.toString(),r=ni[t];delete ni[t],r(e)}function la(e){let t=S(),r=t.artThreadList;t["art::ThreadList::SuspendAll"](r,Memory.allocUtf8String("frida"),!1?1:0);try{e()}finally{t["art::ThreadList::ResumeAll"](r)}}var da=class{constructor(r){let e=Memory.alloc(4*v),t=e.add(v);e.writePointer(t);let n=new NativeCallback((e,t)=>r(t)===!0?1:0,"bool",["pointer","pointer"]);t.add(2*v).writePointer(n),this.handle=e,this._onVisit=n}};function ca(t){return S()["art::ClassLinker::VisitClasses"]instanceof NativeFunction?new da(t):new NativeCallback(e=>t(e)===!0?1:0,"bool",["pointer","pointer"])}var ua=class{constructor(r){let e=Memory.alloc(4*v),t=e.add(v);e.writePointer(t);let n=new NativeCallback((e,t)=>{r(t)},"void",["pointer","pointer"]);t.add(2*v).writePointer(n),this.handle=e,this._onVisit=n}};function ha(e){return new ua(e)}var pa={"include-inlined-frames":0,"skip-inlined-frames":1},fa=class{constructor(e,t,r,n=0,i=!0){let a=S(),o=512,s=3*v,l=Memory.alloc(o+s);a["art::StackVisitor::StackVisitor"](l,e,t,pa[r],n,i?1:0);let d=l.add(o);l.writePointer(d);let c=new NativeCallback(this._visitFrame.bind(this),"bool",["pointer"]);d.add(2*v).writePointer(c),this.handle=l,this._onVisitFrame=c;let u=l.add(v===4?12:24);this._curShadowFrame=u,this._curQuickFrame=u.add(v),this._curQuickFramePc=u.add(2*v),this._curOatQuickMethodHeader=u.add(3*v),this._getMethodImpl=a["art::StackVisitor::GetMethod"],this._descLocImpl=a["art::StackVisitor::DescribeLocation"],this._getCQFIImpl=a["art::StackVisitor::GetCurrentQuickFrameInfo"]}walkStack(e=!1){S()["art::StackVisitor::WalkStack"](this.handle,e?1:0)}_visitFrame(){return this.visitFrame()?1:0}visitFrame(){throw new Error("Subclass must implement visitFrame")}getMethod(){let e=this._getMethodImpl(this.handle);return e.isNull()?null:new _a(e)}getCurrentQuickFramePc(){return this._curQuickFramePc.readPointer()}getCurrentQuickFrame(){return this._curQuickFrame.readPointer()}getCurrentShadowFrame(){return this._curShadowFrame.readPointer()}describeLocation(){let e=new Ro;return this._descLocImpl(e,this.handle),e.disposeToString()}getCurrentOatQuickMethodHeader(){return this._curOatQuickMethodHeader.readPointer()}getCurrentQuickFrameInfo(){return this._getCQFIImpl(this.handle)}},_a=class{constructor(e){this.handle=e}prettyMethod(e=!0){let t=new Ro;return S()["art::ArtMethod::PrettyMethod"](t,this.handle,e?1:0),t.disposeToString()}toString(){return`ArtMethod(handle=${this.handle})`}};function ma(r){return function(e){let t=Memory.alloc(12);return ti(r)(t,e),{frameSizeInBytes:t.readU32(),coreSpillMask:t.add(4).readU32(),fpSpillMask:t.add(8).readU32()}}}function ga(t){let e=NULL;switch(Process.arch){case"ia32":e=ya(32,e=>{e.putMovRegRegOffsetPtr("ecx","esp",4),e.putMovRegRegOffsetPtr("edx","esp",8),e.putCallAddressWithArguments(t,["ecx","edx"]),e.putMovRegReg("esp","ebp"),e.putPopReg("ebp"),e.putRet()});break;case"x64":e=ya(32,e=>{e.putPushReg("rdi"),e.putCallAddressWithArguments(t,["rsi"]),e.putPopReg("rdi"),e.putMovRegPtrReg("rdi","rax"),e.putMovRegOffsetPtrReg("rdi",8,"edx"),e.putRet()});break;case"arm":e=ya(16,e=>{e.putCallAddressWithArguments(t,["r0","r1"]),e.putPopRegs(["r0","lr"]),e.putMovRegReg("pc","lr")});break;case"arm64":e=ya(64,e=>{e.putPushRegReg("x0","lr"),e.putCallAddressWithArguments(t,["x1"]),e.putPopRegReg("x2","lr"),e.putStrRegRegOffset("x0","x2",0),e.putStrRegRegOffset("w1","x2",8),e.putRet()});break}return new NativeFunction(e,"void",["pointer","pointer"],y)}var ba={ia32:globalThis.X86Relocator,x64:globalThis.X86Relocator,arm:globalThis.ThumbRelocator,arm64:globalThis.Arm64Relocator},va={ia32:globalThis.X86Writer,x64:globalThis.X86Writer,arm:globalThis.ThumbWriter,arm64:globalThis.Arm64Writer};function ya(r,n){ci===null&&(ci=Memory.alloc(Process.pageSize));let i=ci.add(ui),e=Process.arch,a=va[e];return Memory.patchCode(i,r,e=>{let t=new a(e,{pc:i});if(n(t),t.flush(),t.offset>r)throw new Error(`Wrote ${t.offset}, exceeding maximum of ${r}`)}),ui+=r,e==="arm"?i.or(1):i}function wa(e,t){Sa(t),ja(t)}function Ea(e,t){let r=Wn(t).offset,n=Kn().offset,i=`
#include <gum/guminterceptor.h>

extern GMutex lock;
extern GHashTable * methods;
extern GHashTable * replacements;
extern gpointer last_seen_art_method;

extern gpointer get_oat_quick_method_header_impl (gpointer method, gpointer pc);

void
init (void)
{
  g_mutex_init (&lock);
  methods = g_hash_table_new_full (NULL, NULL, NULL, NULL);
  replacements = g_hash_table_new_full (NULL, NULL, NULL, NULL);
}

void
finalize (void)
{
  g_hash_table_unref (replacements);
  g_hash_table_unref (methods);
  g_mutex_clear (&lock);
}

gboolean
is_replacement_method (gpointer method)
{
  gboolean is_replacement;

  g_mutex_lock (&lock);

  is_replacement = g_hash_table_contains (replacements, method);

  g_mutex_unlock (&lock);

  return is_replacement;
}

gpointer
get_replacement_method (gpointer original_method)
{
  gpointer replacement_method;

  g_mutex_lock (&lock);

  replacement_method = g_hash_table_lookup (methods, original_method);

  g_mutex_unlock (&lock);

  return replacement_method;
}

void
set_replacement_method (gpointer original_method,
                        gpointer replacement_method)
{
  g_mutex_lock (&lock);

  g_hash_table_insert (methods, original_method, replacement_method);
  g_hash_table_insert (replacements, replacement_method, original_method);

  g_mutex_unlock (&lock);
}

void
synchronize_replacement_methods (guint quick_code_offset,
                                 void * nterp_entrypoint,
                                 void * quick_to_interpreter_bridge)
{
  GHashTableIter iter;
  gpointer hooked_method, replacement_method;

  g_mutex_lock (&lock);

  g_hash_table_iter_init (&iter, methods);
  while (g_hash_table_iter_next (&iter, &hooked_method, &replacement_method))
  {
    void ** quick_code;

    *((uint32_t *) replacement_method) = *((uint32_t *) hooked_method);

    quick_code = hooked_method + quick_code_offset;
    if (*quick_code == nterp_entrypoint)
      *quick_code = quick_to_interpreter_bridge;
  }

  g_mutex_unlock (&lock);
}

void
delete_replacement_method (gpointer original_method)
{
  gpointer replacement_method;

  g_mutex_lock (&lock);

  replacement_method = g_hash_table_lookup (methods, original_method);
  if (replacement_method != NULL)
  {
    g_hash_table_remove (methods, original_method);
    g_hash_table_remove (replacements, replacement_method);
  }

  g_mutex_unlock (&lock);
}

gpointer
translate_method (gpointer method)
{
  gpointer translated_method;

  g_mutex_lock (&lock);

  translated_method = g_hash_table_lookup (replacements, method);

  g_mutex_unlock (&lock);

  return (translated_method != NULL) ? translated_method : method;
}

gpointer
find_replacement_method_from_quick_code (gpointer method,
                                         gpointer thread)
{
  gpointer replacement_method;
  gpointer managed_stack;
  gpointer top_quick_frame;
  gpointer link_managed_stack;
  gpointer * link_top_quick_frame;

  replacement_method = get_replacement_method (method);
  if (replacement_method == NULL)
    return NULL;

  /*
   * Stack check.
   *
   * Return NULL to indicate that the original method should be invoked, otherwise
   * return a pointer to the replacement ArtMethod.
   *
   * If the caller is our own JNI replacement stub, then a stack transition must
   * have been pushed onto the current thread's linked list.
   *
   * Therefore, we invoke the original method if the following conditions are met:
   *   1- The current managed stack is empty.
   *   2- The ArtMethod * inside the linked managed stack's top quick frame is the
   *      same as our replacement.
   */
  managed_stack = thread + ${r.managedStack};
  top_quick_frame = *((gpointer *) (managed_stack + ${n.topQuickFrame}));
  if (top_quick_frame != NULL)
    return replacement_method;

  link_managed_stack = *((gpointer *) (managed_stack + ${n.link}));
  if (link_managed_stack == NULL)
    return replacement_method;

  link_top_quick_frame = GSIZE_TO_POINTER (*((gsize *) (link_managed_stack + ${n.topQuickFrame})) & ~((gsize) 1));
  if (link_top_quick_frame == NULL || *link_top_quick_frame != replacement_method)
    return replacement_method;

  return NULL;
}

void
on_interpreter_do_call (GumInvocationContext * ic)
{
  gpointer method, replacement_method;

  method = gum_invocation_context_get_nth_argument (ic, 0);

  replacement_method = get_replacement_method (method);
  if (replacement_method != NULL)
    gum_invocation_context_replace_nth_argument (ic, 0, replacement_method);
}

gpointer
on_art_method_get_oat_quick_method_header (gpointer method,
                                           gpointer pc)
{
  if (is_replacement_method (method))
    return NULL;

  return get_oat_quick_method_header_impl (method, pc);
}

void
on_art_method_pretty_method (GumInvocationContext * ic)
{
  const guint this_arg_index = ${Process.arch==="arm64"?0:1};
  gpointer method;

  method = gum_invocation_context_get_nth_argument (ic, this_arg_index);
  if (method == NULL)
    gum_invocation_context_replace_nth_argument (ic, this_arg_index, last_seen_art_method);
  else
    last_seen_art_method = method;
}

void
on_leave_gc_concurrent_copying_copying_phase (GumInvocationContext * ic)
{
  GHashTableIter iter;
  gpointer hooked_method, replacement_method;

  g_mutex_lock (&lock);

  g_hash_table_iter_init (&iter, methods);
  while (g_hash_table_iter_next (&iter, &hooked_method, &replacement_method))
    *((uint32_t *) replacement_method) = *((uint32_t *) hooked_method);

  g_mutex_unlock (&lock);
}
`,a=8,o=v,s=v,l=v,d=Memory.alloc(a+o+s+l),c=d.add(a),u=c.add(o),h=u.add(s),p=e.find(v===4?"_ZN3art9ArtMethod23GetOatQuickMethodHeaderEj":"_ZN3art9ArtMethod23GetOatQuickMethodHeaderEm"),f=new CModule(i,{lock:d,methods:c,replacements:u,last_seen_art_method:h,get_oat_quick_method_header_impl:p??ptr("0xdeadbeef")}),_={exceptions:"propagate",scheduling:"exclusive"};return{handle:f,replacedMethods:{isReplacement:new NativeFunction(f.is_replacement_method,"bool",["pointer"],_),get:new NativeFunction(f.get_replacement_method,"pointer",["pointer"],_),set:new NativeFunction(f.set_replacement_method,"void",["pointer","pointer"],_),synchronize:new NativeFunction(f.synchronize_replacement_methods,"void",["uint","pointer","pointer"],_),delete:new NativeFunction(f.delete_replacement_method,"void",["pointer"],_),translate:new NativeFunction(f.translate_method,"pointer",["pointer"],_),findReplacementFromQuickCode:f.find_replacement_method_from_quick_code},getOatQuickMethodHeaderImpl:p,hooks:{Interpreter:{doCall:f.on_interpreter_do_call},ArtMethod:{getOatQuickMethodHeader:f.on_art_method_get_oat_quick_method_header,prettyMethod:f.on_art_method_pretty_method},Gc:{copyingPhase:{onLeave:f.on_leave_gc_concurrent_copying_copying_phase},runFlip:{onEnter:f.on_leave_gc_concurrent_copying_copying_phase}}}}}function Sa(e){pi||(pi=!0,Na(e),La(),ka(),Ca())}function Na(r){let e=S();[e.artQuickGenericJniTrampoline,e.artQuickToInterpreterBridge,e.artQuickResolutionTrampoline].forEach(e=>{Memory.protect(e,32,"rwx");let t=new ro(e);t.activate(r),di.push(t)})}function La(){let e=S(),t=m(),{isApiLevel34OrApexEquivalent:r}=e,n;if(t<=22)n=/^_ZN3art11interpreter6DoCallILb[0-1]ELb[0-1]EEEbPNS_6mirror9ArtMethodEPNS_6ThreadERNS_11ShadowFrameEPKNS_11InstructionEtPNS_6JValueE$/;else if(t<=33&&!r)n=/^_ZN3art11interpreter6DoCallILb[0-1]ELb[0-1]EEEbPNS_9ArtMethodEPNS_6ThreadERNS_11ShadowFrameEPKNS_11InstructionEtPNS_6JValueE$/;else if(r)n=/^_ZN3art11interpreter6DoCallILb[0-1]EEEbPNS_9ArtMethodEPNS_6ThreadERNS_11ShadowFrameEPKNS_11InstructionEtbPNS_6JValueE$/;else throw new Error("Unable to find method invocation in ART; please file a bug");let i=e.module,a=[...i.enumerateExports(),...i.enumerateSymbols()].filter(e=>n.test(e.name));if(a.length===0)throw new Error("Unable to find method invocation in ART; please file a bug");for(let e of a)Interceptor.attach(e.address,_.hooks.Interpreter.doCall)}function ka(){let e=S(),t=e.module.findSymbolByName("_ZN3art2gc4Heap22CollectGarbageInternalENS0_9collector6GcTypeENS0_7GcCauseEbj");if(t===null)return;let{artNterpEntryPoint:r,artQuickToInterpreterBridge:n}=e,i=E(e.vm).offset.quickCode;Interceptor.attach(t,{onLeave(){_.replacedMethods.synchronize(i,r,n)}})}function Ca(){let e=[["_ZN3art11ClassLinker26VisiblyInitializedCallback22MarkVisiblyInitializedEPNS_6ThreadE","e90340f8 : ff0ff0ff"],["_ZN3art11ClassLinker26VisiblyInitializedCallback29AdjustThreadVisibilityCounterEPNS_6ThreadEl","7f0f00f9 : 1ffcffff"]],s=S(),l=s.module;for(let[a,o]of e){let e=l.findSymbolByName(a);if(e===null)continue;let t=Memory.scanSync(e,8192,o);if(t.length===0)return;let{artNterpEntryPoint:r,artQuickToInterpreterBridge:n}=s,i=E(s.vm).offset.quickCode;Interceptor.attach(t[0].address,function(){_.replacedMethods.synchronize(i,r,n)});return}}function ja(e){if(hi)return;if(hi=!0,!Ra()){let{getOatQuickMethodHeaderImpl:e}=_;if(e===null)return;try{Interceptor.replace(e,_.hooks.ArtMethod.getOatQuickMethodHeader)}catch{}}let t=m(),r=null,n=S();t>28?r=n.find("_ZN3art2gc9collector17ConcurrentCopying12CopyingPhaseEv"):t>22&&(r=n.find("_ZN3art2gc9collector17ConcurrentCopying12MarkingPhaseEv")),r!==null&&Interceptor.attach(r,_.hooks.Gc.copyingPhase);let i=null;i=n.find("_ZN3art6Thread15RunFlipFunctionEPS0_"),i===null&&(i=n.find("_ZN3art6Thread15RunFlipFunctionEPS0_b")),i!==null&&Interceptor.attach(i,_.hooks.Gc.runFlip)}var Ma={arm:{signatures:[{pattern:["b0 68","01 30","0c d0","1b 98",":","c0 ff","c0 ff","00 ff","00 2f"],validateMatch:Ia},{pattern:["d8 f8 08 00","01 30","0c d0","1b 98",":","f0 ff ff 0f","ff ff","00 ff","00 2f"],validateMatch:Ia},{pattern:["b0 68","01 30","40 f0 c3 80","00 25",":","c0 ff","c0 ff","c0 fb 00 d0","ff f8"],validateMatch:Ia}],instrument:Ta},arm64:{signatures:[{pattern:["0a 40 b9","1f 05 00 31","40 01 00 54","88 39 00 f0",":","fc ff ff","1f fc ff ff","1f 00 00 ff","00 00 00 9f"],offset:1,validateMatch:Pa},{pattern:["0a 40 b9","1f 05 00 31","40 01 00 54","00 0e 40 f9",":","fc ff ff","1f fc ff ff","1f 00 00 ff","00 fc ff ff"],offset:1,validateMatch:Pa},{pattern:["0a 40 b9","1f 05 00 31","01 34 00 54","e0 03 1f aa",":","fc ff ff","1f fc ff ff","1f 00 00 ff","e0 ff ff ff"],offset:1,validateMatch:Pa}],instrument:Ua}};function Ia({address:e,size:t}){let r=Instruction.parse(e.or(1)),[n,i]=r.operands,a=i.value.base,o=n.value,s=Instruction.parse(r.next.add(2)),l=ptr(s.operands[0].value),d=s.address.add(s.size),c,u;return s.mnemonic==="beq"?(c=d,u=l):(c=l,u=d),p(c.or(1),h,{limit:3});function h(e){let{mnemonic:t}=e;if(!(t==="ldr"||t==="ldr.w"))return null;let{base:r,disp:n}=e.operands[1].value;return r===a&&n===20?{methodReg:a,scratchReg:o,target:{whenTrue:l,whenRegularMethod:c,whenRuntimeMethod:u}}:null}}function Pa({address:e,size:t}){let[r,n]=Instruction.parse(e).operands,i=n.value.base,a="x"+r.value.substring(1),o=Instruction.parse(e.add(8)),s=ptr(o.operands[0].value),l=e.add(12),d,c;return o.mnemonic==="b.eq"?(d=l,c=s):(d=s,c=l),p(d,u,{limit:3});function u(e){if(e.mnemonic!=="ldr")return null;let{base:t,disp:r}=e.operands[1].value;return t===i&&r===24?{methodReg:i,scratchReg:a,target:{whenTrue:s,whenRegularMethod:d,whenRuntimeMethod:c}}:null}}function Ra(){if(m()<31)return!1;let e=Ma[Process.arch];if(e===void 0)return!1;let o=e.signatures.map(({pattern:e,offset:t=0,validateMatch:r=Aa})=>({pattern:new MatchPattern(e.join("")),offset:t,validateMatch:r})),s=[];for(let{base:i,size:a}of S().module.enumerateRanges("--x"))for(let{pattern:t,offset:r,validateMatch:n}of o){let e=Memory.scanSync(i,a,t).map(({address:e,size:t})=>({address:e.sub(r),size:t+r})).filter(e=>{let t=n(e);return t===null?!1:(e.validationResult=t,!0)});s.push(...e)}return s.length===0?!1:(s.forEach(e.instrument),!0)}function Aa(){return{}}var xa=class{constructor(e,t,r){this.address=e,this.size=t,this.originalCode=e.readByteArray(t),this.trampoline=r}revert(){Memory.patchCode(this.address,this.size,e=>{e.writeByteArray(this.originalCode)})}};function Ta({address:s,size:e,validationResult:t}){let{methodReg:l,target:d}=t,c=Memory.alloc(Process.pageSize),u=e;Memory.patchCode(c,256,e=>{let t=new ThumbWriter(e,{pc:c}),r=new ThumbRelocator(s,t);for(let e=0;e!==2;e++)r.readOne();r.writeAll(),r.readOne(),r.skipOne(),t.putBCondLabel("eq","runtime_or_replacement_method");let n=[45,237,16,10];t.putBytes(n);let i=["r0","r1","r2","r3"];t.putPushRegs(i),t.putCallAddressWithArguments(_.replacedMethods.isReplacement,[l]),t.putCmpRegImm("r0",0),t.putPopRegs(i);let a=[189,236,16,10];t.putBytes(a),t.putBCondLabel("ne","runtime_or_replacement_method"),t.putBLabel("regular_method"),r.readOne();let o=r.input.address.equals(d.whenRegularMethod);for(t.putLabel(o?"regular_method":"runtime_or_replacement_method"),r.writeOne();u<10;){let e=r.readOne();if(e===0){u=10;break}u=e}r.writeAll(),t.putBranchAddress(s.add(u+1)),t.putLabel(o?"runtime_or_replacement_method":"regular_method"),t.putBranchAddress(d.whenTrue),t.flush()}),si.push(new xa(s,u,c)),Memory.patchCode(s,u,e=>{let t=new ThumbWriter(e,{pc:s});t.putLdrRegAddress("pc",c.or(1)),t.flush()})}function Ua({address:s,size:e,validationResult:t}){let{methodReg:l,scratchReg:r,target:d}=t,c=Memory.alloc(Process.pageSize);Memory.patchCode(c,256,e=>{let t=new Arm64Writer(e,{pc:c}),r=new Arm64Relocator(s,t);for(let e=0;e!==2;e++)r.readOne();r.writeAll(),r.readOne(),r.skipOne(),t.putBCondLabel("eq","runtime_or_replacement_method");let n=["d0","d1","d2","d3","d4","d5","d6","d7","x0","x1","x2","x3","x4","x5","x6","x7","x8","x9","x10","x11","x12","x13","x14","x15","x16","x17"],i=n.length;for(let e=0;e!==i;e+=2)t.putPushRegReg(n[e],n[e+1]);t.putCallAddressWithArguments(_.replacedMethods.isReplacement,[l]),t.putCmpRegReg("x0","xzr");for(let e=i-2;e>=0;e-=2)t.putPopRegReg(n[e],n[e+1]);t.putBCondLabel("ne","runtime_or_replacement_method"),t.putBLabel("regular_method"),r.readOne();let a=r.input,o=a.address.equals(d.whenRegularMethod);t.putLabel(o?"regular_method":"runtime_or_replacement_method"),r.writeOne(),t.putBranchAddress(a.next),t.putLabel(o?"runtime_or_replacement_method":"regular_method"),t.putBranchAddress(d.whenTrue),t.flush()}),si.push(new xa(s,e,c)),Memory.patchCode(s,e,e=>{let t=new Arm64Writer(e,{pc:s});t.putLdrRegAddress(r,c),t.putBrReg(r),t.flush()})}function Oa(e){return new oi(e)}function Fa(e){return _.replacedMethods.translate(e)}function Da(e,t={}){let{limit:r=16}=t,n=e.getEnv();return fi===null&&(fi=za(e,n)),fi.backtrace(n,r)}function za(e,t){let r=S(),n=Memory.alloc(Process.pointerSize),i=new CModule(`
#include <glib.h>
#include <stdbool.h>
#include <string.h>
#include <gum/gumtls.h>
#include <json-glib/json-glib.h>

typedef struct _ArtBacktrace ArtBacktrace;
typedef struct _ArtStackFrame ArtStackFrame;

typedef struct _ArtStackVisitor ArtStackVisitor;
typedef struct _ArtStackVisitorVTable ArtStackVisitorVTable;

typedef struct _ArtClass ArtClass;
typedef struct _ArtMethod ArtMethod;
typedef struct _ArtThread ArtThread;
typedef struct _ArtContext ArtContext;

typedef struct _JNIEnv JNIEnv;

typedef struct _StdString StdString;
typedef struct _StdTinyString StdTinyString;
typedef struct _StdLargeString StdLargeString;

typedef enum {
  STACK_WALK_INCLUDE_INLINED_FRAMES,
  STACK_WALK_SKIP_INLINED_FRAMES,
} StackWalkKind;

struct _StdTinyString
{
  guint8 unused;
  gchar data[(3 * sizeof (gpointer)) - 1];
};

struct _StdLargeString
{
  gsize capacity;
  gsize size;
  gchar * data;
};

struct _StdString
{
  union
  {
    guint8 flags;
    StdTinyString tiny;
    StdLargeString large;
  };
};

struct _ArtBacktrace
{
  GChecksum * id;
  GArray * frames;
  gchar * frames_json;
};

struct _ArtStackFrame
{
  ArtMethod * method;
  gsize dexpc;
  StdString description;
};

struct _ArtStackVisitorVTable
{
  void (* unused1) (void);
  void (* unused2) (void);
  bool (* visit) (ArtStackVisitor * visitor);
};

struct _ArtStackVisitor
{
  ArtStackVisitorVTable * vtable;

  guint8 padding[512];

  ArtStackVisitorVTable vtable_storage;

  ArtBacktrace * backtrace;
};

struct _ArtMethod
{
  guint32 declaring_class;
  guint32 access_flags;
};

extern GumTlsKey current_backtrace;

extern void (* perform_art_thread_state_transition) (JNIEnv * env);

extern ArtContext * art_make_context (ArtThread * thread);

extern void art_stack_visitor_init (ArtStackVisitor * visitor, ArtThread * thread, void * context, StackWalkKind walk_kind,
    size_t num_frames, bool check_suspended);
extern void art_stack_visitor_walk_stack (ArtStackVisitor * visitor, bool include_transitions);
extern ArtMethod * art_stack_visitor_get_method (ArtStackVisitor * visitor);
extern void art_stack_visitor_describe_location (StdString * description, ArtStackVisitor * visitor);
extern ArtMethod * translate_method (ArtMethod * method);
extern void translate_location (ArtMethod * method, guint32 pc, const gchar ** source_file, gint32 * line_number);
extern void get_class_location (StdString * result, ArtClass * klass);
extern void cxx_delete (void * mem);
extern unsigned long strtoul (const char * str, char ** endptr, int base);

static bool visit_frame (ArtStackVisitor * visitor);
static void art_stack_frame_destroy (ArtStackFrame * frame);

static void append_jni_type_name (GString * s, const gchar * name, gsize length);

static void std_string_destroy (StdString * str);
static gchar * std_string_get_data (StdString * str);

void
init (void)
{
  current_backtrace = gum_tls_key_new ();
}

void
finalize (void)
{
  gum_tls_key_free (current_backtrace);
}

ArtBacktrace *
_create (JNIEnv * env,
         guint limit)
{
  ArtBacktrace * bt;

  bt = g_new (ArtBacktrace, 1);
  bt->id = g_checksum_new (G_CHECKSUM_SHA1);
  bt->frames = (limit != 0)
      ? g_array_sized_new (FALSE, FALSE, sizeof (ArtStackFrame), limit)
      : g_array_new (FALSE, FALSE, sizeof (ArtStackFrame));
  g_array_set_clear_func (bt->frames, (GDestroyNotify) art_stack_frame_destroy);
  bt->frames_json = NULL;

  gum_tls_key_set_value (current_backtrace, bt);

  perform_art_thread_state_transition (env);

  gum_tls_key_set_value (current_backtrace, NULL);

  return bt;
}

void
_on_thread_state_transition_complete (ArtThread * thread)
{
  ArtContext * context;
  ArtStackVisitor visitor = {
    .vtable_storage = {
      .visit = visit_frame,
    },
  };

  context = art_make_context (thread);

  art_stack_visitor_init (&visitor, thread, context, STACK_WALK_SKIP_INLINED_FRAMES, 0, true);
  visitor.vtable = &visitor.vtable_storage;
  visitor.backtrace = gum_tls_key_get_value (current_backtrace);

  art_stack_visitor_walk_stack (&visitor, false);

  cxx_delete (context);
}

static bool
visit_frame (ArtStackVisitor * visitor)
{
  ArtBacktrace * bt = visitor->backtrace;
  ArtStackFrame frame;
  const gchar * description, * dexpc_part;

  frame.method = art_stack_visitor_get_method (visitor);

  art_stack_visitor_describe_location (&frame.description, visitor);

  description = std_string_get_data (&frame.description);
  if (strstr (description, " '<") != NULL)
    goto skip;

  dexpc_part = strstr (description, " at dex PC 0x");
  if (dexpc_part == NULL)
    goto skip;
  frame.dexpc = strtoul (dexpc_part + 13, NULL, 16);

  g_array_append_val (bt->frames, frame);

  g_checksum_update (bt->id, (guchar *) &frame.method, sizeof (frame.method));
  g_checksum_update (bt->id, (guchar *) &frame.dexpc, sizeof (frame.dexpc));

  return true;

skip:
  std_string_destroy (&frame.description);
  return true;
}

static void
art_stack_frame_destroy (ArtStackFrame * frame)
{
  std_string_destroy (&frame->description);
}

void
_destroy (ArtBacktrace * backtrace)
{
  g_free (backtrace->frames_json);
  g_array_free (backtrace->frames, TRUE);
  g_checksum_free (backtrace->id);
  g_free (backtrace);
}

const gchar *
_get_id (ArtBacktrace * backtrace)
{
  return g_checksum_get_string (backtrace->id);
}

const gchar *
_get_frames (ArtBacktrace * backtrace)
{
  GArray * frames = backtrace->frames;
  JsonBuilder * b;
  guint i;
  JsonNode * root;

  if (backtrace->frames_json != NULL)
    return backtrace->frames_json;

  b = json_builder_new_immutable ();

  json_builder_begin_array (b);

  for (i = 0; i != frames->len; i++)
  {
    ArtStackFrame * frame = &g_array_index (frames, ArtStackFrame, i);
    gchar * description, * ret_type, * paren_open, * paren_close, * arg_types, * token, * method_name, * class_name;
    GString * signature;
    gchar * cursor;
    ArtMethod * translated_method;
    StdString location;
    gsize dexpc;
    const gchar * source_file;
    gint32 line_number;

    description = std_string_get_data (&frame->description);

    ret_type = strchr (description, '\\'') + 1;

    paren_open = strchr (ret_type, '(');
    paren_close = strchr (paren_open, ')');
    *paren_open = '\\0';
    *paren_close = '\\0';

    arg_types = paren_open + 1;

    token = strrchr (ret_type, '.');
    *token = '\\0';

    method_name = token + 1;

    token = strrchr (ret_type, ' ');
    *token = '\\0';

    class_name = token + 1;

    signature = g_string_sized_new (128);

    append_jni_type_name (signature, class_name, method_name - class_name - 1);
    g_string_append_c (signature, ',');
    g_string_append (signature, method_name);
    g_string_append (signature, ",(");

    if (arg_types != paren_close)
    {
      for (cursor = arg_types; cursor != NULL;)
      {
        gsize length;
        gchar * next;

        token = strstr (cursor, ", ");
        if (token != NULL)
        {
          length = token - cursor;
          next = token + 2;
        }
        else
        {
          length = paren_close - cursor;
          next = NULL;
        }

        append_jni_type_name (signature, cursor, length);

        cursor = next;
      }
    }

    g_string_append_c (signature, ')');

    append_jni_type_name (signature, ret_type, class_name - ret_type - 1);

    translated_method = translate_method (frame->method);
    dexpc = (translated_method == frame->method) ? frame->dexpc : 0;

    get_class_location (&location, GSIZE_TO_POINTER (translated_method->declaring_class));

    translate_location (translated_method, dexpc, &source_file, &line_number);

    json_builder_begin_object (b);

    json_builder_set_member_name (b, "signature");
    json_builder_add_string_value (b, signature->str);

    json_builder_set_member_name (b, "origin");
    json_builder_add_string_value (b, std_string_get_data (&location));

    json_builder_set_member_name (b, "className");
    json_builder_add_string_value (b, class_name);

    json_builder_set_member_name (b, "methodName");
    json_builder_add_string_value (b, method_name);

    json_builder_set_member_name (b, "methodFlags");
    json_builder_add_int_value (b, translated_method->access_flags);

    json_builder_set_member_name (b, "fileName");
    json_builder_add_string_value (b, source_file);

    json_builder_set_member_name (b, "lineNumber");
    json_builder_add_int_value (b, line_number);

    json_builder_end_object (b);

    std_string_destroy (&location);
    g_string_free (signature, TRUE);
  }

  json_builder_end_array (b);

  root = json_builder_get_root (b);
  backtrace->frames_json = json_to_string (root, FALSE);
  json_node_unref (root);

  return backtrace->frames_json;
}

static void
append_jni_type_name (GString * s,
                      const gchar * name,
                      gsize length)
{
  gchar shorty = '\\0';
  gsize i;

  switch (name[0])
  {
    case 'b':
      if (strncmp (name, "boolean", length) == 0)
        shorty = 'Z';
      else if (strncmp (name, "byte", length) == 0)
        shorty = 'B';
      break;
    case 'c':
      if (strncmp (name, "char", length) == 0)
        shorty = 'C';
      break;
    case 'd':
      if (strncmp (name, "double", length) == 0)
        shorty = 'D';
      break;
    case 'f':
      if (strncmp (name, "float", length) == 0)
        shorty = 'F';
      break;
    case 'i':
      if (strncmp (name, "int", length) == 0)
        shorty = 'I';
      break;
    case 'l':
      if (strncmp (name, "long", length) == 0)
        shorty = 'J';
      break;
    case 's':
      if (strncmp (name, "short", length) == 0)
        shorty = 'S';
      break;
    case 'v':
      if (strncmp (name, "void", length) == 0)
        shorty = 'V';
      break;
  }

  if (shorty != '\\0')
  {
    g_string_append_c (s, shorty);

    return;
  }

  if (length > 2 && name[length - 2] == '[' && name[length - 1] == ']')
  {
    g_string_append_c (s, '[');
    append_jni_type_name (s, name, length - 2);

    return;
  }

  g_string_append_c (s, 'L');

  for (i = 0; i != length; i++)
  {
    gchar ch = name[i];
    if (ch != '.')
      g_string_append_c (s, ch);
    else
      g_string_append_c (s, '/');
  }

  g_string_append_c (s, ';');
}

static void
std_string_destroy (StdString * str)
{
  bool is_large = (str->flags & 1) != 0;
  if (is_large)
    cxx_delete (str->large.data);
}

static gchar *
std_string_get_data (StdString * str)
{
  bool is_large = (str->flags & 1) != 0;
  return is_large ? str->large.data : str->tiny.data;
}
`,{current_backtrace:Memory.alloc(Process.pointerSize),perform_art_thread_state_transition:n,art_make_context:r["art::Thread::GetLongJumpContext"]??r["art::Context::Create"],art_stack_visitor_init:r["art::StackVisitor::StackVisitor"],art_stack_visitor_walk_stack:r["art::StackVisitor::WalkStack"],art_stack_visitor_get_method:r["art::StackVisitor::GetMethod"],art_stack_visitor_describe_location:r["art::StackVisitor::DescribeLocation"],translate_method:_.replacedMethods.translate,translate_location:r["art::Monitor::TranslateLocation"],get_class_location:r["art::mirror::Class::GetLocation"],cxx_delete:r.$delete,strtoul:Process.getModuleByName("libc.so").getExportByName("strtoul")}),a=new NativeFunction(i._create,"pointer",["pointer","uint"],y),o=new NativeFunction(i._destroy,"void",["pointer"],y),s={exceptions:"propagate",scheduling:"exclusive"},l=new NativeFunction(i._get_id,"pointer",["pointer"],s),d=new NativeFunction(i._get_frames,"pointer",["pointer"],s),c=So(e,t,i._on_thread_state_transition_complete);i._performData=c,n.writePointer(c),i.backtrace=(e,t)=>{let r=a(e,t),n=new $a(r);return Script.bindWeak(n,u.bind(null,r)),n};function u(e){o(e)}return i.getId=e=>l(e).readUtf8String(),i.getFrames=e=>JSON.parse(d(e).readUtf8String()),i}var $a=class{constructor(e){this.handle=e}get id(){return fi.getId(this.handle)}get frames(){return fi.getFrames(this.handle)}};function Va(){li.forEach(e=>{e.vtablePtr.writePointer(e.vtable),e.vtableCountPtr.writeS32(e.vtableCount)}),li.clear();for(let e of di.splice(0))e.deactivate();for(let e of si.splice(0))e.revert()}function Ja(e){return Ga(e,"art::jni::JniIdManager::DecodeMethodId")}function Ba(e){return Ga(e,"art::jni::JniIdManager::DecodeFieldId")}function Ga(r,n){let i=S(),e=Hn(i).offset,a=e.jniIdManager,o=e.jniIdsIndirection;if(a!==null&&o!==null){let t=i.artRuntime;if(t.add(o).readInt()!==pn){let e=t.add(a).readPointer();return i[n](e,r)}}return r}var Za={ia32:Ha,x64:qa,arm:Wa,arm64:Ka};function Ha(a,o,s,e,t){let l=Wn(t).offset,d=E(t).offset,c;return Memory.patchCode(a,128,e=>{let t=new X86Writer(e,{pc:a}),r=new X86Relocator(o,t),n=[15,174,4,36],i=[15,174,12,36];t.putPushax(),t.putMovRegReg("ebp","esp"),t.putAndRegU32("esp",4294967280),t.putSubRegImm("esp",512),t.putBytes(n),t.putMovRegFsU32Ptr("ebx",l.self),t.putCallAddressWithAlignedArguments(_.replacedMethods.findReplacementFromQuickCode,["eax","ebx"]),t.putTestRegReg("eax","eax"),t.putJccShortLabel("je","restore_registers","no-hint"),t.putMovRegOffsetPtrReg("ebp",7*4,"eax"),t.putLabel("restore_registers"),t.putBytes(i),t.putMovRegReg("esp","ebp"),t.putPopax(),t.putJccShortLabel("jne","invoke_replacement","no-hint");do{c=r.readOne()}while(c<s&&!r.eoi);r.writeAll(),r.eoi||t.putJmpAddress(o.add(c)),t.putLabel("invoke_replacement"),t.putJmpRegOffsetPtr("eax",d.quickCode),t.flush()}),c}function qa(a,o,s,e,t){let l=Wn(t).offset,d=E(t).offset,c;return Memory.patchCode(a,256,e=>{let t=new X86Writer(e,{pc:a}),r=new X86Relocator(o,t),n=[15,174,4,36],i=[15,174,12,36];t.putPushax(),t.putMovRegReg("rbp","rsp"),t.putAndRegU32("rsp",4294967280),t.putSubRegImm("rsp",512),t.putBytes(n),t.putMovRegGsU32Ptr("rbx",l.self),t.putCallAddressWithAlignedArguments(_.replacedMethods.findReplacementFromQuickCode,["rdi","rbx"]),t.putTestRegReg("rax","rax"),t.putJccShortLabel("je","restore_registers","no-hint"),t.putMovRegOffsetPtrReg("rbp",8*8,"rax"),t.putLabel("restore_registers"),t.putBytes(i),t.putMovRegReg("rsp","rbp"),t.putPopax(),t.putJccShortLabel("jne","invoke_replacement","no-hint");do{c=r.readOne()}while(c<s&&!r.eoi);r.writeAll(),r.eoi||t.putJmpAddress(o.add(c)),t.putLabel("invoke_replacement"),t.putJmpRegOffsetPtr("rdi",d.quickCode),t.flush()}),c}function Wa(a,o,s,e,t){let l=E(t).offset,d=o.and(mn),c;return Memory.patchCode(a,128,e=>{let t=new ThumbWriter(e,{pc:a}),r=new ThumbRelocator(d,t),n=[45,237,16,10],i=[189,236,16,10];t.putPushRegs(["r1","r2","r3","r5","r6","r7","r8","r10","r11","lr"]),t.putBytes(n),t.putSubRegRegImm("sp","sp",8),t.putStrRegRegOffset("r0","sp",0),t.putCallAddressWithArguments(_.replacedMethods.findReplacementFromQuickCode,["r0","r9"]),t.putCmpRegImm("r0",0),t.putBCondLabel("eq","restore_registers"),t.putStrRegRegOffset("r0","sp",0),t.putLabel("restore_registers"),t.putLdrRegRegOffset("r0","sp",0),t.putAddRegRegImm("sp","sp",8),t.putBytes(i),t.putPopRegs(["lr","r11","r10","r8","r7","r6","r5","r3","r2","r1"]),t.putBCondLabel("ne","invoke_replacement");do{c=r.readOne()}while(c<s&&!r.eoi);r.writeAll(),r.eoi||t.putLdrRegAddress("pc",o.add(c)),t.putLabel("invoke_replacement"),t.putLdrRegRegOffset("pc","r0",l.quickCode),t.flush()}),c}function Ka(n,i,a,{availableScratchRegs:o},e){let s=E(e).offset,l;return Memory.patchCode(n,256,e=>{let t=new Arm64Writer(e,{pc:n}),r=new Arm64Relocator(i,t);t.putPushRegReg("d0","d1"),t.putPushRegReg("d2","d3"),t.putPushRegReg("d4","d5"),t.putPushRegReg("d6","d7"),t.putPushRegReg("x1","x2"),t.putPushRegReg("x3","x4"),t.putPushRegReg("x5","x6"),t.putPushRegReg("x7","x20"),t.putPushRegReg("x21","x22"),t.putPushRegReg("x23","x24"),t.putPushRegReg("x25","x26"),t.putPushRegReg("x27","x28"),t.putPushRegReg("x29","lr"),t.putSubRegRegImm("sp","sp",16),t.putStrRegRegOffset("x0","sp",0),t.putCallAddressWithArguments(_.replacedMethods.findReplacementFromQuickCode,["x0","x19"]),t.putCmpRegReg("x0","xzr"),t.putBCondLabel("eq","restore_registers"),t.putStrRegRegOffset("x0","sp",0),t.putLabel("restore_registers"),t.putLdrRegRegOffset("x0","sp",0),t.putAddRegRegImm("sp","sp",16),t.putPopRegReg("x29","lr"),t.putPopRegReg("x27","x28"),t.putPopRegReg("x25","x26"),t.putPopRegReg("x23","x24"),t.putPopRegReg("x21","x22"),t.putPopRegReg("x7","x20"),t.putPopRegReg("x5","x6"),t.putPopRegReg("x3","x4"),t.putPopRegReg("x1","x2"),t.putPopRegReg("d6","d7"),t.putPopRegReg("d4","d5"),t.putPopRegReg("d2","d3"),t.putPopRegReg("d0","d1"),t.putBCondLabel("ne","invoke_replacement");do{l=r.readOne()}while(l<a&&!r.eoi);if(r.writeAll(),!r.eoi){let e=Array.from(o)[0];t.putLdrRegAddress(e,i.add(l)),t.putBrReg(e)}t.putLabel("invoke_replacement"),t.putLdrRegRegOffset("x16","x0",s.quickCode),t.putBrReg("x16"),t.flush()}),l}var Qa={ia32:Xa,x64:Xa,arm:Ya,arm64:eo};function Xa(r,n,e){Memory.patchCode(r,16,e=>{let t=new X86Writer(e,{pc:r});t.putJmpAddress(n),t.flush()})}function Ya(e,r,t){let n=e.and(mn);Memory.patchCode(n,16,e=>{let t=new ThumbWriter(e,{pc:n});t.putLdrRegAddress("pc",r.or(1)),t.flush()})}function eo(r,n,i){Memory.patchCode(r,16,e=>{let t=new Arm64Writer(e,{pc:r});i===16?t.putLdrRegAddress("x16",n):t.putAdrpRegAddress("x16",n),t.putBrReg("x16"),t.flush()})}var to={ia32:5,x64:16,arm:8,arm64:16},ro=class{constructor(e){this.quickCode=e,this.quickCodeAddress=Process.arch==="arm"?e.and(mn):e,this.redirectSize=0,this.trampoline=null,this.overwrittenPrologue=null,this.overwrittenPrologueLength=0}_canRelocateCode(e,t){let r=va[Process.arch],n=ba[Process.arch],{quickCodeAddress:i}=this,a=new r(i),o=new n(i,a),s;if(Process.arch==="arm64"){let i=new Set(["x16","x17"]);do{let e=o.readOne(),r=new Set(i),{read:t,written:n}=o.input.regsAccessed;for(let e of[t,n])for(let t of e){let e;t.startsWith("w")?e="x"+t.substring(1):e=t,r.delete(e)}if(r.size===0)break;s=e,i=r}while(s<e&&!o.eoi);t.availableScratchRegs=i}else do{s=o.readOne()}while(s<e&&!o.eoi);return s>=e}_allocateTrampoline(){gi===null&&(gi=He(v===4?128:256));let e=to[Process.arch],t,r,n=1,i={};if(v===4||this._canRelocateCode(e,i))t=e,r={};else{let e;Process.arch==="x64"?(t=5,e=gn):Process.arch==="arm64"&&(t=8,e=bn,n=4096),r={near:this.quickCodeAddress,maxDistance:e}}return this.redirectSize=t,this.trampoline=gi.allocateSlice(r,n),i}_destroyTrampoline(){gi.freeSlice(this.trampoline)}activate(e){let t=this._allocateTrampoline(),{trampoline:r,quickCode:n,redirectSize:i}=this,a=Za[Process.arch],o=a(r,n,i,t,e);this.overwrittenPrologueLength=o,this.overwrittenPrologue=Memory.dup(this.quickCodeAddress,o);let s=Qa[Process.arch];s(n,r,i)}deactivate(){let{quickCodeAddress:n,overwrittenPrologueLength:i}=this,a=va[Process.arch];Memory.patchCode(n,i,e=>{let t=new a(e,{pc:n}),{overwrittenPrologue:r}=this;t.putBytes(r.readByteArray(i)),t.flush()}),this._destroyTrampoline()}};function no(e){let t=S(),{module:r,artClassLinker:n}=t;return e.equals(n.quickGenericJniTrampoline)||e.equals(n.quickToInterpreterBridgeTrampoline)||e.equals(n.quickResolutionTrampoline)||e.equals(n.quickImtConflictTrampoline)||e.compare(r.base)>=0&&e.compare(r.base.add(r.size))<0}var io=class{constructor(e){let t=Ja(e);this.methodId=t,this.originalMethod=null,this.hookedMethodId=t,this.replacementMethodId=null,this.interceptor=null}replace(e,t,r,n,i){let{kAccCompileDontBother:a,artNterpEntryPoint:o}=i;this.originalMethod=oo(this.methodId,n);let s=this.originalMethod.accessFlags;if((s&hn)!==0&&ao()){let e=this.originalMethod.jniCode;this.hookedMethodId=e.add(2*v).readPointer(),this.originalMethod=oo(this.hookedMethodId,n)}let{hookedMethodId:l}=this,d=uo(l,n);this.replacementMethodId=d,so(d,{jniCode:e,accessFlags:(s&~(an|nn|dn)|rn|a)>>>0,quickCode:i.artClassLinker.quickGenericJniTrampoline,interpreterCode:i.artInterpreterToCompiledCodeBridge},n);let c=on|ln|dn;(s&rn)===0&&(c|=sn),so(l,{accessFlags:(s&~c|a)>>>0},n);let u=this.originalMethod.quickCode;if(o!==null&&u.equals(o)&&so(l,{quickCode:i.artQuickToInterpreterBridge},n),!no(u)){let e=new ro(u);e.activate(n),this.interceptor=e}_.replacedMethods.set(l,d),wa(l,n)}revert(e){let{hookedMethodId:t,interceptor:r}=this;so(t,this.originalMethod,e),_.replacedMethods.delete(t),r!==null&&(r.deactivate(),this.interceptor=null)}resolveTarget(e,t,r,n){return this.hookedMethodId}};function ao(){return m()<28}function oo(a,e){let o=E(e).offset;return["jniCode","accessFlags","quickCode","interpreterCode"].reduce((e,t)=>{let r=o[t];if(r===void 0)return e;let n=a.add(r),i=t==="accessFlags"?Wr:Kr;return e[t]=i.call(n),e},{})}function so(n,i,e){let a=E(e).offset;Object.keys(i).forEach(e=>{let t=a[e];if(t===void 0)return;let r=n.add(t);(e==="accessFlags"?Qr:Xr).call(r,i[e])})}var lo=class{constructor(e){this.methodId=e,this.originalMethod=null}replace(e,t,r,n,i){let{methodId:a}=this;this.originalMethod=Memory.dup(a,Ln);let o=r.reduce((e,t)=>e+t.size,0);t&&o++;let s=(a.add(kn).readU32()|rn)>>>0,l=o,d=0,c=o;a.add(kn).writeU32(s),a.add(jn).writeU16(l),a.add(Mn).writeU16(d),a.add(In).writeU16(c),a.add(Rn).writeU32(co(a)),i.dvmUseJNIBridge(a,e)}revert(e){Memory.copy(this.methodId,this.originalMethod,Ln)}resolveTarget(t,e,r,n){let i=r.handle.add(wn).readPointer(),a;if(e)a=n.dvmDecodeIndirectRef(i,t.$h);else{let e=t.$borrowClassHandle(r);a=n.dvmDecodeIndirectRef(i,e.value),e.unref(r)}let o;e?o=a.add(Nn).readPointer():o=a;let s=o.toString(16),l=li.get(s);if(l===void 0){let e=o.add(Sn),t=o.add(En),r=e.readPointer(),n=t.readS32(),i=n*v,a=Memory.alloc(2*i);Memory.copy(a,r,i),e.writePointer(a),l={classObject:o,vtablePtr:e,vtableCountPtr:t,vtable:r,vtableCount:n,shadowVtable:a,shadowVtableCount:n,targetMethods:new Map},li.set(s,l)}let d=this.methodId.toString(16),c=l.targetMethods.get(d);if(c===void 0){c=Memory.dup(this.originalMethod,Ln);let e=l.shadowVtableCount++;l.shadowVtable.add(e*v).writePointer(c),c.add(Cn).writeU16(e),l.vtableCountPtr.writeS32(l.shadowVtableCount),l.targetMethods.set(d,c)}return c}};function co(e){if(Process.arch!=="ia32")return $n;let r=e.add(Pn).readPointer().readCString();if(r===null||r.length===0||r.length>65535)return $n;let t;switch(r[0]){case"V":t=An;break;case"F":t=xn;break;case"D":t=Tn;break;case"J":t=Un;break;case"Z":case"B":t=zn;break;case"C":t=Dn;break;case"S":t=Fn;break;default:t=On;break}let n=0;for(let t=r.length-1;t>0;t--){let e=r[t];n+=e==="D"||e==="J"?2:1}return t<<Vn|n}function uo(t,e){let r=S();if(m()<23){let e=r["art::Thread::CurrentFromGdb"]();return r["art::mirror::Object::Clone"](t,e)}return Memory.dup(t,E(e).size)}function ho(e,t,r){_o(e,t,_n,r)}function po(e,t){_o(e,t,fn)}function fo(e,t){let r=S();if(m()<26)throw new Error("This API is only available on Android >= 8.0");w(e,t,e=>{r["art::Runtime::DeoptimizeBootImage"](r.artRuntime)})}function _o(e,t,r,n){let i=S();if(m()<24)throw new Error("This API is only available on Android >= 7.0");w(e,t,e=>{if(m()<30){if(!i.isJdwpStarted()){let e=go(i);_i.push(e)}i.isDebuggerActive()||i["art::Dbg::GoActive"]();let e=Memory.alloc(8+v);switch(e.writeU32(r),r){case fn:break;case _n:e.add(8).writePointer(n);break;default:throw new Error("Unsupported deoptimization kind")}i["art::Dbg::RequestDeoptimization"](e),i["art::Dbg::ManageDeoptimization"]()}else{let e=i.artInstrumentation;if(e===null)throw new Error("Unable to find Instrumentation class in ART; please file a bug");let t=i["art::Instrumentation::EnableDeoptimization"];switch(t!==void 0&&(e.add(qn().offset.deoptimizationEnabled).readU8()||t(e)),r){case fn:i["art::Instrumentation::DeoptimizeEverything"](e,Memory.allocUtf8String("frida"));break;case _n:i["art::Instrumentation::Deoptimize"](e,n);break;default:throw new Error("Unsupported deoptimization kind")}}})}var mo=class{constructor(){let e=Process.getModuleByName("libart.so"),t=e.getExportByName("_ZN3art4JDWP12JdwpAdbState6AcceptEv"),r=e.getExportByName("_ZN3art4JDWP12JdwpAdbState15ReceiveClientFdEv"),n=vo(),i=vo();this._controlFd=n[0],this._clientFd=i[0];let a=null;a=Interceptor.attach(t,function(e){let t=e[0];Memory.scanSync(t.add(8252),256,"00 ff ff ff ff 00")[0].address.add(1).writeS32(n[1]),a.detach()}),Interceptor.replace(r,new NativeCallback(function(e){return Interceptor.revert(r),i[1]},"int",["pointer"])),Interceptor.flush(),this._handshakeRequest=this._performHandshake()}async _performHandshake(){let e=new UnixInputStream(this._clientFd,{autoClose:!1}),t=new UnixOutputStream(this._clientFd,{autoClose:!1}),r=[74,68,87,80,45,72,97,110,100,115,104,97,107,101];try{await t.writeAll(r),await e.readAll(r.length)}catch{}}};function go(e){let t=new mo;e["art::Dbg::SetJdwpAllowed"](1);let r=bo();e["art::Dbg::ConfigureJdwp"](r);let n=e["art::InternalDebuggerControlCallback::StartDebugger"];return n!==void 0?n(NULL):e["art::Dbg::StartJdwp"](),t}function bo(){let e=m()<28?2:3,t=0,r=e,n=!0,i=!1,a=t,o=8+Jn+2,s=Memory.alloc(o);return s.writeU32(r).add(4).writeU8(n?1:0).add(1).writeU8(i?1:0).add(1).add(Jn).writeU16(a),s}function vo(){mi===null&&(mi=new NativeFunction(Process.getModuleByName("libc.so").getExportByName("socketpair"),"int",["int","int","int","pointer"]));let e=Memory.alloc(8);if(mi(Gn,Zn,0,e)===-1)throw new Error("Unable to create socketpair for JDWP");return[e.readS32(),e.add(4).readS32()]}function yo(e){let t=wi().offset,n=e.vm.add(t.globalsLock),i=e.vm.add(t.globals),a=e["art::IndirectReferenceTable::Add"],o=e["art::ReaderWriterMutex::ExclusiveLock"],s=e["art::ReaderWriterMutex::ExclusiveUnlock"],l=0;return function(e,t,r){o(n,t);try{return a(i,l,r)}finally{s(n,t)}}}function wo(e){let n=e["art::Thread::DecodeJObject"];if(n===void 0)throw new Error("art::Thread::DecodeJObject is not available; please file a bug");return function(e,t,r){return n(t,r)}}var Eo={ia32:No,x64:No,arm:Lo,arm64:ko};function So(e,t,r){let n=S(),i=t.handle.readPointer(),a,o=n.find("_ZN3art3JNIILb1EE14ExceptionClearEP7_JNIEnv");o!==null?a=o:a=i.add(vn).readPointer();let s,l=n.find("_ZN3art3JNIILb1EE10FatalErrorEP7_JNIEnvPKc");l!==null?s=l:s=i.add(yn).readPointer();let d=Eo[Process.arch];if(d===void 0)throw new Error("Not yet implemented for "+Process.arch);let c=null,u=Wn(e).offset,h=u.exception,p=new Set,f=u.isExceptionReportedToInstrumentation;f!==null&&p.add(f);let _=u.throwLocation;_!==null&&(p.add(_),p.add(_+v),p.add(_+2*v));let m=65536,g=Memory.alloc(m);return Memory.patchCode(g,m,e=>{c=d(e,g,a,s,h,p,r)}),c._code=g,c._callback=r,c}function No(e,t,r,l,s,d,c){let u={},h=new Set,p=[r];for(;p.length>0;){let n=p.shift();if(Object.values(u).some(({begin:e,end:t})=>n.compare(e)>=0&&n.compare(t)<0))continue;let i=n.toString(),a={begin:n},o=null,s=!1;do{if(n.equals(l)){s=!0;break}let e=Instruction.parse(n);o=e;let t=u[e.address.toString()];if(t!==void 0){delete u[t.begin.toString()],u[i]=t,t.begin=a.begin,a=null;break}let r=null;switch(e.mnemonic){case"jmp":r=ptr(e.operands[0].value),s=!0;break;case"je":case"jg":case"jle":case"jne":case"js":r=ptr(e.operands[0].value);break;case"ret":s=!0;break}r!==null&&(h.add(r.toString()),p.push(r),p.sort((e,t)=>e.compare(t))),n=e.next}while(!s);a!==null&&(a.end=o.address.add(o.size),u[i]=a)}let n=Object.keys(u).map(e=>u[e]);n.sort((e,t)=>e.begin.compare(t.begin));let i=u[r.toString()];n.splice(n.indexOf(i),1),n.unshift(i);let f=new X86Writer(e,{pc:t}),_=!1,m=null;return n.forEach(e=>{let n=e.end.sub(e.begin).toInt32(),a=new X86Relocator(e.begin,f),o;for(;(o=a.readOne())!==0;){let t=a.input,{mnemonic:e}=t,r=t.address.toString();h.has(r)&&f.putLabel(r);let i=!0;switch(e){case"jmp":f.putJmpNearLabel(N(t.operands[0])),i=!1;break;case"je":case"jg":case"jle":case"jne":case"js":f.putJccNearLabel(e,N(t.operands[0]),"no-hint"),i=!1;break;case"mov":{let[r,n]=t.operands;if(r.type==="mem"&&n.type==="imm"){let e=r.value,t=e.disp;if(t===s&&n.value.valueOf()===0){if(m=e.base,f.putPushfx(),f.putPushax(),f.putMovRegReg("xbp","xsp"),v===4)f.putAndRegU32("esp",4294967280);else{let e=m!=="rdi"?"rdi":"rsi";f.putMovRegU64(e,uint64("0xfffffffffffffff0")),f.putAndRegReg("rsp",e)}f.putCallAddressWithAlignedArguments(c,[m]),f.putMovRegReg("xsp","xbp"),f.putPopax(),f.putPopfx(),_=!0,i=!1}else d.has(t)&&e.base===m&&(i=!1)}break}case"call":{let e=t.operands[0];e.type==="mem"&&e.value.disp===vn&&(v===4?(f.putPopReg("eax"),f.putMovRegRegOffsetPtr("eax","eax",4),f.putPushReg("eax")):f.putMovRegRegOffsetPtr("rdi","rdi",8),f.putCallAddressWithArguments(c,[]),_=!0,i=!1);break}}if(i?a.writeAll():a.skipOne(),o===n)break}a.dispose()}),f.dispose(),_||Co(),new NativeFunction(t,"void",["pointer"],y)}function Lo(e,t,r,p,s,l,d){let f={},_=new Set,m=ptr(1).not(),g=[r];for(;g.length>0;){let o=g.shift();if(Object.values(f).some(({begin:e,end:t})=>o.compare(e)>=0&&o.compare(t)<0))continue;let e=o.and(m),s=e.toString(),l=o.and(1),d={begin:e},c=null,u=!1,h=0;do{if(o.equals(p)){u=!0;break}let e=Instruction.parse(o),{mnemonic:t}=e;c=e;let r=o.and(m).toString(),n=f[r];if(n!==void 0){delete f[n.begin.toString()],f[s]=n,n.begin=d.begin,d=null;break}let i=h===0,a=null;switch(t){case"b":a=ptr(e.operands[0].value),u=i;break;case"beq.w":case"beq":case"bne":case"bne.w":case"bgt":a=ptr(e.operands[0].value);break;case"cbz":case"cbnz":a=ptr(e.operands[1].value);break;case"pop.w":i&&(u=e.operands.filter(e=>e.value==="pc").length===1);break}switch(t){case"it":h=1;break;case"itt":h=2;break;case"ittt":h=3;break;case"itttt":h=4;break;default:h>0&&h--;break}a!==null&&(_.add(a.toString()),g.push(a.or(l)),g.sort((e,t)=>e.compare(t))),o=e.next}while(!u);d!==null&&(d.end=c.address.add(c.size),f[s]=d)}let n=Object.keys(f).map(e=>f[e]);n.sort((e,t)=>e.begin.compare(t.begin));let i=f[r.and(m).toString()];n.splice(n.indexOf(i),1),n.unshift(i);let c=new ThumbWriter(e,{pc:t}),u=!1,h=null,b=null;return n.forEach(e=>{let r=new ThumbRelocator(e.begin,c),a=e.begin,t=e.end,o=0;do{if(r.readOne()===0)throw new Error("Unexpected end of block");let n=r.input;a=n.address,o=n.size;let{mnemonic:e}=n,t=a.toString();_.has(t)&&c.putLabel(t);let i=!0;switch(e){case"b":c.putBLabel(N(n.operands[0])),i=!1;break;case"beq.w":c.putBCondLabelWide("eq",N(n.operands[0])),i=!1;break;case"bne.w":c.putBCondLabelWide("ne",N(n.operands[0])),i=!1;break;case"beq":case"bne":case"bgt":c.putBCondLabelWide(e.substr(1),N(n.operands[0])),i=!1;break;case"cbz":{let e=n.operands;c.putCbzRegLabel(e[0].value,N(e[1])),i=!1;break}case"cbnz":{let e=n.operands;c.putCbnzRegLabel(e[0].value,N(e[1])),i=!1;break}case"str":case"str.w":{let r=n.operands[1].value,e=r.disp;if(e===s){h=r.base;let e=h!=="r4"?"r4":"r5",t=["r0","r1","r2","r3",e,"r9","r12","lr"];c.putPushRegs(t),c.putMrsRegReg(e,"apsr-nzcvq"),c.putCallAddressWithArguments(d,[h]),c.putMsrRegReg("apsr-nzcvq",e),c.putPopRegs(t),u=!0,i=!1}else l.has(e)&&r.base===h&&(i=!1);break}case"ldr":{let[t,r]=n.operands;if(r.type==="mem"){let e=r.value;e.base[0]==="r"&&e.disp===vn&&(b=t.value)}break}case"blx":n.operands[0].value===b&&(c.putLdrRegRegOffset("r0","r0",4),c.putCallAddressWithArguments(d,["r0"]),u=!0,b=null,i=!1);break}i?r.writeAll():r.skipOne()}while(!a.add(o).equals(t));r.dispose()}),c.dispose(),u||Co(),new NativeFunction(t.or(1),"void",["pointer"],y)}function ko(e,t,r,l,s,d,c){let u={},h=new Set,p=[r];for(;p.length>0;){let n=p.shift();if(Object.values(u).some(({begin:e,end:t})=>n.compare(e)>=0&&n.compare(t)<0))continue;let i=n.toString(),a={begin:n},o=null,s=!1;do{if(n.equals(l)){s=!0;break}let e;try{e=Instruction.parse(n)}catch(e){if(n.readU32()===0){s=!0;break}else throw e}o=e;let t=u[e.address.toString()];if(t!==void 0){delete u[t.begin.toString()],u[i]=t,t.begin=a.begin,a=null;break}let r=null;switch(e.mnemonic){case"b":r=ptr(e.operands[0].value),s=!0;break;case"b.eq":case"b.ne":case"b.le":case"b.gt":r=ptr(e.operands[0].value);break;case"cbz":case"cbnz":r=ptr(e.operands[1].value);break;case"tbz":case"tbnz":r=ptr(e.operands[2].value);break;case"ret":s=!0;break}r!==null&&(h.add(r.toString()),p.push(r),p.sort((e,t)=>e.compare(t))),n=e.next}while(!s);a!==null&&(a.end=o.address.add(o.size),u[i]=a)}let n=Object.keys(u).map(e=>u[e]);n.sort((e,t)=>e.begin.compare(t.begin));let i=u[r.toString()];n.splice(n.indexOf(i),1),n.unshift(i);let f=new Arm64Writer(e,{pc:t});f.putBLabel("performTransition");let _=t.add(f.offset);f.putPushAllXRegisters(),f.putCallAddressWithArguments(c,["x0"]),f.putPopAllXRegisters(),f.putRet(),f.putLabel("performTransition");let m=!1,g=null,b=null;return n.forEach(e=>{let r=e.end.sub(e.begin).toInt32(),n=new Arm64Relocator(e.begin,f),o;for(;(o=n.readOne())!==0;){let i=n.input,{mnemonic:e}=i,t=i.address.toString();h.has(t)&&f.putLabel(t);let a=!0;switch(e){case"b":f.putBLabel(N(i.operands[0])),a=!1;break;case"b.eq":case"b.ne":case"b.le":case"b.gt":f.putBCondLabel(e.substr(2),N(i.operands[0])),a=!1;break;case"cbz":{let e=i.operands;f.putCbzRegLabel(e[0].value,N(e[1])),a=!1;break}case"cbnz":{let e=i.operands;f.putCbnzRegLabel(e[0].value,N(e[1])),a=!1;break}case"tbz":{let e=i.operands;f.putTbzRegImmLabel(e[0].value,e[1].value.valueOf(),N(e[2])),a=!1;break}case"tbnz":{let e=i.operands;f.putTbnzRegImmLabel(e[0].value,e[1].value.valueOf(),N(e[2])),a=!1;break}case"str":{let e=i.operands,t=e[0].value,r=e[1].value,n=r.disp;t==="xzr"&&n===s?(g=r.base,f.putPushRegReg("x0","lr"),f.putMovRegReg("x0",g),f.putBlImm(_),f.putPopRegReg("x0","lr"),m=!0,a=!1):d.has(n)&&r.base===g&&(a=!1);break}case"ldr":{let e=i.operands,t=e[1].value;t.base[0]==="x"&&t.disp===vn&&(b=e[0].value);break}case"blr":i.operands[0].value===b&&(f.putLdrRegRegOffset("x0","x0",8),f.putCallAddressWithArguments(c,["x0"]),m=!0,b=null,a=!1);break}if(a?n.writeAll():n.skipOne(),o===r)break}n.dispose()}),f.dispose(),m||Co(),new NativeFunction(t,"void",["pointer"],y)}function Co(){throw new Error("Unable to parse ART internals; please file a bug")}function jo(e){let t=e["art::ArtMethod::PrettyMethod"];t!==void 0&&(Interceptor.attach(t.impl,_.hooks.ArtMethod.prettyMethod),Interceptor.flush())}function N(e){return ptr(e.value).toString()}function Mo(e,t){return new NativeFunction(e,"pointer",t,y)}function Io(e,t){let r=new NativeFunction(e,"void",["pointer"].concat(t),y);return function(){let e=Memory.alloc(v);return r(e,...arguments),e.readPointer()}}function Po(i,a){let{arch:n}=Process;switch(n){case"ia32":case"arm64":{let e;n==="ia32"?e=ya(64,r=>{let e=1+a.length,n=e*4;r.putSubRegImm("esp",n);for(let t=0;t!==e;t++){let e=t*4;r.putMovRegRegOffsetPtr("eax","esp",n+4+e),r.putMovRegOffsetPtrReg("esp",e,"eax")}r.putCallAddress(i),r.putAddRegImm("esp",n-4),r.putRet()}):e=ya(32,r=>{r.putMovRegReg("x8","x0"),a.forEach((e,t)=>{r.putMovRegReg("x"+t,"x"+(t+1))}),r.putLdrRegAddress("x7",i),r.putBrReg("x7")});let t=new NativeFunction(e,"void",["pointer"].concat(a),y),r=function(...e){t(...e)};return r.handle=e,r.impl=i,r}default:{let e=new NativeFunction(i,"void",["pointer"].concat(a),y);return e.impl=i,e}}}var Ro=class{constructor(){this.handle=Memory.alloc(Jn)}dispose(){let[e,t]=this._getData();t||S().$delete(e)}disposeToString(){let e=this.toString();return this.dispose(),e}toString(){let[e]=this._getData();return e.readUtf8String()}_getData(){let e=this.handle,t=(e.readU8()&1)===0;return[t?e.add(1):e.add(2*v).readPointer(),t]}},Ao=class{$delete(){this.dispose(),S().$delete(this)}constructor(e,t){this.handle=e,this._begin=e,this._end=e.add(v),this._storage=e.add(2*v),this._elementSize=t}init(){this.begin=NULL,this.end=NULL,this.storage=NULL}dispose(){S().$delete(this.begin)}get begin(){return this._begin.readPointer()}set begin(e){this._begin.writePointer(e)}get end(){return this._end.readPointer()}set end(e){this._end.writePointer(e)}get storage(){return this._storage.readPointer()}set storage(e){this._storage.writePointer(e)}get size(){return this.end.sub(this.begin).toInt32()/this._elementSize}},xo=class A extends Ao{static $new(){let e=new A(S().$new(Bn));return e.init(),e}constructor(e){super(e,v)}get handles(){let e=[],t=this.begin,r=this.end;for(;!t.equals(r);)e.push(t.readPointer()),t=t.add(v);return e}},To=0,Uo=v,Oo=Uo+4,Fo=-1,Do=class A{$delete(){this.dispose(),S().$delete(this)}constructor(e){this.handle=e,this._link=e.add(To),this._numberOfReferences=e.add(Uo)}init(e,t){this.link=e,this.numberOfReferences=t}dispose(){}get link(){return new A(this._link.readPointer())}set link(e){this._link.writePointer(e)}get numberOfReferences(){return this._numberOfReferences.readS32()}set numberOfReferences(e){this._numberOfReferences.writeS32(e)}},zo=qo(Oo),$o=zo+v,Vo=$o+v,Jo=class A extends Do{static $new(e,t){let r=new A(S().$new(Vo));return r.init(e,t),r}constructor(e){super(e),this._self=e.add(zo),this._currentScope=e.add($o);let t=(64-v-4-4)/4;this._scopeLayout=Bo.layoutForCapacity(t),this._topHandleScopePtr=null}init(e,t){let r=e.add(Wn(t).offset.topHandleScope);this._topHandleScopePtr=r,super.init(r.readPointer(),Fo),this.self=e,this.currentScope=Bo.$new(this._scopeLayout),r.writePointer(this)}dispose(){this._topHandleScopePtr.writePointer(this.link);let t;for(;(t=this.currentScope)!==null;){let e=t.link;t.$delete(),this.currentScope=e}}get self(){return this._self.readPointer()}set self(e){this._self.writePointer(e)}get currentScope(){let e=this._currentScope.readPointer();return e.isNull()?null:new Bo(e,this._scopeLayout)}set currentScope(e){this._currentScope.writePointer(e)}newHandle(e){return this.currentScope.newHandle(e)}},Bo=class A extends Do{static $new(e){let t=new A(S().$new(e.size),e);return t.init(),t}constructor(e,t){super(e);let{offset:r}=t;this._refsStorage=e.add(r.refsStorage),this._pos=e.add(r.pos),this._layout=t}init(){super.init(NULL,this._layout.numberOfReferences),this.pos=0}get pos(){return this._pos.readU32()}set pos(e){this._pos.writeU32(e)}newHandle(e){let t=this.pos,r=this._refsStorage.add(t*4);return r.writeS32(e.toInt32()),this.pos=t+1,r}static layoutForCapacity(e){let t=Oo,r=t+e*4;return{size:r+4,numberOfReferences:e,offset:{refsStorage:t,pos:r}}}},Go={arm:function(e,t){let r=Process.pageSize,n=Memory.alloc(r);Memory.protect(n,r,"rwx");let i=new NativeCallback(t,"void",["pointer"]);n._onMatchCallback=i;let a=[26625,18947,17041,53505,19202,18200,18288,48896],o=a.length*2,s=o+4,l=s+4;return Memory.patchCode(n,l,function(r){a.forEach((e,t)=>{r.add(t*2).writeU16(e)}),r.add(o).writeS32(e),r.add(s).writePointer(i)}),n.or(1)},arm64:function(e,t){let r=Process.pageSize,n=Memory.alloc(r);Memory.protect(n,r,"rwx");let i=new NativeCallback(t,"void",["pointer"]);n._onMatchCallback=i;let a=[3107979265,402653378,1795293247,1409286241,1476395139,3592355936,3596551104],o=a.length*4,s=o+4,l=s+8;return Memory.patchCode(n,l,function(r){a.forEach((e,t)=>{r.add(t*4).writeU32(e)}),r.add(o).writeS32(e),r.add(s).writePointer(i)}),n}};function Zo(e,t){return(Go[Process.arch]||Ho)(e,t)}function Ho(t,r){return new NativeCallback(e=>{e.readS32()===t&&r(e)},"void",["pointer","pointer"])}function qo(e){let t=e%v;return t!==0?e+v-t:e}var Wo=4,{pointerSize:C}=Process,Ko=256,Qo=65536,Xo=131072,Yo=33554432,es=67108864,ts=134217728,f={exceptions:"propagate"},rs=e(Cs),ns=e(Ms),is=e(ys),as=null,os=!1,ss=new Map,ls=new Map;function j(){return as===null&&(as=ds()),as}function ds(){let e=Process.enumerateModules().filter(e=>/jvm.(dll|dylib|so)$/.test(e.name));if(e.length===0)return null;let t=e[0],s={flavor:"jvm"},r=Process.platform==="windows"?[{module:t,functions:{JNI_GetCreatedJavaVMs:["JNI_GetCreatedJavaVMs","int",["pointer","int","pointer"]],JVM_Sleep:["JVM_Sleep","void",["pointer","pointer","long"]],"VMThread::execute":["VMThread::execute","void",["pointer"]],"Method::size":["Method::size","int",["int"]],"Method::set_native_function":["Method::set_native_function","void",["pointer","pointer","int"]],"Method::clear_native_function":["Method::clear_native_function","void",["pointer"]],"Method::jmethod_id":["Method::jmethod_id","pointer",["pointer"]],"ClassLoaderDataGraph::classes_do":["ClassLoaderDataGraph::classes_do","void",["pointer"]],"NMethodSweeper::sweep_code_cache":["NMethodSweeper::sweep_code_cache","void",[]],"OopMapCache::flush_obsolete_entries":["OopMapCache::flush_obsolete_entries","void",["pointer"]]},variables:{"VM_RedefineClasses::`vftable'":function(e){this.vtableRedefineClasses=e},"VM_RedefineClasses::doit":function(e){this.redefineClassesDoIt=e},"VM_RedefineClasses::doit_prologue":function(e){this.redefineClassesDoItPrologue=e},"VM_RedefineClasses::doit_epilogue":function(e){this.redefineClassesDoItEpilogue=e},"VM_RedefineClasses::allow_nested_vm_operations":function(e){this.redefineClassesAllow=e},"NMethodSweeper::_traversals":function(e){this.traversals=e},"NMethodSweeper::_should_sweep":function(e){this.shouldSweep=e}},optionals:[]}]:[{module:t,functions:{JNI_GetCreatedJavaVMs:["JNI_GetCreatedJavaVMs","int",["pointer","int","pointer"]],_ZN6Method4sizeEb:["Method::size","int",["int"]],_ZN6Method19set_native_functionEPhb:["Method::set_native_function","void",["pointer","pointer","int"]],_ZN6Method21clear_native_functionEv:["Method::clear_native_function","void",["pointer"]],_ZN6Method24restore_unshareable_infoEP10JavaThread:["Method::restore_unshareable_info","void",["pointer","pointer"]],_ZN6Method24restore_unshareable_infoEP6Thread:["Method::restore_unshareable_info","void",["pointer","pointer"]],_ZN6Method11link_methodERK12methodHandleP10JavaThread:["Method::link_method","void",["pointer","pointer","pointer"]],_ZN6Method10jmethod_idEv:["Method::jmethod_id","pointer",["pointer"]],_ZN6Method10clear_codeEv:function(e){let t=new NativeFunction(e,"void",["pointer"],f);this["Method::clear_code"]=function(e){t(e)}},_ZN6Method10clear_codeEb:function(e){let t=new NativeFunction(e,"void",["pointer","int"],f),r=0;this["Method::clear_code"]=function(e){t(e,r)}},_ZN18VM_RedefineClasses19mark_dependent_codeEP13InstanceKlass:["VM_RedefineClasses::mark_dependent_code","void",["pointer","pointer"]],_ZN18VM_RedefineClasses20flush_dependent_codeEv:["VM_RedefineClasses::flush_dependent_code","void",[]],_ZN18VM_RedefineClasses20flush_dependent_codeEP13InstanceKlassP6Thread:["VM_RedefineClasses::flush_dependent_code","void",["pointer","pointer","pointer"]],_ZN18VM_RedefineClasses20flush_dependent_codeE19instanceKlassHandleP6Thread:["VM_RedefineClasses::flush_dependent_code","void",["pointer","pointer","pointer"]],_ZN19ResolvedMethodTable21adjust_method_entriesEPb:["ResolvedMethodTable::adjust_method_entries","void",["pointer"]],_ZN15MemberNameTable21adjust_method_entriesEP13InstanceKlassPb:["MemberNameTable::adjust_method_entries","void",["pointer","pointer","pointer"]],_ZN17ConstantPoolCache21adjust_method_entriesEPb:function(e){let n=new NativeFunction(e,"void",["pointer","pointer"],f);this["ConstantPoolCache::adjust_method_entries"]=function(e,t,r){n(e,r)}},_ZN17ConstantPoolCache21adjust_method_entriesEP13InstanceKlassPb:function(e){let n=new NativeFunction(e,"void",["pointer","pointer","pointer"],f);this["ConstantPoolCache::adjust_method_entries"]=function(e,t,r){n(e,t,r)}},_ZN20ClassLoaderDataGraph10classes_doEP12KlassClosure:["ClassLoaderDataGraph::classes_do","void",["pointer"]],_ZN20ClassLoaderDataGraph22clean_deallocate_listsEb:["ClassLoaderDataGraph::clean_deallocate_lists","void",["int"]],_ZN10JavaThread27thread_from_jni_environmentEP7JNIEnv_:["JavaThread::thread_from_jni_environment","pointer",["pointer"]],_ZN8VMThread7executeEP12VM_Operation:["VMThread::execute","void",["pointer"]],_ZN11OopMapCache22flush_obsolete_entriesEv:["OopMapCache::flush_obsolete_entries","void",["pointer"]],_ZN14NMethodSweeper11force_sweepEv:["NMethodSweeper::force_sweep","void",[]],_ZN14NMethodSweeper16sweep_code_cacheEv:["NMethodSweeper::sweep_code_cache","void",[]],_ZN14NMethodSweeper17sweep_in_progressEv:["NMethodSweeper::sweep_in_progress","bool",[]],JVM_Sleep:["JVM_Sleep","void",["pointer","pointer","long"]]},variables:{_ZN18VM_RedefineClasses14_the_class_oopE:function(e){this.redefineClass=e},_ZN18VM_RedefineClasses10_the_classE:function(e){this.redefineClass=e},_ZN18VM_RedefineClasses25AdjustCpoolCacheAndVtable8do_klassEP5Klass:function(e){this.doKlass=e},_ZN18VM_RedefineClasses22AdjustAndCleanMetadata8do_klassEP5Klass:function(e){this.doKlass=e},_ZTV18VM_RedefineClasses:function(e){this.vtableRedefineClasses=e},_ZN18VM_RedefineClasses4doitEv:function(e){this.redefineClassesDoIt=e},_ZN18VM_RedefineClasses13doit_prologueEv:function(e){this.redefineClassesDoItPrologue=e},_ZN18VM_RedefineClasses13doit_epilogueEv:function(e){this.redefineClassesDoItEpilogue=e},_ZN18VM_RedefineClassesD0Ev:function(e){this.redefineClassesDispose0=e},_ZN18VM_RedefineClassesD1Ev:function(e){this.redefineClassesDispose1=e},_ZNK18VM_RedefineClasses26allow_nested_vm_operationsEv:function(e){this.redefineClassesAllow=e},_ZNK18VM_RedefineClasses14print_on_errorEP12outputStream:function(e){this.redefineClassesOnError=e},_ZN13InstanceKlass33create_new_default_vtable_indicesEiP10JavaThread:function(e){this.createNewDefaultVtableIndices=e},_ZN13InstanceKlass33create_new_default_vtable_indicesEiP6Thread:function(e){this.createNewDefaultVtableIndices=e},_ZN19Abstract_VM_Version19jre_release_versionEv:function(e){let t=new NativeFunction(e,"pointer",[],f)().readCString();this.version=t.startsWith("1.8")?8:t.startsWith("9.")?9:parseInt(t.slice(0,2),10),this.versionS=t},_ZN14NMethodSweeper11_traversalsE:function(e){this.traversals=e},_ZN14NMethodSweeper21_sweep_fractions_leftE:function(e){this.fractions=e},_ZN14NMethodSweeper13_should_sweepE:function(e){this.shouldSweep=e}},optionals:["_ZN6Method24restore_unshareable_infoEP10JavaThread","_ZN6Method24restore_unshareable_infoEP6Thread","_ZN6Method11link_methodERK12methodHandleP10JavaThread","_ZN6Method10clear_codeEv","_ZN6Method10clear_codeEb","_ZN18VM_RedefineClasses19mark_dependent_codeEP13InstanceKlass","_ZN18VM_RedefineClasses20flush_dependent_codeEv","_ZN18VM_RedefineClasses20flush_dependent_codeEP13InstanceKlassP6Thread","_ZN18VM_RedefineClasses20flush_dependent_codeE19instanceKlassHandleP6Thread","_ZN19ResolvedMethodTable21adjust_method_entriesEPb","_ZN15MemberNameTable21adjust_method_entriesEP13InstanceKlassPb","_ZN17ConstantPoolCache21adjust_method_entriesEPb","_ZN17ConstantPoolCache21adjust_method_entriesEP13InstanceKlassPb","_ZN20ClassLoaderDataGraph22clean_deallocate_listsEb","_ZN10JavaThread27thread_from_jni_environmentEP7JNIEnv_","_ZN14NMethodSweeper11force_sweepEv","_ZN14NMethodSweeper17sweep_in_progressEv","_ZN18VM_RedefineClasses14_the_class_oopE","_ZN18VM_RedefineClasses10_the_classE","_ZN18VM_RedefineClasses25AdjustCpoolCacheAndVtable8do_klassEP5Klass","_ZN18VM_RedefineClasses22AdjustAndCleanMetadata8do_klassEP5Klass","_ZN18VM_RedefineClassesD0Ev","_ZN18VM_RedefineClassesD1Ev","_ZNK18VM_RedefineClasses14print_on_errorEP12outputStream","_ZN13InstanceKlass33create_new_default_vtable_indicesEiP10JavaThread","_ZN13InstanceKlass33create_new_default_vtable_indicesEiP6Thread","_ZN14NMethodSweeper21_sweep_fractions_leftE"]}],l=[];if(r.forEach(function(e){let t=e.module,n=e.functions||{},r=e.variables||{},i=new Set(e.optionals||[]),a=t.enumerateExports().reduce(function(e,t){return e[t.name]=t,e},{}),o=t.enumerateSymbols().reduce(function(e,t){return e[t.name]=t,e},a);Object.keys(n).forEach(function(t){let r=o[t];if(r!==void 0){let e=n[t];typeof e=="function"?e.call(s,r.address):s[e[0]]=new NativeFunction(r.address,e[1],e[2],f)}else i.has(t)||l.push(t)}),Object.keys(r).forEach(function(e){let t=o[e];t!==void 0?r[e].call(s,t.address):i.has(e)||l.push(e)})}),l.length>0)throw new Error("Java API only partially available; please file a bug. Missing: "+l.join(", "));let n=Memory.alloc(C),i=Memory.alloc(Wo);if(h("JNI_GetCreatedJavaVMs",s.JNI_GetCreatedJavaVMs(n,1,i)),i.readInt()===0)return null;s.vm=n.readPointer();let a=Process.platform==="windows"?{$new:["??2@YAPEAX_K@Z","pointer",["ulong"]],$delete:["??3@YAXPEAX@Z","void",["pointer"]]}:{$new:["_Znwm","pointer",["ulong"]],$delete:["_ZdlPv","void",["pointer"]]};for(let[t,[r,n,i]]of Object.entries(a)){let e=Module.findGlobalExportByName(r);if(e===null&&(e=DebugSymbol.fromName(r).address,e.isNull()))throw new Error(`unable to find C++ allocator API, missing: '${r}'`);s[t]=new NativeFunction(e,n,i,f)}return s.jvmti=cs(s),s["JavaThread::thread_from_jni_environment"]===void 0&&(s["JavaThread::thread_from_jni_environment"]=hs(s)),s}function cs(e){let n=new Hr(e),i;return n.perform(()=>{let e=n.tryGetEnvHandle(qe.v1_0);if(e===null)throw new Error("JVMTI not available");i=new c(e,n);let t=Memory.alloc(8);t.writeU64(We.canTagObjects);let r=i.addCapabilities(t);h("getEnvJvmti::AddCapabilities",r)}),i}var us={x64:ps};function hs(t){let r=null,n=us[Process.arch];if(n!==void 0){let e=new Hr(t).perform(e=>e.handle.readPointer().add(6*C).readPointer());r=p(e,n,{limit:11})}return r===null?()=>{throw new Error("Unable to make thread_from_jni_environment() helper for the current architecture")}:e=>e.add(r)}function ps(e){if(e.mnemonic!=="lea")return null;let{base:t,disp:r}=e.operands[1].value;return t==="rdi"&&r<0?r:null}function fs(e,t){}var _s=class{constructor(e){this.methodId=e,this.method=e.readPointer(),this.originalMethod=null,this.newMethod=null,this.resolved=null,this.impl=null,this.key=e.toString(16)}replace(e,t,r,n,i){let{key:a}=this,o=ls.get(a);o!==void 0&&(ls.delete(a),this.method=o.method,this.originalMethod=o.originalMethod,this.newMethod=o.newMethod,this.resolved=o.resolved),this.impl=e,ss.set(a,this),ms(n)}revert(e){let{key:t}=this;ss.delete(t),ls.set(t,this),ms(e)}resolveTarget(e,t,r,n){let{resolved:i,originalMethod:a,methodId:o}=this;if(i!==null)return i;if(a===null)return o;a.oldMethod.vtableIndexPtr.writeS32(-2);let s=Memory.alloc(C);return s.writePointer(this.method),this.resolved=s,s}};function ms(e){os||(os=!0,Script.nextTick(gs,e))}function gs(e){let t=new Map(ss),r=new Map(ls);ss.clear(),ls.clear(),os=!1,e.perform(e=>{let o=j(),s=o["JavaThread::thread_from_jni_environment"](e.handle),i=!1;vs(()=>{t.forEach(e=>{let{method:t,originalMethod:r,impl:n,methodId:i,newMethod:a}=e;r===null?(e.originalMethod=Ns(t),e.newMethod=Ss(t,n,s),Es(e.newMethod,i,s)):o["Method::set_native_function"](a.method,n,0)}),r.forEach(e=>{let{originalMethod:t,methodId:r,newMethod:n}=e;if(t!==null){ks(t);let e=t.oldMethod;e.oldMethod=n,Es(e,r,s),i=!0}})}),i&&bs(e.handle)})}function bs(r){let{fractions:n,shouldSweep:i,traversals:a,"NMethodSweeper::sweep_code_cache":o,"NMethodSweeper::sweep_in_progress":s,"NMethodSweeper::force_sweep":e,JVM_Sleep:l}=j();if(e!==void 0)Thread.sleep(.05),e(),Thread.sleep(.05),e();else{let e=a.readS64(),t=e+2;for(;t>e;)n.writeS32(1),l(r,NULL,50),s()||vs(()=>{Thread.sleep(.05)}),i.readU8()===0&&(n.writeS32(1),o()),e=a.readS64()}}function vs(e,t,r){let{execute:n,vtable:i,vtableSize:a,doItOffset:o,prologueOffset:s,epilogueOffset:l}=is(),d=Memory.dup(i,a),c=Memory.alloc(C*25);c.writePointer(d);let u=new NativeCallback(e,"void",["pointer"]);d.add(o).writePointer(u);let h=null;t!==void 0&&(h=new NativeCallback(t,"int",["pointer"]),d.add(s).writePointer(h));let p=null;r!==void 0&&(p=new NativeCallback(r,"void",["pointer"]),d.add(l).writePointer(p)),n(c)}function ys(){let{vtableRedefineClasses:e,redefineClassesDoIt:n,redefineClassesDoItPrologue:i,redefineClassesDoItEpilogue:a,redefineClassesOnError:o,redefineClassesAllow:s,redefineClassesDispose0:l,redefineClassesDispose1:d,"VMThread::execute":t}=j(),r=e.add(2*C),c=15*C,u=Memory.dup(r,c),h=new NativeCallback(()=>{},"void",["pointer"]),p,f,_;for(let r=0;r!==c;r+=C){let e=u.add(r),t=e.readPointer();o!==void 0&&t.equals(o)||l!==void 0&&t.equals(l)||d!==void 0&&t.equals(d)?e.writePointer(h):t.equals(n)?p=r:t.equals(i)?(f=r,e.writePointer(s)):t.equals(a)&&(_=r,e.writePointer(h))}return{execute:t,emptyCallback:h,vtable:u,vtableSize:c,doItOffset:p,prologueOffset:f,epilogueOffset:_}}function ws(e){return new _s(e)}function Es(r,e,t){let{method:n,oldMethod:i}=r,a=j();r.methodsArray.add(r.methodIndex*C).writePointer(n),r.vtableIndex>=0&&r.vtable.add(r.vtableIndex*C).writePointer(n),e.writePointer(n),i.accessFlagsPtr.writeU32((i.accessFlags|Qo|Xo)>>>0);let o=a["OopMapCache::flush_obsolete_entries"];if(o!==void 0){let{oopMapCache:e}=r;e.isNull()||o(e)}let s=a["VM_RedefineClasses::mark_dependent_code"],l=a["VM_RedefineClasses::flush_dependent_code"];s!==void 0?(s(NULL,r.instanceKlass),l()):l(NULL,r.instanceKlass,t);let d=Memory.alloc(1);d.writeU8(1),a["ConstantPoolCache::adjust_method_entries"](r.cache,r.instanceKlass,d);let c=Memory.alloc(3*C),u=Memory.alloc(C);u.writePointer(a.doKlass),c.writePointer(u),c.add(C).writePointer(t),c.add(2*C).writePointer(t),a.redefineClass!==void 0&&a.redefineClass.writePointer(r.instanceKlass),a["ClassLoaderDataGraph::classes_do"](c);let h=a["ResolvedMethodTable::adjust_method_entries"];if(h!==void 0)h(d);else{let{memberNames:t}=r;if(!t.isNull()){let e=a["MemberNameTable::adjust_method_entries"];e!==void 0&&e(t,r.instanceKlass,d)}}let p=a["ClassLoaderDataGraph::clean_deallocate_lists"];p!==void 0&&p(0)}function Ss(e,t,r){let n=j(),i=Ns(e);i.constPtr.writePointer(i.const);let a=(i.accessFlags|Ko|Yo|es|ts)>>>0;if(i.accessFlagsPtr.writeU32(a),i.signatureHandler.writePointer(NULL),i.adapter.writePointer(NULL),i.i2iEntry.writePointer(NULL),n["Method::clear_code"](i.method),i.dataPtr.writePointer(NULL),i.countersPtr.writePointer(NULL),i.stackmapPtr.writePointer(NULL),n["Method::clear_native_function"](i.method),n["Method::set_native_function"](i.method,t,0),n["Method::restore_unshareable_info"](i.method,r),n.version>=17){let e=Memory.alloc(2*C);e.writePointer(i.method),e.add(C).writePointer(r),n["Method::link_method"](i.method,e,r)}return i}function Ns(e){let t=rs(),r=e.add(t.method.constMethodOffset).readPointer(),n=r.add(t.constMethod.sizeOffset).readS32()*C,i=Memory.alloc(n+t.method.size);Memory.copy(i,r,n);let a=i.add(n);Memory.copy(a,e,t.method.size);let o=Ls(a,i,n),s=Ls(e,r,n);return o.oldMethod=s,o}function Ls(e,t,r){let n=j(),i=rs(),a=e.add(i.method.constMethodOffset),o=e.add(i.method.methodDataOffset),s=e.add(i.method.methodCountersOffset),l=e.add(i.method.accessFlagsOffset),d=l.readU32(),c=i.getAdapterPointer(e,t),u=e.add(i.method.i2iEntryOffset),h=e.add(i.method.signatureHandlerOffset),p=t.add(i.constMethod.constantPoolOffset).readPointer(),f=t.add(i.constMethod.stackmapDataOffset),_=p.add(i.constantPool.instanceKlassOffset).readPointer(),m=p.add(i.constantPool.cacheOffset).readPointer(),g=ns(),b=_.add(g.methodsOffset).readPointer(),v=b.readS32(),y=b.add(C),w=t.add(i.constMethod.methodIdnumOffset).readU16(),E=e.add(i.method.vtableIndexOffset),S=E.readS32(),N=_.add(g.vtableOffset),L=_.add(g.oopMapCacheOffset).readPointer(),k=n.version>=10?_.add(g.memberNamesOffset).readPointer():NULL;return{method:e,methodSize:i.method.size,const:t,constSize:r,constPtr:a,dataPtr:o,countersPtr:s,stackmapPtr:f,instanceKlass:_,methodsArray:y,methodsCount:v,methodIndex:w,vtableIndex:S,vtableIndexPtr:E,vtable:N,accessFlags:d,accessFlagsPtr:l,adapter:c,i2iEntry:u,signatureHandler:h,memberNames:k,cache:m,oopMapCache:L}}function ks(e){let{oldMethod:t}=e;t.accessFlagsPtr.writeU32(t.accessFlags),t.vtableIndexPtr.writeS32(t.vtableIndex)}function Cs(){let e=j(),{version:t}=e,r;t>=17?r="method:early":t>=9&&t<=16?r="const-method":r="method:late";let n=e["Method::size"](1)*C,i=C,a=2*C,o=3*C,s=4*C,l=r==="method:early"?C:0,d=s+l,c=d+4,u=c+4+8,h=u+C,p=l!==0?s:h,f=n-2*C,_=n-C,m=8,g=m+C,b=g+C,v=r==="const-method"?C:0,y=b+v,w=y+14,E=2*C,S=3*C;return{getAdapterPointer:v!==0?function(e,t){return t.add(b)}:function(e,t){return e.add(p)},method:{size:n,constMethodOffset:i,methodDataOffset:a,methodCountersOffset:o,accessFlagsOffset:d,vtableIndexOffset:c,i2iEntryOffset:u,nativeFunctionOffset:f,signatureHandlerOffset:_},constMethod:{constantPoolOffset:m,stackmapDataOffset:g,sizeOffset:y,methodIdnumOffset:w},constantPool:{cacheOffset:E,instanceKlassOffset:S}}}var js={x64:Is};function Ms(){let{version:e,createNewDefaultVtableIndices:t}=j(),r=js[Process.arch];if(r===void 0)throw new Error(`Missing vtable offset parser for ${Process.arch}`);let n=p(t,r,{limit:32});if(n===null)throw new Error("Unable to deduce vtable offset");let i=e>=10&&e<=11||e>=15?17:18,a=n-7*C,o=n-17*C,s=n-i*C;return{vtableOffset:n,methodsOffset:a,memberNamesOffset:o,oopMapCacheOffset:s}}function Is(e){if(e.mnemonic!=="mov")return null;let t=e.operands[0];if(t.type!=="mem")return null;let{value:r}=t;if(r.scale!==1)return null;let{disp:n}=r;return n<256?null:n+16}var Ps=S;try{Xn()}catch{Ps=j}var Rs=Ps;var As=`#include <json-glib/json-glib.h>
#include <string.h>

#define kAccStatic 0x0008
#define kAccConstructor 0x00010000

typedef struct _Model Model;
typedef struct _EnumerateMethodsContext EnumerateMethodsContext;

typedef struct _JavaApi JavaApi;
typedef struct _JavaClassApi JavaClassApi;
typedef struct _JavaMethodApi JavaMethodApi;
typedef struct _JavaFieldApi JavaFieldApi;

typedef struct _JNIEnv JNIEnv;
typedef guint8 jboolean;
typedef gint32 jint;
typedef jint jsize;
typedef gpointer jobject;
typedef jobject jclass;
typedef jobject jstring;
typedef jobject jarray;
typedef jarray jobjectArray;
typedef gpointer jfieldID;
typedef gpointer jmethodID;

typedef struct _jvmtiEnv jvmtiEnv;
typedef enum
{
  JVMTI_ERROR_NONE = 0
} jvmtiError;

typedef struct _ArtApi ArtApi;
typedef guint32 ArtHeapReference;
typedef struct _ArtObject ArtObject;
typedef struct _ArtClass ArtClass;
typedef struct _ArtClassLinker ArtClassLinker;
typedef struct _ArtClassVisitor ArtClassVisitor;
typedef struct _ArtClassVisitorVTable ArtClassVisitorVTable;
typedef struct _ArtMethod ArtMethod;
typedef struct _ArtString ArtString;

typedef union _StdString StdString;
typedef struct _StdStringShort StdStringShort;
typedef struct _StdStringLong StdStringLong;

typedef void (* ArtVisitClassesFunc) (ArtClassLinker * linker, ArtClassVisitor * visitor);
typedef const char * (* ArtGetClassDescriptorFunc) (ArtClass * klass, StdString * storage);
typedef void (* ArtPrettyMethodFunc) (StdString * result, ArtMethod * method, jboolean with_signature);

struct _Model
{
  GHashTable * members;
};

struct _EnumerateMethodsContext
{
  GPatternSpec * class_query;
  GPatternSpec * method_query;
  jboolean include_signature;
  jboolean ignore_case;
  jboolean skip_system_classes;
  GHashTable * groups;
};

struct _JavaClassApi
{
  jmethodID get_declared_methods;
  jmethodID get_declared_fields;
};

struct _JavaMethodApi
{
  jmethodID get_name;
  jmethodID get_modifiers;
};

struct _JavaFieldApi
{
  jmethodID get_name;
  jmethodID get_modifiers;
};

struct _JavaApi
{
  jvmtiEnv * jvmti;
  JavaClassApi clazz;
  JavaMethodApi method;
  JavaFieldApi field;
};

struct _JNIEnv
{
  gpointer * functions;
};

struct _jvmtiEnv
{
  gpointer * functions;
};

struct _ArtApi
{
  gboolean available;

  guint class_offset_ifields;
  guint class_offset_methods;
  guint class_offset_sfields;
  guint class_offset_copied_methods_offset;

  guint method_size;
  guint method_offset_access_flags;

  guint field_size;
  guint field_offset_access_flags;

  guint alignment_padding;

  ArtClassLinker * linker;
  ArtVisitClassesFunc visit_classes;
  ArtGetClassDescriptorFunc get_class_descriptor;
  ArtPrettyMethodFunc pretty_method;

  void (* free) (gpointer mem);
};

struct _ArtObject
{
  ArtHeapReference klass;
  ArtHeapReference monitor;
};

struct _ArtClass
{
  ArtObject parent;

  ArtHeapReference class_loader;
};

struct _ArtClassVisitor
{
  ArtClassVisitorVTable * vtable;
  gpointer user_data;
};

struct _ArtClassVisitorVTable
{
  void (* reserved1) (ArtClassVisitor * self);
  void (* reserved2) (ArtClassVisitor * self);
  jboolean (* visit) (ArtClassVisitor * self, ArtClass * klass);
};

struct _ArtString
{
  ArtObject parent;

  gint32 count;
  guint32 hash_code;

  union
  {
    guint16 value[0];
    guint8 value_compressed[0];
  };
};

struct _StdStringShort
{
  guint8 size;
  gchar data[(3 * sizeof (gpointer)) - sizeof (guint8)];
};

struct _StdStringLong
{
  gsize capacity;
  gsize size;
  gchar * data;
};

union _StdString
{
  StdStringShort s;
  StdStringLong l;
};

static void model_add_method (Model * self, const gchar * name, jmethodID id, jint modifiers);
static void model_add_field (Model * self, const gchar * name, jfieldID id, jint modifiers);
static void model_free (Model * model);

static jboolean collect_matching_class_methods (ArtClassVisitor * self, ArtClass * klass);
static gchar * finalize_method_groups_to_json (GHashTable * groups);
static GPatternSpec * make_pattern_spec (const gchar * pattern, jboolean ignore_case);
static gchar * class_name_from_signature (const gchar * signature);
static gchar * format_method_signature (const gchar * name, const gchar * signature);
static void append_type (GString * output, const gchar ** type);

static gpointer read_art_array (gpointer object_base, guint field_offset, guint length_size, guint * length);

static void std_string_destroy (StdString * str);
static gchar * std_string_c_str (StdString * self);

extern GMutex lock;
extern GArray * models;
extern JavaApi java_api;
extern ArtApi art_api;

void
init (void)
{
  g_mutex_init (&lock);
  models = g_array_new (FALSE, FALSE, sizeof (Model *));
}

void
finalize (void)
{
  guint n, i;

  n = models->len;
  for (i = 0; i != n; i++)
  {
    Model * model = g_array_index (models, Model *, i);
    model_free (model);
  }

  g_array_unref (models);
  g_mutex_clear (&lock);
}

Model *
model_new (jclass class_handle,
           gpointer class_object,
           JNIEnv * env)
{
  Model * model;
  GHashTable * members;
  jvmtiEnv * jvmti = java_api.jvmti;
  gpointer * funcs = env->functions;
  jmethodID (* from_reflected_method) (JNIEnv *, jobject) = funcs[7];
  jfieldID (* from_reflected_field) (JNIEnv *, jobject) = funcs[8];
  jobject (* to_reflected_method) (JNIEnv *, jclass, jmethodID, jboolean) = funcs[9];
  jobject (* to_reflected_field) (JNIEnv *, jclass, jfieldID, jboolean) = funcs[12];
  void (* delete_local_ref) (JNIEnv *, jobject) = funcs[23];
  jobject (* call_object_method) (JNIEnv *, jobject, jmethodID, ...) = funcs[34];
  jint (* call_int_method) (JNIEnv *, jobject, jmethodID, ...) = funcs[49];
  const char * (* get_string_utf_chars) (JNIEnv *, jstring, jboolean *) = funcs[169];
  void (* release_string_utf_chars) (JNIEnv *, jstring, const char *) = funcs[170];
  jsize (* get_array_length) (JNIEnv *, jarray) = funcs[171];
  jobject (* get_object_array_element) (JNIEnv *, jobjectArray, jsize) = funcs[173];
  jsize n, i;

  model = g_new (Model, 1);

  members = g_hash_table_new_full (g_str_hash, g_str_equal, g_free, g_free);
  model->members = members;

  if (jvmti != NULL)
  {
    gpointer * jf = jvmti->functions - 1;
    jvmtiError (* deallocate) (jvmtiEnv *, void * mem) = jf[47];
    jvmtiError (* get_class_methods) (jvmtiEnv *, jclass, jint *, jmethodID **) = jf[52];
    jvmtiError (* get_class_fields) (jvmtiEnv *, jclass, jint *, jfieldID **) = jf[53];
    jvmtiError (* get_field_name) (jvmtiEnv *, jclass, jfieldID, char **, char **, char **) = jf[60];
    jvmtiError (* get_field_modifiers) (jvmtiEnv *, jclass, jfieldID, jint *) = jf[62];
    jvmtiError (* get_method_name) (jvmtiEnv *, jmethodID, char **, char **, char **) = jf[64];
    jvmtiError (* get_method_modifiers) (jvmtiEnv *, jmethodID, jint *) = jf[66];
    jint method_count;
    jmethodID * methods;
    jint field_count;
    jfieldID * fields;
    char * name;
    jint modifiers;

    get_class_methods (jvmti, class_handle, &method_count, &methods);
    for (i = 0; i != method_count; i++)
    {
      jmethodID method = methods[i];

      get_method_name (jvmti, method, &name, NULL, NULL);
      get_method_modifiers (jvmti, method, &modifiers);

      model_add_method (model, name, method, modifiers);

      deallocate (jvmti, name);
    }
    deallocate (jvmti, methods);

    get_class_fields (jvmti, class_handle, &field_count, &fields);
    for (i = 0; i != field_count; i++)
    {
      jfieldID field = fields[i];

      get_field_name (jvmti, class_handle, field, &name, NULL, NULL);
      get_field_modifiers (jvmti, class_handle, field, &modifiers);

      model_add_field (model, name, field, modifiers);

      deallocate (jvmti, name);
    }
    deallocate (jvmti, fields);
  }
  else if (art_api.available)
  {
    gpointer elements;
    guint n, i;
    const guint field_arrays[] = {
      art_api.class_offset_ifields,
      art_api.class_offset_sfields
    };
    guint field_array_cursor;
    gboolean merged_fields = art_api.class_offset_sfields == 0;

    elements = read_art_array (class_object, art_api.class_offset_methods, sizeof (gsize), NULL);
    n = *(guint16 *) (class_object + art_api.class_offset_copied_methods_offset);
    for (i = 0; i != n; i++)
    {
      jmethodID id;
      guint32 access_flags;
      jboolean is_static;
      jobject method, name;
      const char * name_str;
      jint modifiers;

      id = elements + (i * art_api.method_size);

      access_flags = *(guint32 *) (id + art_api.method_offset_access_flags);
      if ((access_flags & kAccConstructor) != 0)
        continue;
      is_static = (access_flags & kAccStatic) != 0;
      method = to_reflected_method (env, class_handle, id, is_static);
      name = call_object_method (env, method, java_api.method.get_name);
      name_str = get_string_utf_chars (env, name, NULL);
      modifiers = access_flags & 0xffff;

      model_add_method (model, name_str, id, modifiers);

      release_string_utf_chars (env, name, name_str);
      delete_local_ref (env, name);
      delete_local_ref (env, method);
    }

    for (field_array_cursor = 0; field_array_cursor != G_N_ELEMENTS (field_arrays); field_array_cursor++)
    {
      jboolean is_static;

      if (field_arrays[field_array_cursor] == 0)
        continue;

      if (!merged_fields)
        is_static = field_array_cursor == 1;

      elements = read_art_array (class_object, field_arrays[field_array_cursor], sizeof (guint32), &n);
      for (i = 0; i != n; i++)
      {
        jfieldID id;
        guint32 access_flags;
        jobject field, name;
        const char * name_str;
        jint modifiers;

        id = elements + (i * art_api.field_size);

        access_flags = *(guint32 *) (id + art_api.field_offset_access_flags);
        if (merged_fields)
          is_static = (access_flags & kAccStatic) != 0;
        field = to_reflected_field (env, class_handle, id, is_static);
        name = call_object_method (env, field, java_api.field.get_name);
        name_str = get_string_utf_chars (env, name, NULL);
        modifiers = access_flags & 0xffff;

        model_add_field (model, name_str, id, modifiers);

        release_string_utf_chars (env, name, name_str);
        delete_local_ref (env, name);
        delete_local_ref (env, field);
      }
    }
  }
  else
  {
    jobject elements;

    elements = call_object_method (env, class_handle, java_api.clazz.get_declared_methods);
    n = get_array_length (env, elements);
    for (i = 0; i != n; i++)
    {
      jobject method, name;
      const char * name_str;
      jmethodID id;
      jint modifiers;

      method = get_object_array_element (env, elements, i);
      name = call_object_method (env, method, java_api.method.get_name);
      name_str = get_string_utf_chars (env, name, NULL);
      id = from_reflected_method (env, method);
      modifiers = call_int_method (env, method, java_api.method.get_modifiers);

      model_add_method (model, name_str, id, modifiers);

      release_string_utf_chars (env, name, name_str);
      delete_local_ref (env, name);
      delete_local_ref (env, method);
    }
    delete_local_ref (env, elements);

    elements = call_object_method (env, class_handle, java_api.clazz.get_declared_fields);
    n = get_array_length (env, elements);
    for (i = 0; i != n; i++)
    {
      jobject field, name;
      const char * name_str;
      jfieldID id;
      jint modifiers;

      field = get_object_array_element (env, elements, i);
      name = call_object_method (env, field, java_api.field.get_name);
      name_str = get_string_utf_chars (env, name, NULL);
      id = from_reflected_field (env, field);
      modifiers = call_int_method (env, field, java_api.field.get_modifiers);

      model_add_field (model, name_str, id, modifiers);

      release_string_utf_chars (env, name, name_str);
      delete_local_ref (env, name);
      delete_local_ref (env, field);
    }
    delete_local_ref (env, elements);
  }

  g_mutex_lock (&lock);
  g_array_append_val (models, model);
  g_mutex_unlock (&lock);

  return model;
}

static void
model_add_method (Model * self,
                  const gchar * name,
                  jmethodID id,
                  jint modifiers)
{
  GHashTable * members = self->members;
  gchar * key, type;
  const gchar * value;

  if (name[0] == '$')
    key = g_strdup_printf ("_%s", name);
  else
    key = g_strdup (name);

  type = (modifiers & kAccStatic) != 0 ? 's' : 'i';

  value = g_hash_table_lookup (members, key);
  if (value == NULL)
    g_hash_table_insert (members, key, g_strdup_printf ("m:%c0x%zx", type, id));
  else
    g_hash_table_insert (members, key, g_strdup_printf ("%s:%c0x%zx", value, type, id));
}

static void
model_add_field (Model * self,
                 const gchar * name,
                 jfieldID id,
                 jint modifiers)
{
  GHashTable * members = self->members;
  gchar * key, type;

  if (name[0] == '$')
    key = g_strdup_printf ("_%s", name);
  else
    key = g_strdup (name);
  while (g_hash_table_contains (members, key))
  {
    gchar * new_key = g_strdup_printf ("_%s", key);
    g_free (key);
    key = new_key;
  }

  type = (modifiers & kAccStatic) != 0 ? 's' : 'i';

  g_hash_table_insert (members, key, g_strdup_printf ("f:%c0x%zx", type, id));
}

static void
model_free (Model * model)
{
  g_hash_table_unref (model->members);

  g_free (model);
}

gboolean
model_has (Model * self,
           const gchar * member)
{
  return g_hash_table_contains (self->members, member);
}

const gchar *
model_find (Model * self,
            const gchar * member)
{
  return g_hash_table_lookup (self->members, member);
}

gchar *
model_list (Model * self)
{
  GString * result;
  GHashTableIter iter;
  guint i;
  const gchar * name;

  result = g_string_sized_new (128);

  g_string_append_c (result, '[');

  g_hash_table_iter_init (&iter, self->members);
  for (i = 0; g_hash_table_iter_next (&iter, (gpointer *) &name, NULL); i++)
  {
    if (i > 0)
      g_string_append_c (result, ',');

    g_string_append_c (result, '"');
    g_string_append (result, name);
    g_string_append_c (result, '"');
  }

  g_string_append_c (result, ']');

  return g_string_free (result, FALSE);
}

gchar *
enumerate_methods_art (const gchar * class_query,
                       const gchar * method_query,
                       jboolean include_signature,
                       jboolean ignore_case,
                       jboolean skip_system_classes)
{
  gchar * result;
  EnumerateMethodsContext ctx;
  ArtClassVisitor visitor;
  ArtClassVisitorVTable visitor_vtable = { NULL, };

  ctx.class_query = make_pattern_spec (class_query, ignore_case);
  ctx.method_query = make_pattern_spec (method_query, ignore_case);
  ctx.include_signature = include_signature;
  ctx.ignore_case = ignore_case;
  ctx.skip_system_classes = skip_system_classes;
  ctx.groups = g_hash_table_new_full (NULL, NULL, NULL, NULL);

  visitor.vtable = &visitor_vtable;
  visitor.user_data = &ctx;

  visitor_vtable.visit = collect_matching_class_methods;

  art_api.visit_classes (art_api.linker, &visitor);

  result = finalize_method_groups_to_json (ctx.groups);

  g_hash_table_unref (ctx.groups);
  g_pattern_spec_free (ctx.method_query);
  g_pattern_spec_free (ctx.class_query);

  return result;
}

static jboolean
collect_matching_class_methods (ArtClassVisitor * self,
                                ArtClass * klass)
{
  EnumerateMethodsContext * ctx = self->user_data;
  const char * descriptor;
  StdString descriptor_storage = { 0, };
  gchar * class_name = NULL;
  gchar * class_name_copy = NULL;
  const gchar * normalized_class_name;
  JsonBuilder * group;
  size_t class_name_length;
  GHashTable * seen_method_names;
  gpointer elements;
  guint n, i;

  if (ctx->skip_system_classes && klass->class_loader == 0)
    goto skip_class;

  descriptor = art_api.get_class_descriptor (klass, &descriptor_storage);
  if (descriptor[0] != 'L')
    goto skip_class;

  class_name = class_name_from_signature (descriptor);

  if (ctx->ignore_case)
  {
    class_name_copy = g_utf8_strdown (class_name, -1);
    normalized_class_name = class_name_copy;
  }
  else
  {
    normalized_class_name = class_name;
  }

  if (!g_pattern_match_string (ctx->class_query, normalized_class_name))
    goto skip_class;

  group = NULL;
  class_name_length = strlen (class_name);
  seen_method_names = ctx->include_signature ? NULL : g_hash_table_new_full (g_str_hash, g_str_equal, g_free, NULL);

  elements = read_art_array (klass, art_api.class_offset_methods, sizeof (gsize), NULL);
  n = *(guint16 *) ((gpointer) klass + art_api.class_offset_copied_methods_offset);
  for (i = 0; i != n; i++)
  {
    ArtMethod * method;
    guint32 access_flags;
    jboolean is_constructor;
    StdString method_name = { 0, };
    const gchar * bare_method_name;
    gchar * bare_method_name_copy = NULL;
    const gchar * normalized_method_name;
    gchar * normalized_method_name_copy = NULL;

    method = elements + (i * art_api.method_size);

    access_flags = *(guint32 *) ((gpointer) method + art_api.method_offset_access_flags);
    is_constructor = (access_flags & kAccConstructor) != 0;

    art_api.pretty_method (&method_name, method, ctx->include_signature);
    bare_method_name = std_string_c_str (&method_name);
    if (ctx->include_signature)
    {
      const gchar * return_type_end, * name_begin;
      GString * name;

      return_type_end = strchr (bare_method_name, ' ');
      name_begin = return_type_end + 1 + class_name_length + 1;
      if (is_constructor && g_str_has_prefix (name_begin, "<clinit>"))
        goto skip_method;

      name = g_string_sized_new (64);

      if (is_constructor)
      {
        g_string_append (name, "$init");
        g_string_append (name, strchr (name_begin, '>') + 1);
      }
      else
      {
        g_string_append (name, name_begin);
      }
      g_string_append (name, ": ");
      g_string_append_len (name, bare_method_name, return_type_end - bare_method_name);

      bare_method_name_copy = g_string_free (name, FALSE);
      bare_method_name = bare_method_name_copy;
    }
    else
    {
      const gchar * name_begin;

      name_begin = bare_method_name + class_name_length + 1;
      if (is_constructor && strcmp (name_begin, "<clinit>") == 0)
        goto skip_method;

      if (is_constructor)
        bare_method_name = "$init";
      else
        bare_method_name += class_name_length + 1;
    }

    if (seen_method_names != NULL && g_hash_table_contains (seen_method_names, bare_method_name))
      goto skip_method;

    if (ctx->ignore_case)
    {
      normalized_method_name_copy = g_utf8_strdown (bare_method_name, -1);
      normalized_method_name = normalized_method_name_copy;
    }
    else
    {
      normalized_method_name = bare_method_name;
    }

    if (!g_pattern_match_string (ctx->method_query, normalized_method_name))
      goto skip_method;

    if (group == NULL)
    {
      group = g_hash_table_lookup (ctx->groups, GUINT_TO_POINTER (klass->class_loader));
      if (group == NULL)
      {
        group = json_builder_new_immutable ();
        g_hash_table_insert (ctx->groups, GUINT_TO_POINTER (klass->class_loader), group);

        json_builder_begin_object (group);

        json_builder_set_member_name (group, "loader");
        json_builder_add_int_value (group, klass->class_loader);

        json_builder_set_member_name (group, "classes");
        json_builder_begin_array (group);
      }

      json_builder_begin_object (group);

      json_builder_set_member_name (group, "name");
      json_builder_add_string_value (group, class_name);

      json_builder_set_member_name (group, "methods");
      json_builder_begin_array (group);
    }

    json_builder_add_string_value (group, bare_method_name);

    if (seen_method_names != NULL)
      g_hash_table_add (seen_method_names, g_strdup (bare_method_name));

skip_method:
    g_free (normalized_method_name_copy);
    g_free (bare_method_name_copy);
    std_string_destroy (&method_name);
  }

  if (seen_method_names != NULL)
    g_hash_table_unref (seen_method_names);

  if (group == NULL)
    goto skip_class;

  json_builder_end_array (group);
  json_builder_end_object (group);

skip_class:
  g_free (class_name_copy);
  g_free (class_name);
  std_string_destroy (&descriptor_storage);

  return TRUE;
}

gchar *
enumerate_methods_jvm (const gchar * class_query,
                       const gchar * method_query,
                       jboolean include_signature,
                       jboolean ignore_case,
                       jboolean skip_system_classes,
                       JNIEnv * env)
{
  gchar * result;
  GPatternSpec * class_pattern, * method_pattern;
  GHashTable * groups;
  gpointer * ef = env->functions;
  jobject (* new_global_ref) (JNIEnv *, jobject) = ef[21];
  void (* delete_local_ref) (JNIEnv *, jobject) = ef[23];
  jboolean (* is_same_object) (JNIEnv *, jobject, jobject) = ef[24];
  jvmtiEnv * jvmti = java_api.jvmti;
  gpointer * jf = jvmti->functions - 1;
  jvmtiError (* deallocate) (jvmtiEnv *, void * mem) = jf[47];
  jvmtiError (* get_class_signature) (jvmtiEnv *, jclass, char **, char **) = jf[48];
  jvmtiError (* get_class_methods) (jvmtiEnv *, jclass, jint *, jmethodID **) = jf[52];
  jvmtiError (* get_class_loader) (jvmtiEnv *, jclass, jobject *) = jf[57];
  jvmtiError (* get_method_name) (jvmtiEnv *, jmethodID, char **, char **, char **) = jf[64];
  jvmtiError (* get_loaded_classes) (jvmtiEnv *, jint *, jclass **) = jf[78];
  jint class_count, class_index;
  jclass * classes;

  class_pattern = make_pattern_spec (class_query, ignore_case);
  method_pattern = make_pattern_spec (method_query, ignore_case);
  groups = g_hash_table_new_full (NULL, NULL, NULL, NULL);

  if (get_loaded_classes (jvmti, &class_count, &classes) != JVMTI_ERROR_NONE)
    goto emit_results;

  for (class_index = 0; class_index != class_count; class_index++)
  {
    jclass klass = classes[class_index];
    jobject loader = NULL;
    gboolean have_loader = FALSE;
    char * signature = NULL;
    gchar * class_name = NULL;
    gchar * class_name_copy = NULL;
    const gchar * normalized_class_name;
    jint method_count, method_index;
    jmethodID * methods = NULL;
    JsonBuilder * group = NULL;
    GHashTable * seen_method_names = NULL;

    if (skip_system_classes)
    {
      if (get_class_loader (jvmti, klass, &loader) != JVMTI_ERROR_NONE)
        goto skip_class;
      have_loader = TRUE;

      if (loader == NULL)
        goto skip_class;
    }

    if (get_class_signature (jvmti, klass, &signature, NULL) != JVMTI_ERROR_NONE)
      goto skip_class;

    class_name = class_name_from_signature (signature);

    if (ignore_case)
    {
      class_name_copy = g_utf8_strdown (class_name, -1);
      normalized_class_name = class_name_copy;
    }
    else
    {
      normalized_class_name = class_name;
    }

    if (!g_pattern_match_string (class_pattern, normalized_class_name))
      goto skip_class;

    if (get_class_methods (jvmti, klass, &method_count, &methods) != JVMTI_ERROR_NONE)
      goto skip_class;

    if (!include_signature)
      seen_method_names = g_hash_table_new_full (g_str_hash, g_str_equal, g_free, NULL);

    for (method_index = 0; method_index != method_count; method_index++)
    {
      jmethodID method = methods[method_index];
      const gchar * method_name;
      char * method_name_value = NULL;
      char * method_signature_value = NULL;
      gchar * method_name_copy = NULL;
      const gchar * normalized_method_name;
      gchar * normalized_method_name_copy = NULL;

      if (get_method_name (jvmti, method, &method_name_value, include_signature ? &method_signature_value : NULL, NULL) != JVMTI_ERROR_NONE)
        goto skip_method;
      method_name = method_name_value;

      if (method_name[0] == '<')
      {
        if (strcmp (method_name, "<init>") == 0)
          method_name = "$init";
        else if (strcmp (method_name, "<clinit>") == 0)
          goto skip_method;
      }

      if (include_signature)
      {
        method_name_copy = format_method_signature (method_name, method_signature_value);
        method_name = method_name_copy;
      }

      if (seen_method_names != NULL && g_hash_table_contains (seen_method_names, method_name))
        goto skip_method;

      if (ignore_case)
      {
        normalized_method_name_copy = g_utf8_strdown (method_name, -1);
        normalized_method_name = normalized_method_name_copy;
      }
      else
      {
        normalized_method_name = method_name;
      }

      if (!g_pattern_match_string (method_pattern, normalized_method_name))
        goto skip_method;

      if (group == NULL)
      {
        if (!have_loader && get_class_loader (jvmti, klass, &loader) != JVMTI_ERROR_NONE)
          goto skip_method;

        if (loader == NULL)
        {
          group = g_hash_table_lookup (groups, NULL);
        }
        else
        {
          GHashTableIter iter;
          jobject cur_loader;
          JsonBuilder * cur_group;

          g_hash_table_iter_init (&iter, groups);
          while (g_hash_table_iter_next (&iter, (gpointer *) &cur_loader, (gpointer *) &cur_group))
          {
            if (cur_loader != NULL && is_same_object (env, cur_loader, loader))
            {
              group = cur_group;
              break;
            }
          }
        }

        if (group == NULL)
        {
          jobject l;
          gchar * str;

          l = (loader != NULL) ? new_global_ref (env, loader) : NULL;

          group = json_builder_new_immutable ();
          g_hash_table_insert (groups, l, group);

          json_builder_begin_object (group);

          json_builder_set_member_name (group, "loader");
          str = g_strdup_printf ("0x%" G_GSIZE_MODIFIER "x", GPOINTER_TO_SIZE (l));
          json_builder_add_string_value (group, str);
          g_free (str);

          json_builder_set_member_name (group, "classes");
          json_builder_begin_array (group);
        }

        json_builder_begin_object (group);

        json_builder_set_member_name (group, "name");
        json_builder_add_string_value (group, class_name);

        json_builder_set_member_name (group, "methods");
        json_builder_begin_array (group);
      }

      json_builder_add_string_value (group, method_name);

      if (seen_method_names != NULL)
        g_hash_table_add (seen_method_names, g_strdup (method_name));

skip_method:
      g_free (normalized_method_name_copy);
      g_free (method_name_copy);
      deallocate (jvmti, method_signature_value);
      deallocate (jvmti, method_name_value);
    }

skip_class:
    if (group != NULL)
    {
      json_builder_end_array (group);
      json_builder_end_object (group);
    }

    if (seen_method_names != NULL)
      g_hash_table_unref (seen_method_names);

    deallocate (jvmti, methods);

    g_free (class_name_copy);
    g_free (class_name);
    deallocate (jvmti, signature);

    if (loader != NULL)
      delete_local_ref (env, loader);

    delete_local_ref (env, klass);
  }

  deallocate (jvmti, classes);

emit_results:
  result = finalize_method_groups_to_json (groups);

  g_hash_table_unref (groups);
  g_pattern_spec_free (method_pattern);
  g_pattern_spec_free (class_pattern);

  return result;
}

static gchar *
finalize_method_groups_to_json (GHashTable * groups)
{
  GString * result;
  GHashTableIter iter;
  guint i;
  JsonBuilder * group;

  result = g_string_sized_new (1024);

  g_string_append_c (result, '[');

  g_hash_table_iter_init (&iter, groups);
  for (i = 0; g_hash_table_iter_next (&iter, NULL, (gpointer *) &group); i++)
  {
    JsonNode * root;
    gchar * json;

    if (i > 0)
      g_string_append_c (result, ',');

    json_builder_end_array (group);
    json_builder_end_object (group);

    root = json_builder_get_root (group);
    json = json_to_string (root, FALSE);
    g_string_append (result, json);
    g_free (json);
    json_node_unref (root);

    g_object_unref (group);
  }

  g_string_append_c (result, ']');

  return g_string_free (result, FALSE);
}

static GPatternSpec *
make_pattern_spec (const gchar * pattern,
                   jboolean ignore_case)
{
  GPatternSpec * spec;

  if (ignore_case)
  {
    gchar * str = g_utf8_strdown (pattern, -1);
    spec = g_pattern_spec_new (str);
    g_free (str);
  }
  else
  {
    spec = g_pattern_spec_new (pattern);
  }

  return spec;
}

static gchar *
class_name_from_signature (const gchar * descriptor)
{
  gchar * result, * c;

  result = g_strdup (descriptor + 1);

  for (c = result; *c != '\\0'; c++)
  {
    if (*c == '/')
      *c = '.';
  }

  c[-1] = '\\0';

  return result;
}

static gchar *
format_method_signature (const gchar * name,
                         const gchar * signature)
{
  GString * sig;
  const gchar * cursor;
  gint arg_index;

  sig = g_string_sized_new (128);

  g_string_append (sig, name);

  cursor = signature;
  arg_index = -1;
  while (TRUE)
  {
    const gchar c = *cursor;

    if (c == '(')
    {
      g_string_append_c (sig, c);
      cursor++;
      arg_index = 0;
    }
    else if (c == ')')
    {
      g_string_append_c (sig, c);
      cursor++;
      break;
    }
    else
    {
      if (arg_index >= 1)
        g_string_append (sig, ", ");

      append_type (sig, &cursor);

      if (arg_index != -1)
        arg_index++;
    }
  }

  g_string_append (sig, ": ");
  append_type (sig, &cursor);

  return g_string_free (sig, FALSE);
}

static void
append_type (GString * output,
             const gchar ** type)
{
  const gchar * cursor = *type;

  switch (*cursor)
  {
    case 'Z':
      g_string_append (output, "boolean");
      cursor++;
      break;
    case 'B':
      g_string_append (output, "byte");
      cursor++;
      break;
    case 'C':
      g_string_append (output, "char");
      cursor++;
      break;
    case 'S':
      g_string_append (output, "short");
      cursor++;
      break;
    case 'I':
      g_string_append (output, "int");
      cursor++;
      break;
    case 'J':
      g_string_append (output, "long");
      cursor++;
      break;
    case 'F':
      g_string_append (output, "float");
      cursor++;
      break;
    case 'D':
      g_string_append (output, "double");
      cursor++;
      break;
    case 'V':
      g_string_append (output, "void");
      cursor++;
      break;
    case 'L':
    {
      gchar ch;

      cursor++;
      for (; (ch = *cursor) != ';'; cursor++)
      {
        g_string_append_c (output, (ch != '/') ? ch : '.');
      }
      cursor++;

      break;
    }
    case '[':
      *type = cursor + 1;
      append_type (output, type);
      g_string_append (output, "[]");
      return;
    default:
      g_string_append (output, "BUG");
      cursor++;
  }

  *type = cursor;
}

void
dealloc (gpointer mem)
{
  g_free (mem);
}

static gpointer
read_art_array (gpointer object_base,
                guint field_offset,
                guint length_size,
                guint * length)
{
  gpointer result, header;
  guint n;

  header = GSIZE_TO_POINTER (*(guint64 *) (object_base + field_offset));
  if (header != NULL)
  {
    result = header + length_size;
    if (length_size == sizeof (guint32))
      n = *(guint32 *) header;
    else
      n = *(guint64 *) header;
  }
  else
  {
    result = NULL;
    n = 0;
  }

  if (length != NULL)
    *length = n;

  return result;
}

static void
std_string_destroy (StdString * str)
{
  if ((str->l.capacity & 1) != 0)
    art_api.free (str->l.data);
}

static gchar *
std_string_c_str (StdString * self)
{
  if ((self->l.capacity & 1) != 0)
    return self->l.data;

  return self->s.data;
}
`,xs=/(.+)!([^/]+)\/?([isu]+)?/,b=null,Ts=null,Us=class A{static build(t,r){return Os(r),Ts(t,r,e=>new A(b.new(t,e,r)))}static enumerateMethods(e,t,r){Os(r);let n=e.match(xs);if(n===null)throw new Error("Invalid query; format is: class!method -- see documentation of Java.enumerateMethods(query) for details");let a=Memory.allocUtf8String(n[1]),o=Memory.allocUtf8String(n[2]),s=!1,l=!1,d=!1,i=n[3];i!==void 0&&(s=i.indexOf("s")!==-1,l=i.indexOf("i")!==-1,d=i.indexOf("u")!==-1);let c;if(t.jvmti!==null){let e=b.enumerateMethodsJvm(a,o,$s(s),$s(l),$s(d),r);try{c=JSON.parse(e.readUtf8String()).map(e=>{let t=ptr(e.loader);return e.loader=t.isNull()?null:t,e})}finally{b.dealloc(e)}}else w(r.vm,r,i=>{let e=b.enumerateMethodsArt(a,o,$s(s),$s(l),$s(d));try{let r=t["art::JavaVMExt::AddGlobalRef"],{vm:n}=t;c=JSON.parse(e.readUtf8String()).map(e=>{let t=e.loader;return e.loader=t!==0?r(n,i,ptr(t)):null,e})}finally{b.dealloc(e)}});return c}constructor(e){this.handle=e}has(e){return b.has(this.handle,Memory.allocUtf8String(e))!==0}find(e){return b.find(this.handle,Memory.allocUtf8String(e)).readUtf8String()}list(){let e=b.list(this.handle);try{return JSON.parse(e.readUtf8String())}finally{b.dealloc(e)}}};function Os(e){b===null&&(b=Fs(e),Ts=Ds(b,e.vm))}function Fs(e){let i=S(),{jvmti:a=null}=i,{pointerSize:o}=Process,t=8,r=o,n=7*o,s=10*4+5*o,l=t+r+n+s,d=Memory.alloc(l),c=d.add(t),u=c.add(r),{getDeclaredMethods:h,getDeclaredFields:p}=e.javaLangClass(),f=e.javaLangReflectMethod(),_=e.javaLangReflectField(),m=u;[a!==null?a:NULL,h,p,f.getName,f.getModifiers,_.getName,_.getModifiers].forEach(e=>{m=m.writePointer(e).add(o)});let g=u.add(n),{vm:b}=e;if(i.flavor==="art"){let t;if(a!==null)t=[0,0,0,0];else{let e=$i(b).offset;t=[e.ifields,e.methods,e.sfields,e.copiedMethodsOffset]}let e=E(b),r=Ji(b),n=g;[1,...t,e.size,e.offset.accessFlags,r.size,r.offset.accessFlags,4294967295].forEach(e=>{n=n.writeUInt(e).add(4)}),[i.artClassLinker.address,i["art::ClassLinker::VisitClasses"],i["art::mirror::Class::GetDescriptor"],i["art::ArtMethod::PrettyMethod"],Process.getModuleByName("libc.so").getExportByName("free")].forEach((e,t)=>{e===void 0&&(e=NULL),n=n.writePointer(e).add(o)})}let v=new CModule(As,{lock:d,models:c,java_api:u,art_api:g}),y={exceptions:"propagate"},w={exceptions:"propagate",scheduling:"exclusive"};return{handle:v,new:new NativeFunction(v.model_new,"pointer",["pointer","pointer","pointer"],y),has:new NativeFunction(v.model_has,"bool",["pointer","pointer"],w),find:new NativeFunction(v.model_find,"pointer",["pointer","pointer"],w),list:new NativeFunction(v.model_list,"pointer",["pointer"],w),enumerateMethodsArt:new NativeFunction(v.enumerate_methods_art,"pointer",["pointer","pointer","bool","bool","bool"],y),enumerateMethodsJvm:new NativeFunction(v.enumerate_methods_jvm,"pointer",["pointer","pointer","bool","bool","bool","pointer"],y),dealloc:new NativeFunction(v.dealloc,"void",["pointer"],w)}}function Ds(e,a){let t=S();if(t.flavor!=="art")return zs;let o=t["art::JavaVMExt::DecodeGlobal"];return function(r,e,n){let i;return w(a,e,e=>{let t=o(a,e,r);i=n(t)}),i}}function zs(e,t,r){return r(NULL)}function $s(e){return e?1:0}var Vs=class{constructor(e,t){this.items=new Map,this.capacity=e,this.destroy=t}dispose(t){let{items:e,destroy:r}=this;e.forEach(e=>{r(e,t)}),e.clear()}get(e){let{items:t}=this,r=t.get(e);return r!==void 0&&(t.delete(e),t.set(e,r)),r}set(e,t,r){let{items:n}=this,i=n.get(e);if(i!==void 0)n.delete(e),this.destroy(i,r);else if(n.size===this.capacity){let e=n.keys().next().value,t=n.get(e);n.delete(e),this.destroy(t,r)}n.set(e,t)}};var Js=1,Bs=256,Gs=65536,Zs=305419896,Hs=32,qs=12,Ws=8,Ks=8,Qs=4,Xs=4,Ys=12,el=0,tl=1,rl=2,nl=3,il=4,al=5,ol=6,sl=4096,ll=4097,dl=4099,cl=8192,ul=8193,hl=8194,pl=8195,fl=8196,_l=8198,ml=24,gl=28,bl=2,vl=24,yl=Q.from([3,0,7,14,0]),wl="Ldalvik/annotation/Throws;",El=Q.from([0]);function Sl(e){let t=new Nl,r=Object.assign({},e);return t.addClass(r),t.build()}var Nl=class{constructor(){this.classes=[]}addClass(e){this.classes.push(e)}build(){let e=Cl(this.classes),{classes:r,interfaces:t,fields:n,methods:i,protos:a,parameters:o,annotationDirectories:s,annotationSets:l,throwsAnnotations:d,types:c,strings:u}=e,h=0,U=0,O=8,p=12,F=20,f=112;h+=f;let _=h,D=u.length*Xs;h+=D;let m=h,z=c.length*Qs;h+=z;let g=h,$=a.length*qs;h+=$;let b=h,V=n.length*Ws;h+=V;let v=h,J=i.length*Ks;h+=J;let y=h,B=r.length*Hs;h+=B;let w=h,E=l.map(e=>{let t=h;return e.offset=t,h+=4+e.items.length*4,t}),S=r.reduce((n,e)=>(e.classData.constructorMethods.forEach(e=>{let[,t,r]=e;(t&Bs)===0&&r>=0&&(e.push(h),n.push({offset:h,superConstructor:r}),h+=vl)}),n),[]);s.forEach(e=>{e.offset=h,h+=16+e.methods.length*8});let N=t.map(e=>{h=xl(h,4);let t=h;return e.offset=t,h+=4+2*e.types.length,t}),L=o.map(e=>{h=xl(h,4);let t=h;return e.offset=t,h+=4+2*e.types.length,t}),k=[],C=u.map(e=>{let t=h,r=Q.from(X(e.length)),n=Q.from(e,"utf8"),i=Q.concat([r,n,El]);return k.push(i),h+=i.length,t}),j=S.map(e=>{let t=h;return h+=yl.length,t}),G=d.map(e=>{let t=kl(e);return e.offset=h,h+=t.length,t}),Z=r.map((e,t)=>{e.classData.offset=h;let r=Ll(e);return h+=r.length,r}),H=0,q=0;h=xl(h,4);let M=h,I=t.length+o.length,P=4+(n.length>0?1:0)+2+l.length+S.length+s.length+(I>0?1:0)+1+j.length+d.length+r.length+1,W=4+P*Ys;h+=W;let K=h-w,R=h,A=Q.alloc(R);A.write(`dex
035`),A.writeUInt32LE(R,32),A.writeUInt32LE(f,36),A.writeUInt32LE(Zs,40),A.writeUInt32LE(H,44),A.writeUInt32LE(q,48),A.writeUInt32LE(M,52),A.writeUInt32LE(u.length,56),A.writeUInt32LE(_,60),A.writeUInt32LE(c.length,64),A.writeUInt32LE(m,68),A.writeUInt32LE(a.length,72),A.writeUInt32LE(g,76),A.writeUInt32LE(n.length,80),A.writeUInt32LE(n.length>0?b:0,84),A.writeUInt32LE(i.length,88),A.writeUInt32LE(v,92),A.writeUInt32LE(r.length,96),A.writeUInt32LE(y,100),A.writeUInt32LE(K,104),A.writeUInt32LE(w,108),C.forEach((e,t)=>{A.writeUInt32LE(e,_+t*Xs)}),c.forEach((e,t)=>{A.writeUInt32LE(e,m+t*Qs)}),a.forEach((e,t)=>{let[r,n,i]=e,a=g+t*qs;A.writeUInt32LE(r,a),A.writeUInt32LE(n,a+4),A.writeUInt32LE(i!==null?i.offset:0,a+8)}),n.forEach((e,t)=>{let[r,n,i]=e,a=b+t*Ws;A.writeUInt16LE(r,a),A.writeUInt16LE(n,a+2),A.writeUInt32LE(i,a+4)}),i.forEach((e,t)=>{let[r,n,i]=e,a=v+t*Ks;A.writeUInt16LE(r,a),A.writeUInt16LE(n,a+2),A.writeUInt32LE(i,a+4)}),r.forEach((e,t)=>{let{interfaces:r,annotationsDirectory:n}=e,i=r!==null?r.offset:0,a=n!==null?n.offset:0,o=0,s=y+t*Hs;A.writeUInt32LE(e.index,s),A.writeUInt32LE(e.accessFlags,s+4),A.writeUInt32LE(e.superClassIndex,s+8),A.writeUInt32LE(i,s+12),A.writeUInt32LE(e.sourceFileIndex,s+16),A.writeUInt32LE(a,s+20),A.writeUInt32LE(e.classData.offset,s+24),A.writeUInt32LE(o,s+28)}),l.forEach((e,t)=>{let{items:r}=e,n=E[t];A.writeUInt32LE(r.length,n),r.forEach((e,t)=>{A.writeUInt32LE(e.offset,n+4+t*4)})}),S.forEach((e,t)=>{let{offset:r,superConstructor:n}=e,i=1,a=1,o=1,s=0,l=4;A.writeUInt16LE(i,r),A.writeUInt16LE(a,r+2),A.writeUInt16LE(o,r+4),A.writeUInt16LE(s,r+6),A.writeUInt32LE(j[t],r+8),A.writeUInt32LE(l,r+12),A.writeUInt16LE(4208,r+16),A.writeUInt16LE(n,r+18),A.writeUInt16LE(0,r+20),A.writeUInt16LE(14,r+22)}),s.forEach(e=>{let a=e.offset,t=0,r=0,n=e.methods.length,i=0;A.writeUInt32LE(t,a),A.writeUInt32LE(r,a+4),A.writeUInt32LE(n,a+8),A.writeUInt32LE(i,a+12),e.methods.forEach((e,t)=>{let r=a+16+t*8,[n,i]=e;A.writeUInt32LE(n,r),A.writeUInt32LE(i.offset,r+4)})}),t.forEach((e,t)=>{let r=N[t];A.writeUInt32LE(e.types.length,r),e.types.forEach((e,t)=>{A.writeUInt16LE(e,r+4+t*2)})}),o.forEach((e,t)=>{let r=L[t];A.writeUInt32LE(e.types.length,r),e.types.forEach((e,t)=>{A.writeUInt16LE(e,r+4+t*2)})}),k.forEach((e,t)=>{e.copy(A,C[t])}),j.forEach(e=>{yl.copy(A,e)}),G.forEach((e,t)=>{e.copy(A,d[t].offset)}),Z.forEach((e,t)=>{e.copy(A,r[t].classData.offset)}),A.writeUInt32LE(P,M);let x=[[el,1,U],[tl,u.length,_],[rl,c.length,m],[nl,a.length,g]];n.length>0&&x.push([il,n.length,b]),x.push([al,i.length,v]),x.push([ol,r.length,y]),l.forEach((e,t)=>{x.push([dl,e.items.length,E[t]])}),S.forEach(e=>{x.push([ul,1,e.offset])}),s.forEach(e=>{x.push([_l,1,e.offset])}),I>0&&x.push([ll,I,N.concat(L)[0]]),x.push([hl,u.length,C[0]]),j.forEach(e=>{x.push([pl,1,e])}),d.forEach(e=>{x.push([fl,1,e.offset])}),r.forEach(e=>{x.push([cl,1,e.classData.offset])}),x.push([sl,1,M]),x.forEach((e,t)=>{let[r,n,i]=e,a=M+4+t*Ys;A.writeUInt16LE(r,a),A.writeUInt32LE(n,a+4),A.writeUInt32LE(i,a+8)});let T=new Checksum("sha1");return T.update(A.slice(p+F)),Q.from(T.getDigest()).copy(A,p),A.writeUInt32LE(Tl(A,p),O),A}};function Ll(e){let{instanceFields:t,constructorMethods:r,virtualMethods:n}=e.classData;return Q.from([0].concat(X(t.length)).concat(X(r.length)).concat(X(n.length)).concat(t.reduce((e,[t,r])=>e.concat(X(t)).concat(X(r)),[])).concat(r.reduce((e,[t,r,,n])=>e.concat(X(t)).concat(X(r)).concat(X(n||0)),[])).concat(n.reduce((e,[t,r])=>e.concat(X(t)).concat(X(r)).concat([0]),[])))}function kl(e){let{thrownTypes:t}=e;return Q.from([bl].concat(X(e.type)).concat([1]).concat(X(e.value)).concat([gl,t.length]).concat(t.reduce((e,t)=>(e.push(ml,t),e),[])))}function Cl(e){let u=new Set,h=new Set,a={},n=[],p=[],f={},_=new Set,m=new Set;e.forEach(l=>{let{name:d,superClass:c,sourceFileName:e}=l;u.add("this"),u.add(d),h.add(d),u.add(c),h.add(c),u.add(e),l.interfaces.forEach(e=>{u.add(e),h.add(e)}),l.fields.forEach(e=>{let[t,r]=e;u.add(t),u.add(r),h.add(r),n.push([l.name,r,t])}),l.methods.some(([e])=>e==="<init>")||(l.methods.unshift(["<init>","V",[]]),_.add(d)),l.methods.forEach(e=>{let[t,r,n,i=[],a]=e;u.add(t);let o=g(r,n),s=null;if(i.length>0){let e=i.slice();e.sort(),s=e.join("|");let t=f[s];t===void 0&&(t={id:s,types:e},f[s]=t),u.add(wl),h.add(wl),i.forEach(e=>{u.add(e),h.add(e)}),u.add("value")}if(p.push([l.name,o,t,s,a]),t==="<init>"){m.add(d+"|"+o);let e=c+"|"+o;_.has(d)&&!m.has(e)&&(p.push([c,o,t,null,0]),m.add(e))}})});function g(e,t){let r=[e].concat(t),n=r.join("|");if(a[n]!==void 0)return n;u.add(e),h.add(e),t.forEach(e=>{u.add(e),h.add(e)});let i=r.map(Al).join("");return u.add(i),a[n]=[n,i,e,t],n}let i=Array.from(u);i.sort();let b=i.reduce((e,t,r)=>(e[t]=r,e),{}),t=Array.from(h).map(e=>b[e]);t.sort(Ml);let v=t.reduce((e,t,r)=>(e[i[t]]=r,e),{}),r=Object.keys(a).map(e=>a[e]);r.sort(Il);let o={},s=r.map(e=>{let[,t,r,n]=e,i;if(n.length>0){let e=n.join("|");i=o[e],i===void 0&&(i={types:n.map(e=>v[e]),offset:-1},o[e]=i)}else i=null;return[b[t],v[r],i]}),l=r.reduce((e,t,r)=>{let[n]=t;return e[n]=r,e},{}),d=Object.keys(o).map(e=>o[e]),y=n.map(e=>{let[t,r,n]=e;return[v[t],v[r],b[n]]});y.sort(Pl);let w=p.map(e=>{let[t,r,n,i,a]=e;return[v[t],l[r],b[n],i,a]});w.sort(Rl);let c=Object.keys(f).map(e=>f[e]).map(e=>({id:e.id,type:v[wl],value:b.value,thrownTypes:e.types.map(e=>v[e]),offset:-1})),E=c.map(e=>({id:e.id,items:[e],offset:-1})),S=E.reduce((e,t,r)=>(e[t.id]=r,e),{}),N={},L=[],k=e.map(e=>{let l=v[e.name],t=Js,o=v[e.superClass],r,n=e.interfaces.map(e=>v[e]);if(n.length>0){n.sort(Ml);let e=n.join("|");r=N[e],r===void 0&&(r={types:n,offset:-1},N[e]=r)}else r=null;let i=b[e.sourceFileName],a=w.reduce((e,t,r)=>{let[n,i,a,o,s]=t;return n===l&&e.push([r,a,o,i,s]),e},[]),s=null,d=a.filter(([,,e])=>e!==null).map(([e,,t])=>[e,E[S[t]]]);d.length>0&&(s={methods:d,offset:-1},L.push(s));let c=y.reduce((e,t,r)=>{let[n]=t;return n===l&&e.push([r>0?1:0,Js]),e},[]),u=b["<init>"],h=a.filter(([,e])=>e===u).map(([t,,,a])=>{if(_.has(e.name)){let i=-1,e=w.length;for(let n=0;n!==e;n++){let[e,t,r]=w[n];if(e===o&&r===u&&t===a){i=n;break}}return[t,Js|Gs,i]}else return[t,Js|Gs|Bs,-1]}),p=jl(a.filter(([,e])=>e!==u).map(([e,,,,t])=>[e,t|Js|Bs]));return{index:l,accessFlags:t,superClassIndex:o,interfaces:r,sourceFileIndex:i,annotationsDirectory:s,classData:{instanceFields:c,constructorMethods:h,virtualMethods:p,offset:-1}}}),C=Object.keys(N).map(e=>N[e]);return{classes:k,interfaces:C,fields:y,methods:w,protos:s,parameters:d,annotationDirectories:L,annotationSets:E,throwsAnnotations:c,types:t,strings:i}}function jl(e){let i=0;return e.map(([e,t],r)=>{let n;return r===0?n=[e,t]:n=[e-i,t],i=e,n})}function Ml(e,t){return e-t}function Il(e,t){let[,,r,n]=e,[,,i,a]=t;if(r<i)return-1;if(r>i)return 1;let o=n.join("|"),s=a.join("|");return o<s?-1:o>s?1:0}function Pl(e,t){let[r,n,i]=e,[a,o,s]=t;return r!==a?r-a:i!==s?i-s:n-o}function Rl(e,t){let[r,n,i]=e,[a,o,s]=t;return r!==a?r-a:i!==s?i-s:n-o}function Al(e){let t=e[0];return t==="L"||t==="["?"L":e}function X(t){if(t<=127)return[t];let r=[],n=!1;do{let e=t&127;t>>=7,n=t!==0,n&&(e|=128),r.push(e)}while(n);return r}function xl(e,t){let r=e%t;return r===0?e:e+t-r}function Tl(t,r){let n=1,i=0,a=t.length;for(let e=r;e<a;e++)n=(n+t[e])%65521,i=(i+n)%65521;return(i<<16|n)>>>0}var Ul=Sl;var Ol=1,Fl=null,Dl=null;function zl(e){Fl=e}function $l(e,t,r){let n=Bl(e);return n===null&&(e.indexOf("[")===0?n=Kl(e,t,r):(e[0]==="L"&&e[e.length-1]===";"&&(e=e.substring(1,e.length-1)),n=Gl(e,t,r))),Object.assign({className:e},n)}var Vl={boolean:{name:"Z",type:"uint8",size:1,byteSize:1,defaultValue:!1,isCompatible(e){return typeof e=="boolean"},fromJni(e){return!!e},toJni(e){return e?1:0},read(e){return e.readU8()},write(e,t){e.writeU8(t)},toString(){return this.name}},byte:{name:"B",type:"int8",size:1,byteSize:1,defaultValue:0,isCompatible(e){return Number.isInteger(e)&&e>=-128&&e<=127},fromJni:L,toJni:L,read(e){return e.readS8()},write(e,t){e.writeS8(t)},toString(){return this.name}},char:{name:"C",type:"uint16",size:1,byteSize:2,defaultValue:0,isCompatible(e){if(typeof e!="string"||e.length!==1)return!1;let t=e.charCodeAt(0);return t>=0&&t<=65535},fromJni(e){return String.fromCharCode(e)},toJni(e){return e.charCodeAt(0)},read(e){return e.readU16()},write(e,t){e.writeU16(t)},toString(){return this.name}},short:{name:"S",type:"int16",size:1,byteSize:2,defaultValue:0,isCompatible(e){return Number.isInteger(e)&&e>=-32768&&e<=32767},fromJni:L,toJni:L,read(e){return e.readS16()},write(e,t){e.writeS16(t)},toString(){return this.name}},int:{name:"I",type:"int32",size:1,byteSize:4,defaultValue:0,isCompatible(e){return Number.isInteger(e)&&e>=-2147483648&&e<=2147483647},fromJni:L,toJni:L,read(e){return e.readS32()},write(e,t){e.writeS32(t)},toString(){return this.name}},long:{name:"J",type:"int64",size:2,byteSize:8,defaultValue:0,isCompatible(e){return typeof e=="number"||e instanceof Int64},fromJni:L,toJni:L,read(e){return e.readS64()},write(e,t){e.writeS64(t)},toString(){return this.name}},float:{name:"F",type:"float",size:1,byteSize:4,defaultValue:0,isCompatible(e){return typeof e=="number"},fromJni:L,toJni:L,read(e){return e.readFloat()},write(e,t){e.writeFloat(t)},toString(){return this.name}},double:{name:"D",type:"double",size:2,byteSize:8,defaultValue:0,isCompatible(e){return typeof e=="number"},fromJni:L,toJni:L,read(e){return e.readDouble()},write(e,t){e.writeDouble(t)},toString(){return this.name}},void:{name:"V",type:"void",size:0,byteSize:0,defaultValue:void 0,isCompatible(e){return e===void 0},fromJni(){},toJni(){return NULL},toString(){return this.name}}},Jl=new Set(Object.values(Vl).map(e=>e.name));function Bl(e){let t=Vl[e];return t!==void 0?t:null}function Gl(e,t,r){let n=r._types[t?1:0],i=n[e];return i!==void 0||(e==="java.lang.Object"?i=Zl(r):i=Hl(e,t,r),n[e]=i),i}function Zl(n){return{name:"Ljava/lang/Object;",type:"pointer",size:1,defaultValue:NULL,isCompatible(e){return e===null?!0:e===void 0?!1:e.$h instanceof NativePointer?!0:typeof e=="string"},fromJni(e,t,r){return e.isNull()?null:n.cast(e,n.use("java.lang.Object"),r)},toJni(e,t){return e===null?NULL:typeof e=="string"?t.newStringUtf(e):e.$h}}}function Hl(n,i,a){let e=null,r=null,t=null;function o(){return e===null&&(e=a.use(n).class),e}function s(e){let t=o();return r===null&&(r=t.isInstance.overload("java.lang.Object")),r.call(t,e)}function l(){if(t===null){let e=o();t=a.use("java.lang.String").class.isAssignableFrom(e)}return t}return{name:rd(n),type:"pointer",size:1,defaultValue:NULL,isCompatible(e){return e===null?!0:e===void 0?!1:e.$h instanceof NativePointer?s(e):typeof e=="string"&&l()},fromJni(e,t,r){return e.isNull()?null:l()&&i?t.stringFromJni(e):a.cast(e,a.use(n),r)},toJni(e,t){return e===null?NULL:typeof e=="string"?t.newStringUtf(e):e.$h},toString(){return this.name}}}var ql=[["Z","boolean"],["B","byte"],["C","char"],["D","double"],["F","float"],["I","int"],["J","long"],["S","short"]].reduce((e,[t,r])=>(e["["+t]=Wl("["+t,r),e),{});function Wl(e,t){let r=g.prototype,n=nd(t),i={typeName:t,newArray:r["new"+n+"Array"],setRegion:r["set"+n+"ArrayRegion"],getElements:r["get"+n+"ArrayElements"],releaseElements:r["release"+n+"ArrayElements"]};return{name:e,type:"pointer",size:1,defaultValue:NULL,isCompatible(e){return ed(e,t)},fromJni(e,t,r){return Xl(e,i,t,r)},toJni(e,t){return Yl(e,i,t)}}}function Kl(e,t,o){let r=ql[e];if(r!==void 0)return r;if(e.indexOf("[")!==0)throw new Error("Unsupported type: "+e);let s=e.substring(1),l=$l(s,t,o),n=0,i=s.length;for(;n!==i&&s[n]==="[";)n++;s=s.substring(n),s[0]==="L"&&s[s.length-1]===";"&&(s=s.substring(1,s.length-1));let a=s.replace(/\./g,"/");Jl.has(a)?a="[".repeat(n)+a:a="[".repeat(n)+"L"+a+";";let d="["+a;return s="[".repeat(n)+s,{name:e.replace(/\./g,"/"),type:"pointer",size:1,defaultValue:NULL,isCompatible(e){return e===null?!0:typeof e!="object"||e.length===void 0?!1:e.every(function(e){return l.isCompatible(e)})},fromJni(r,n,e){if(r.isNull())return null;let i=[],a=n.getArrayLength(r);for(let t=0;t!==a;t++){let e=n.getObjectArrayElement(r,t);try{i.push(l.fromJni(e,n))}finally{n.deleteLocalRef(e)}}try{i.$w=o.cast(r,o.use(d),e)}catch{o.use("java.lang.reflect.Array").newInstance(o.use(s).class,0),i.$w=o.cast(r,o.use(d),e)}return i.$dispose=Ql,i},toJni(n,i){if(n===null)return NULL;if(!(n instanceof Array))throw new Error("Expected an array");let e=n.$w;if(e!==void 0)return e.$h;let a=n.length,t=o.use(s).$borrowClassHandle(i);try{let r=i.newObjectArray(a,t.value,NULL);i.throwIfExceptionPending();for(let t=0;t!==a;t++){let e=l.toJni(n[t],i);try{i.setObjectArrayElement(r,t,e)}finally{l.type==="pointer"&&i.getObjectRefType(e)===Ol&&i.deleteLocalRef(e)}i.throwIfExceptionPending()}return r}finally{t.unref(i)}}}}function Ql(){let e=this.length;for(let r=0;r!==e;r++){let e=this[r];if(e===null)continue;let t=e.$dispose;if(t===void 0)break;t.call(e)}this.$w.$dispose()}function Xl(e,t,r,n){if(e.isNull())return null;let i=Bl(t.typeName),a=r.getArrayLength(e);return new td(e,t,i,a,r,n)}function Yl(a,e,o){if(a===null)return NULL;let t=a.$h;if(t!==void 0)return t;let s=a.length,l=Bl(e.typeName),d=e.newArray.call(o,s);if(d.isNull())throw new Error("Unable to construct array");if(s>0){let t=l.byteSize,r=l.write,n=l.toJni,i=Memory.alloc(s*l.byteSize);for(let e=0;e!==s;e++)r(i.add(e*t),n(a[e]));e.setRegion.call(o,d,0,s,i),o.throwIfExceptionPending()}return d}function ed(e,t){if(e===null)return!0;if(e instanceof td)return e.$s.typeName===t;if(!(typeof e=="object"&&e.length!==void 0))return!1;let r=Bl(t);return Array.prototype.every.call(e,e=>r.isCompatible(e))}function td(t,e,r,n,i,a=!0){if(a){let e=i.newGlobalRef(t);this.$h=e,this.$r=Script.bindWeak(this,i.vm.makeHandleDestructor(e))}else this.$h=t,this.$r=null;return this.$s=e,this.$t=r,this.length=n,new Proxy(this,Dl)}Dl={has(e,t){return t in e?!0:e.tryParseIndex(t)!==null},get(e,t,r){let n=e.tryParseIndex(t);return n===null?e[t]:e.readElement(n)},set(e,t,r,n){let i=e.tryParseIndex(t);return i===null?(e[t]=r,!0):(e.writeElement(i,r),!0)},ownKeys(e){let r=[],{length:n}=e;for(let t=0;t!==n;t++){let e=t.toString();r.push(e)}return r.push("length"),r},getOwnPropertyDescriptor(e,t){return e.tryParseIndex(t)!==null?{writable:!0,configurable:!0,enumerable:!0}:Object.getOwnPropertyDescriptor(e,t)}};Object.defineProperties(td.prototype,{$dispose:{enumerable:!0,value(){let e=this.$r;e!==null&&(this.$r=null,Script.unbindWeak(e))}},$clone:{value(e){return new td(this.$h,this.$s,this.$t,this.length,e)}},tryParseIndex:{value(e){if(typeof e=="symbol")return null;let t=parseInt(e);return isNaN(t)||t<0||t>=this.length?null:t}},readElement:{value(r){return this.withElements(e=>{let t=this.$t;return t.fromJni(t.read(e.add(r*t.byteSize)))})}},writeElement:{value(e,t){let{$h:r,$s:n,$t:i}=this,a=Fl.getEnv(),o=Memory.alloc(i.byteSize);i.write(o,i.toJni(t)),n.setRegion.call(a,r,e,1,o)}},withElements:{value(e){let{$h:t,$s:r}=this,n=Fl.getEnv(),i=r.getElements.call(n,t);if(i.isNull())throw new Error("Unable to get array elements");try{return e(i)}finally{r.releaseElements.call(n,t,i)}}},toJSON:{value(){let{length:e,$t:t}=this,{byteSize:i,fromJni:a,read:o}=t;return this.withElements(r=>{let n=[];for(let t=0;t!==e;t++){let e=a(o(r.add(t*i)));n.push(e)}return n})}},toString:{value(){return this.toJSON().toString()}}});function rd(e){return"L"+e.replace(/\./g,"/")+";"}function nd(e){return e.charAt(0).toUpperCase()+e.slice(1)}function L(e){return e}var id=4,{ensureClassInitialized:ad,makeMethodMangler:od}=Ve,sd=8,ld=1,dd=2,k=3,cd=1,ud=2,hd=1,pd=2,fd=Symbol("PENDING_USE"),_d="/data/local/tmp",{getCurrentThreadId:md,pointerSize:gd}=Process,M={state:"empty",factories:[],loaders:null,Integer:null},I=null,P=null,bd=null,vd=null,yd=null,wd=null,Ed=null,Sd=null,Nd=null,Ld=new Map,kd=class A{static _initialize(e,t){I=e,P=t,bd=t.flavor==="art",t.flavor==="jvm"&&(ad=fs,od=ws)}static _disposeAll(t){M.factories.forEach(e=>{e._dispose(t)})}static get(e){let t=rc(),r=t.factories[0];if(e===null)return r;let n=t.loaders.get(e);if(n!==null){let e=r.cast(n,t.Integer);return t.factories[e.intValue()]}let i=new A;return i.loader=e,i.cacheDir=r.cacheDir,nc(i,e),i}constructor(){this.cacheDir=_d,this.codeCacheDir=_d+"/dalvik-cache",this.tempFileNaming={prefix:"frida",suffix:""},this._classes={},this._classHandles=new Vs(10,Id),this._patchedMethods=new Set,this._loader=null,this._types=[{},{}],M.factories.push(this)}_dispose(e){Array.from(this._patchedMethods).forEach(e=>{e.implementation=null}),this._patchedMethods.clear(),Va(),this._classHandles.dispose(e),this._classes={}}get loader(){return this._loader}set loader(e){let t=this._loader===null&&e!==null;this._loader=e,t&&M.state==="ready"&&this===M.factories[0]&&nc(this,e)}use(n,e={}){let t=e.cache!=="skip",i=t?this._getUsedClass(n):void 0;if(i===void 0)try{let e=I.getEnv(),{_loader:t}=this,r=t!==null?Rd(n,t,e):Pd(n);i=this._make(n,r,e)}finally{t&&this._setUsedClass(n,i)}return i}_getUsedClass(e){let t;for(;(t=this._classes[e])===fd;)Thread.sleep(.05);return t===void 0&&(this._classes[e]=fd),t}_setUsedClass(e,t){t!==void 0?this._classes[e]=t:delete this._classes[e]}_make(e,t,r){let n=Cd(),i=Object.create(jd.prototype,{[Symbol.for("n")]:{value:e},$n:{get(){return this[Symbol.for("n")]}},[Symbol.for("C")]:{value:n},$C:{get(){return this[Symbol.for("C")]}},[Symbol.for("w")]:{value:null,writable:!0},$w:{get(){return this[Symbol.for("w")]},set(e){this[Symbol.for("w")]=e}},[Symbol.for("_s")]:{writable:!0},$_s:{get(){return this[Symbol.for("_s")]},set(e){this[Symbol.for("_s")]=e}},[Symbol.for("c")]:{value:[null]},$c:{get(){return this[Symbol.for("c")]}},[Symbol.for("m")]:{value:new Map},$m:{get(){return this[Symbol.for("m")]}},[Symbol.for("l")]:{value:null,writable:!0},$l:{get(){return this[Symbol.for("l")]},set(e){this[Symbol.for("l")]=e}},[Symbol.for("gch")]:{value:t},$gch:{get(){return this[Symbol.for("gch")]}},[Symbol.for("f")]:{value:this},$f:{get(){return this[Symbol.for("f")]}}});n.prototype=i;let a=new n(null);i[Symbol.for("w")]=a,i.$w=a;let o=a.$borrowClassHandle(r);try{let e=o.value;ad(r,e),i.$l=Us.build(e,r)}finally{o.unref(r)}return a}retain(e){let t=I.getEnv();return e.$clone(t)}cast(e,t,r){let n=I.getEnv(),i=e.$h;i===void 0&&(i=e);let a=t.$borrowClassHandle(n);try{if(!n.isInstanceOf(i,a.value))throw new Error(`Cast from '${n.getObjectClassName(i)}' to '${t.$n}' isn't possible`)}finally{a.unref(n)}let o=t.$C;return new o(i,hd,n,r)}wrap(e,t,r){let n=t.$C,i=new n(e,hd,r,!1);return i.$r=Script.bindWeak(i,I.makeHandleDestructor(e)),i}array(e,t){let r=I.getEnv(),n=Bl(e);n!==null&&(e=n.name);let i=Kl("["+e,!1,this),a=i.toJni(t,r);return i.fromJni(a,r,!0)}registerClass(p){let w=I.getEnv(),E=[];try{let r=this.use("java.lang.Class"),f=w.javaLangReflectMethod(),_=w.vaMethod("pointer",[]),e=p.name,t=p.implements||[],n=p.superClass||this.use("java.lang.Object"),i=[],a=[],o={name:rd(e),sourceFileName:lc(e),superClass:rd(n.$n),interfaces:t.map(e=>rd(e.$n)),fields:i,methods:a},s=t.slice();t.forEach(e=>{Array.prototype.slice.call(e.class.getInterfaces()).forEach(e=>{let t=this.cast(e,r).getCanonicalName();s.push(this.use(t))})});let l=p.fields||{};Object.getOwnPropertyNames(l).forEach(e=>{let t=this._getType(l[e]);i.push([e,t.name])});let m={},g={};s.forEach(i=>{let e=i.$borrowClassHandle(w);E.push(e);let a=e.value;i.$ownMembers.filter(e=>i[e].overloads!==void 0).forEach(t=>{let e=i[t],r=e.overloads,n=r.map(e=>Dd(t,e.returnType,e.argumentTypes));m[t]=[e,n,a],r.forEach((e,t)=>{let r=n[t];g[r]=[e,a]})})});let d=p.methods||{},c=Object.keys(d).reduce((e,t)=>{let r=d[t],n=t==="$init"?"<init>":t;return r instanceof Array?e.push(...r.map(e=>[n,e])):e.push([n,r]),e},[]),b=[];c.forEach(([s,l])=>{let d=k,c,u,h=[],p;if(typeof l=="function"){let o=m[s];if(o!==void 0&&Array.isArray(o)){let[e,t,r]=o;if(t.length>1)throw new Error(`More than one overload matching '${s}': signature must be specified`);delete g[t[0]];let n=e.overloads[0];d=n.type,c=n.returnType,u=n.argumentTypes,p=l;let i=w.toReflectedMethod(r,n.handle,0),a=_(w.handle,i,f.getGenericExceptionTypes);h=sc(w,a).map(rd),w.deleteLocalRef(a),w.deleteLocalRef(i)}else c=this._getType("void"),u=[],p=l}else{if(l.isStatic&&(d=dd),c=this._getType(l.returnType||"void"),u=(l.argumentTypes||[]).map(e=>this._getType(e)),p=l.implementation,typeof p!="function")throw new Error("Expected a function implementation for method: "+s);let i=Dd(s,c,u),a=g[i];if(a!==void 0){let[e,t]=a;delete g[i],d=e.type,c=e.returnType,u=e.argumentTypes;let r=w.toReflectedMethod(t,e.handle,0),n=_(w.handle,r,f.getGenericExceptionTypes);h=sc(w,n).map(rd),w.deleteLocalRef(n),w.deleteLocalRef(r)}}let e=c.name,t=u.map(e=>e.name),r="("+t.join("")+")"+e;a.push([s,e,t,h,d===dd?sd:0]),b.push([s,r,d,c,u,p])});let u=Object.keys(g);if(u.length>0)throw new Error("Missing implementation for: "+u.join(", "));let h=Yd.fromBuffer(Ul(o),this);try{h.load()}finally{h.file.delete()}let v=this.use(p.name),y=c.length;if(y>0){let c=3*gd,u=Memory.alloc(y*c),h=[],p=[];b.forEach(([e,t,r,n,i,a],o)=>{let s=Memory.allocUtf8String(e),l=Memory.allocUtf8String(t),d=Bd(e,v,r,n,i,a);u.add(o*c).writePointer(s),u.add(o*c+gd).writePointer(l),u.add(o*c+2*gd).writePointer(d),p.push(s,l),h.push(d)});let e=v.$borrowClassHandle(w);E.push(e);let t=e.value;w.registerNatives(t,u,y),w.throwIfExceptionPending(),v.$nativeMethods=h}return v}finally{E.forEach(e=>{e.unref(w)})}}choose(r,n){let i=I.getEnv(),{flavor:e}=P;if(e==="jvm")this._chooseObjectsJvm(r,i,n);else if(e==="art"){let t=P["art::gc::Heap::VisitObjects"]===void 0;if(t&&P["art::gc::Heap::GetInstances"]===void 0)return this._chooseObjectsJvm(r,i,n);w(I,i,e=>{t?this._chooseObjectsArtPreA12(r,i,e,n):this._chooseObjectsArtLegacy(r,i,e,n)})}else this._chooseObjectsDalvik(r,i,n)}_chooseObjectsJvm(e,s,l){let d=this.use(e),{jvmti:c}=P,u=1,h=3,p=d.$borrowClassHandle(s),f=int64(p.value.toString());try{let e=new NativeCallback((e,t,r,n)=>(r.writeS64(f),u),"int",["int64","int64","pointer","pointer"]);c.iterateOverInstancesOfClass(p.value,h,e,p.value);let t=Memory.alloc(8);t.writeS64(f);let r=Memory.alloc(id),n=Memory.alloc(gd);c.getObjectsWithTags(1,t,r,n,NULL);let i=r.readS32(),a=n.readPointer(),o=[];for(let e=0;e!==i;e++)o.push(a.add(e*gd).readPointer());c.deallocate(a);try{for(let t of o){let e=this.cast(t,d);if(l.onMatch(e)==="stop")break}l.onComplete()}finally{o.forEach(e=>{s.deleteLocalRef(e)})}}finally{p.unref(s)}}_chooseObjectsArtPreA12(e,t,r,n){let i=this.use(e),a=Jo.$new(r,I),o,s=i.$borrowClassHandle(t);try{let e=P["art::JavaVMExt::DecodeGlobal"](P.vm,r,s.value);o=a.newHandle(e)}finally{s.unref(t)}let l=0,d=xo.$new();P["art::gc::Heap::GetInstances"](P.artHeap,a,o,l,d);let c=d.handles.map(e=>t.newGlobalRef(e));d.$delete(),a.$delete();try{for(let t of c){let e=this.cast(t,i);if(n.onMatch(e)==="stop")break}n.onComplete()}finally{c.forEach(e=>{t.deleteGlobalRef(e)})}}_chooseObjectsArtLegacy(e,t,r,n){let i=this.use(e),a=[],o=P["art::JavaVMExt::AddGlobalRef"],s=P.vm,l,d=i.$borrowClassHandle(t);try{l=P["art::JavaVMExt::DecodeGlobal"](s,r,d.value).toInt32()}finally{d.unref(t)}let c=Zo(l,e=>{a.push(o(s,r,e))});P["art::gc::Heap::VisitObjects"](P.artHeap,c,NULL);try{for(let t of a){let e=this.cast(t,i);if(n.onMatch(e)==="stop")break}}finally{a.forEach(e=>{t.deleteGlobalRef(e)})}n.onComplete()}_chooseObjectsDalvik(e,t,d){let c=this.use(e);if(P.addLocalReference===null){let e=Process.getModuleByName("libdvm.so"),t;switch(Process.arch){case"arm":t="2d e9 f0 41 05 46 15 4e 0c 46 7e 44 11 b3 43 68";break;case"ia32":t="8d 64 24 d4 89 5c 24 1c 89 74 24 20 e8 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 85 d2";break}Memory.scan(e.base,e.size,t,{onMatch:(n,e)=>{let t;if(Process.arch==="arm")n=n.or(1),t=new NativeFunction(n,"pointer",["pointer","pointer"]);else{let r=Memory.alloc(Process.pageSize);Memory.patchCode(r,16,e=>{let t=new X86Writer(e,{pc:r});t.putMovRegRegOffsetPtr("eax","esp",4),t.putMovRegRegOffsetPtr("edx","esp",8),t.putJmpAddress(n),t.flush()}),t=new NativeFunction(r,"pointer",["pointer","pointer"]),t._thunk=r}return P.addLocalReference=t,I.perform(e=>{r(this,e)}),"stop"},onError(e){},onComplete(){P.addLocalReference===null&&d.onComplete()}})}else r(this,t);function r(a,e){let{DVM_JNI_ENV_OFFSET_SELF:o}=Ve,t=e.handle.add(o).readPointer(),r,n=c.$borrowClassHandle(e);try{r=P.dvmDecodeIndirectRef(t,n.value)}finally{n.unref(e)}let i=r.toMatchPattern(),s=P.dvmHeapSourceGetBase(),l=P.dvmHeapSourceGetLimit().sub(s).toInt32();Memory.scan(s,l,i,{onMatch:(i,e)=>{P.dvmIsValidObject(i)&&I.perform(e=>{let t=e.handle.add(o).readPointer(),r,n=P.addLocalReference(t,i);try{r=a.cast(n,c)}finally{e.deleteLocalRef(n)}if(d.onMatch(r)==="stop")return"stop"})},onError(e){},onComplete(){d.onComplete()}})}}openClassFile(e){return new Yd(e,null,this)}_getType(e,t=!0){return $l(e,t,this)}};function Cd(){return function(e,t,r,n){return jd.call(this,e,t,r,n)}}function jd(t,e,r,n=!0){if(t!==null)if(n){let e=r.newGlobalRef(t);this.$h=e,this.$r=Script.bindWeak(this,I.makeHandleDestructor(e))}else this.$h=t,this.$r=null;else this.$h=null,this.$r=null;return this.$t=e,new Proxy(this,vd)}vd={has(e,t){return t in e?!0:e.$has(t)},get(e,t,r){if(typeof t!="string"||t.startsWith("$")||t==="class")return e[t];let n=e.$find(t);return n!==null?n(r):e[t]},set(e,t,r,n){return e[t]=r,!0},ownKeys(e){return e.$list()},getOwnPropertyDescriptor(e,t){return Object.prototype.hasOwnProperty.call(e,t)?Object.getOwnPropertyDescriptor(e,t):{writable:!1,configurable:!0,enumerable:!0}}};Object.defineProperties(jd.prototype,{[Symbol.for("new")]:{enumerable:!1,get(){return this.$getCtor("allocAndInit")}},$new:{enumerable:!0,get(){return this[Symbol.for("new")]}},[Symbol.for("alloc")]:{enumerable:!1,value(){let t=I.getEnv(),r=this.$borrowClassHandle(t);try{let e=t.allocObject(r.value);return this.$f.cast(e,this)}finally{r.unref(t)}}},$alloc:{enumerable:!0,get(){return this[Symbol.for("alloc")]}},[Symbol.for("init")]:{enumerable:!1,get(){return this.$getCtor("initOnly")}},$init:{enumerable:!0,get(){return this[Symbol.for("init")]}},[Symbol.for("dispose")]:{enumerable:!1,value(){let e=this.$r;e!==null&&(this.$r=null,Script.unbindWeak(e)),this.$h!==null&&(this.$h=void 0)}},$dispose:{enumerable:!0,get(){return this[Symbol.for("dispose")]}},[Symbol.for("clone")]:{enumerable:!1,value(e){let t=this.$C;return new t(this.$h,this.$t,e)}},$clone:{value(e){return this[Symbol.for("clone")](e)}},[Symbol.for("class")]:{enumerable:!1,get(){let e=I.getEnv(),t=this.$borrowClassHandle(e);try{let e=this.$f;return e.cast(t.value,e.use("java.lang.Class"))}finally{t.unref(e)}}},class:{enumerable:!0,get(){return this[Symbol.for("class")]}},[Symbol.for("className")]:{enumerable:!1,get(){let e=this.$h;return e===null?this.$n:I.getEnv().getObjectClassName(e)}},$className:{enumerable:!0,get(){return this[Symbol.for("className")]}},[Symbol.for("ownMembers")]:{enumerable:!1,get(){return this.$l.list()}},$ownMembers:{enumerable:!0,get(){return this[Symbol.for("ownMembers")]}},[Symbol.for("super")]:{enumerable:!1,get(){let e=I.getEnv(),t=this.$s.$C;return new t(this.$h,pd,e)}},$super:{enumerable:!0,get(){return this[Symbol.for("super")]}},[Symbol.for("s")]:{enumerable:!1,get(){let i=Object.getPrototypeOf(this),a=i.$_s;if(a===void 0){let n=I.getEnv(),t=this.$borrowClassHandle(n);try{let e=n.getSuperclass(t.value);if(e.isNull())a=null;else try{let t=n.getClassName(e),r=i.$f;if(a=r._getUsedClass(t),a===void 0)try{let e=Ad(this);a=r._make(t,e,n)}finally{r._setUsedClass(t,a)}}finally{n.deleteLocalRef(e)}}finally{t.unref(n)}i.$_s=a}return a}},$s:{get(){return this[Symbol.for("s")]}},[Symbol.for("isSameObject")]:{enumerable:!1,value(e){return I.getEnv().isSameObject(e.$h,this.$h)}},$isSameObject:{value(e){return this[Symbol.for("isSameObject")](e)}},[Symbol.for("getCtor")]:{enumerable:!1,value(e){let r=this.$c,n=r[0];if(n===null){let e=I.getEnv(),t=this.$borrowClassHandle(e);try{n=xd(t.value,this.$w,e),r[0]=n}finally{t.unref(e)}}return n[e]}},$getCtor:{value(e){return this[Symbol.for("getCtor")](e)}},[Symbol.for("borrowClassHandle")]:{enumerable:!1,value(e){let t=this.$n,r=this.$f._classHandles,n=r.get(t);return n===void 0&&(n=new Md(this.$gch(e),e),r.set(t,n,e)),n.ref()}},$borrowClassHandle:{value(e){return this[Symbol.for("borrowClassHandle")](e)}},[Symbol.for("copyClassHandle")]:{enumerable:!1,value(e){let t=this.$borrowClassHandle(e);try{return e.newLocalRef(t.value)}finally{t.unref(e)}}},$copyClassHandle:{value(e){return this[Symbol.for("copyClassHandle")](e)}},[Symbol.for("getHandle")]:{enumerable:!1,value(e){let t=this.$h;if(t===void 0)throw new Error("Wrapper is disposed; perhaps it was borrowed from a hook instead of calling Java.retain() to make a long-lived wrapper?");return t}},$getHandle:{value(e){return this[Symbol.for("getHandle")](e)}},[Symbol.for("list")]:{enumerable:!1,value(){let e=this.$s,t=e!==null?e.$list():[],r=this.$l;return Array.from(new Set(t.concat(r.list())))}},$list:{get(){return this[Symbol.for("list")]}},[Symbol.for("has")]:{enumerable:!1,value(e){if(this.$m.has(e)||this.$l.has(e))return!0;let t=this.$s;return!!(t!==null&&t.$has(e))}},$has:{value(e){return this[Symbol.for("has")](e)}},[Symbol.for("find")]:{enumerable:!1,value(r){let n=this.$m,i=n.get(r);if(i!==void 0)return i;let a=this.$l.find(r);if(a!==null){let e=I.getEnv(),t=this.$borrowClassHandle(e);try{i=Td(r,a,t.value,this.$w,e)}finally{t.unref(e)}return n.set(r,i),i}let e=this.$s;return e!==null?e.$find(r):null}},$find:{value(e){return this[Symbol.for("find")](e)}},[Symbol.for("toJSON")]:{enumerable:!1,value(){let e=this.$n;if(this.$h===null)return`<class: ${e}>`;let t=this.$className;return e===t?`<instance: ${e}>`:`<instance: ${e}, $className: ${t}>`}},toJSON:{get(){return this[Symbol.for("toJSON")]}}});function Md(e,t){this.value=t.newGlobalRef(e),t.deleteLocalRef(e),this.refs=1}Md.prototype.ref=function(){return this.refs++,this};Md.prototype.unref=function(e){--this.refs===0&&e.deleteGlobalRef(this.value)};function Id(e,t){e.unref(t)}function Pd(e){let r=e.replace(/\./g,"/");return function(e){let t=md();ic(t);try{return e.findClass(r)}finally{ac(t)}}}function Rd(n,i,e){return Nd===null&&(Sd=e.vaMethod("pointer",["pointer"]),Nd=i.loadClass.overload("java.lang.String").handle),e=null,function(t){let r=t.newStringUtf(n),e=md();ic(e);try{let e=Sd(t.handle,i.$h,Nd,r);return t.throwIfExceptionPending(),e}finally{ac(e),t.deleteLocalRef(r)}}}function Ad(r){return function(e){let t=r.$borrowClassHandle(e);try{return e.getSuperclass(t.value)}finally{t.unref(e)}}}function xd(r,a,o){let{$n:e,$f:s}=a,l=oc(e),n=o.javaLangClass(),d=o.javaLangReflectConstructor(),c=o.vaMethod("pointer",[]),i=o.vaMethod("uint8",[]),u=[],h=[],p=s._getType(e,!1),f=s._getType("void",!1),_=c(o.handle,r,n.getDeclaredConstructors);try{let e=o.getArrayLength(_);if(e!==0)for(let i=0;i!==e;i++){let e,t,r=o.getObjectArrayElement(_,i);try{e=o.fromReflectedMethod(r),t=c(o.handle,r,d.getGenericParameterTypes)}finally{o.deleteLocalRef(r)}let n;try{n=sc(o,t).map(e=>s._getType(e))}finally{o.deleteLocalRef(t)}u.push($d(l,a,ld,e,p,n,o)),h.push($d(l,a,k,e,f,n,o))}else{if(i(o.handle,r,n.isInterface))throw new Error("cannot instantiate an interface");let e=o.javaLangObject(),t=o.getMethodId(e,"<init>","()V");u.push($d(l,a,ld,t,p,[],o)),h.push($d(l,a,k,t,f,[],o))}}finally{o.deleteLocalRef(_)}if(h.length===0)throw new Error("no supported overloads");return{allocAndInit:Od(u),initOnly:Od(h)}}function Td(e,t,r,n,i){return t.startsWith("m")?Ud(e,t,r,n,i):Kd(e,t,r,n,i)}function Ud(a,e,o,l,d){let{$f:c}=l,t=e.split(":").slice(1),u=d.javaLangReflectMethod(),h=d.vaMethod("pointer",[]),p=d.vaMethod("uint8",[]),r=t.map(e=>{let t=e[0]==="s"?dd:k,r=ptr(e.substr(1)),n,s=[],i=d.toReflectedMethod(o,r,t===dd?1:0);try{let a=!!p(d.handle,i,u.isVarArgs),e=h(d.handle,i,u.getGenericReturnType);d.throwIfExceptionPending();try{n=c._getType(d.getTypeName(e))}finally{d.deleteLocalRef(e)}let o=h(d.handle,i,u.getParameterTypes);try{let i=d.getArrayLength(o);for(let n=0;n!==i;n++){let e=d.getObjectArrayElement(o,n),t;try{t=a&&n===i-1?d.getArrayTypeName(e):d.getTypeName(e)}finally{d.deleteLocalRef(e)}let r=c._getType(t);s.push(r)}}finally{d.deleteLocalRef(o)}}catch{return null}finally{d.deleteLocalRef(i)}return $d(a,l,t,r,n,s,d)}).filter(e=>e!==null);if(r.length===0)throw new Error("No supported overloads");a==="valueOf"&&Hd(r);let n=Od(r);return function(e){return n}}function Od(e){let t=Fd();return Object.setPrototypeOf(t,yd),t._o=e,t}function Fd(){let e=function(){return e.invoke(this,arguments)};return e}yd=Object.create(Function.prototype,{overloads:{enumerable:!0,get(){return this._o}},overload:{value(...e){let n=this._o,i=e.length,a=e.join(":");for(let r=0;r!==n.length;r++){let e=n[r],{argumentTypes:t}=e;if(t.length!==i)continue;if(t.map(e=>e.className).join(":")===a)return e}zd(this.methodName,this.overloads,"specified argument types do not match any of:")}},methodName:{enumerable:!0,get(){return this._o[0].methodName}},holder:{enumerable:!0,get(){return this._o[0].holder}},type:{enumerable:!0,get(){return this._o[0].type}},handle:{enumerable:!0,get(){return R(this),this._o[0].handle}},implementation:{enumerable:!0,get(){return R(this),this._o[0].implementation},set(e){R(this),this._o[0].implementation=e}},returnType:{enumerable:!0,get(){return R(this),this._o[0].returnType}},argumentTypes:{enumerable:!0,get(){return R(this),this._o[0].argumentTypes}},canInvokeWith:{enumerable:!0,get(e){return R(this),this._o[0].canInvokeWith}},clone:{enumerable:!0,value(e){return R(this),this._o[0].clone(e)}},invoke:{value(r,n){let i=this._o,a=r.$h!==null;for(let t=0;t!==i.length;t++){let e=i[t];if(e.canInvokeWith(n)){if(e.type===k&&!a){let e=this.methodName;if(e==="toString")return`<class: ${r.$n}>`;throw new Error(e+": cannot call instance method without an instance")}return e.apply(r,n)}}if(this.methodName==="toString")return`<class: ${r.$n}>`;zd(this.methodName,this.overloads,"argument types do not match any of:")}}});function Dd(e,t,r){return`${t.className} ${e}(${r.map(e=>e.className).join(", ")})`}function R(e){let t=e._o;t.length>1&&zd(t[0].methodName,t,"has more than one overload, use .overload(<signature>) to choose from:")}function zd(e,t,r){let n=t.slice().sort((e,t)=>e.argumentTypes.length-t.argumentTypes.length).map(e=>e.argumentTypes.length>0?".overload('"+e.argumentTypes.map(e=>e.className).join("', '")+"')":".overload()");throw new Error(`${e}(): ${r}
	${n.join(`
	`)}`)}function $d(e,t,r,n,i,a,o,s){let l=i.type,d=a.map(e=>e.type);o===null&&(o=I.getEnv());let c,u;return r===k?(c=o.vaMethod(l,d,s),u=o.nonvirtualVaMethod(l,d,s)):r===dd?(c=o.staticVaMethod(l,d,s),u=c):(c=o.constructor(d,s),u=c),Vd([e,t,r,n,i,a,c,u])}function Vd(e){let t=Jd();return Object.setPrototypeOf(t,wd),t._p=e,t}function Jd(){let e=function(){return e.invoke(this,arguments)};return e}wd=Object.create(Function.prototype,{methodName:{enumerable:!0,get(){return this._p[0]}},holder:{enumerable:!0,get(){return this._p[1]}},type:{enumerable:!0,get(){return this._p[2]}},handle:{enumerable:!0,get(){return this._p[3]}},implementation:{enumerable:!0,get(){let e=this._r;return e!==void 0?e:null},set(l){let d=this._p,c=d[1];if(d[2]===ld)throw new Error("Reimplementing $new is not possible; replace implementation of $init instead");let e=this._r;if(e!==void 0&&(c.$f._patchedMethods.delete(this),e._m.revert(I),this._r=void 0),l!==null){let[e,t,r,n,i,a]=d,o=Bd(e,t,r,i,a,l,this),s=od(n);o._m=s,this._r=o,s.replace(o,r===k,a,I,P),c.$f._patchedMethods.add(this)}}},returnType:{enumerable:!0,get(){return this._p[4]}},argumentTypes:{enumerable:!0,get(){return this._p[5]}},canInvokeWith:{enumerable:!0,value(r){let e=this._p[5];return r.length!==e.length?!1:e.every((e,t)=>e.isCompatible(r[t]))}},clone:{enumerable:!0,value(e){let t=this._p.slice(0,6);return $d(...t,null,e)}},invoke:{value(o,s){let l=I.getEnv(),d=this._p,e=d[2],c=d[4],u=d[5],h=this._r,p=e===k,f=s.length,t=2+f;l.pushLocalFrame(t);let _=null;try{let e;p?e=o.$getHandle():(_=o.$borrowClassHandle(l),e=_.value);let t,r=o.$t;h===void 0?t=d[3]:(t=h._m.resolveTarget(o,p,l,P),bd&&h._c.has(md())&&(r=pd));let n=[l.handle,e,t];for(let e=0;e!==f;e++)n.push(u[e].toJni(s[e],l));let i;r===hd?i=d[6]:(i=d[7],p&&n.splice(2,0,o.$copyClassHandle(l)));let a=i.apply(null,n);return l.throwIfExceptionPending(),c.fromJni(a,l,!0)}finally{_!==null&&_.unref(l),l.popLocalFrame(NULL)}}},toString:{enumerable:!0,value(){return`function ${this.methodName}(${this.argumentTypes.map(e=>e.className).join(", ")}): ${this.returnType.className}`}}});function Bd(e,t,r,n,i,a,o=null){let s=new Set,l=Gd([e,t,r,n,i,a,o,s]),d=new NativeCallback(l,n.type,["pointer","pointer"].concat(i.map(e=>e.type)));return d._c=s,d}function Gd(e){return function(){return Zd(arguments,e)}}function Zd(a,e){let o=new g(a[0],I),[s,t,r,l,d,c,u,h]=e,p=[],f;if(r===k){let e=t.$C;f=new e(a[1],hd,o,!1)}else f=t;let _=md();o.pushLocalFrame(3);let m=!0;I.link(_,o);try{h.add(_);let e;u===null||!Ld.has(_)?e=c:e=u;let r=[],n=a.length-2;for(let t=0;t!==n;t++){let e=d[t].fromJni(a[2+t],o,!1);r.push(e),p.push(e)}let t=e.apply(f,r);if(!l.isCompatible(t))throw new Error(`Implementation for ${s} expected return value compatible with ${l.className}`);let i=l.toJni(t,o);return l.type==="pointer"&&(i=o.popLocalFrame(i),m=!1,p.push(t)),i}catch(e){let t=e.$h;return t!==void 0?o.throw(t):Script.nextTick(()=>{throw e}),l.defaultValue}finally{I.unlink(_),m&&o.popLocalFrame(NULL),h.delete(_),p.forEach(e=>{if(e===null)return;let t=e.$dispose;t!==void 0&&t.call(e)})}}function Hd(e){let{holder:t,type:r}=e[0];e.some(e=>e.type===r&&e.argumentTypes.length===0)||e.push(qd([t,r]))}function qd(e){let t=Wd();return Object.setPrototypeOf(t,Ed),t._p=e,t}function Wd(){return function(){return this}}Ed=Object.create(Function.prototype,{methodName:{enumerable:!0,get(){return"valueOf"}},holder:{enumerable:!0,get(){return this._p[0]}},type:{enumerable:!0,get(){return this._p[1]}},handle:{enumerable:!0,get(){return NULL}},implementation:{enumerable:!0,get(){return null},set(e){}},returnType:{enumerable:!0,get(){let e=this.holder;return e.$f.use(e.$n)}},argumentTypes:{enumerable:!0,get(){return[]}},canInvokeWith:{enumerable:!0,value(e){return e.length===0}},clone:{enumerable:!0,value(e){throw new Error("Invalid operation")}}});function Kd(e,t,r,n,i){let a=t[2]==="s"?cd:ud,o=ptr(t.substr(3)),{$f:s}=n,l,d=i.toReflectedField(r,o,a===cd?1:0);try{l=i.vaMethod("pointer",[])(i.handle,d,i.javaLangReflectField().getGenericType),i.throwIfExceptionPending()}finally{i.deleteLocalRef(d)}let c;try{c=s._getType(i.getTypeName(l))}finally{i.deleteLocalRef(l)}let u,h,p=c.type;return a===cd?(u=i.getStaticField(p),h=i.setStaticField(p)):(u=i.getField(p),h=i.setField(p)),Qd([a,c,o,u,h])}function Qd(t){return function(e){return new Xd([e].concat(t))}}function Xd(e){this._p=e}Object.defineProperties(Xd.prototype,{value:{enumerable:!0,get(){let[r,n,i,a,o]=this._p,s=I.getEnv();s.pushLocalFrame(4);let l=null;try{let e;if(n===ud){if(e=r.$getHandle(),e===null)throw new Error("Cannot access an instance field without an instance")}else l=r.$borrowClassHandle(s),e=l.value;let t=o(s.handle,e,a);return s.throwIfExceptionPending(),i.fromJni(t,s,!0)}finally{l!==null&&l.unref(s),s.popLocalFrame(NULL)}},set(r){let[n,i,a,o,,s]=this._p,l=I.getEnv();l.pushLocalFrame(4);let d=null;try{let e;if(i===ud){if(e=n.$getHandle(),e===null)throw new Error("Cannot access an instance field without an instance")}else d=n.$borrowClassHandle(l),e=d.value;if(!a.isCompatible(r))throw new Error(`Expected value compatible with ${a.className}`);let t=a.toJni(r,l);s(l.handle,e,o,t),l.throwIfExceptionPending()}finally{d!==null&&d.unref(l),l.popLocalFrame(NULL)}}},holder:{enumerable:!0,get(){return this._p[0]}},fieldType:{enumerable:!0,get(){return this._p[1]}},fieldReturnType:{enumerable:!0,get(){return this._p[2]}},toString:{enumerable:!0,value(){let e=`Java.Field{holder: ${this.holder}, fieldType: ${this.fieldType}, fieldReturnType: ${this.fieldReturnType}, value: ${this.value}}`;return e.length<200?e:`Java.Field{
	holder: ${this.holder},
	fieldType: ${this.fieldType},
	fieldReturnType: ${this.fieldReturnType},
	value: ${this.value},
}`.split(`
`).map(e=>e.length>200?e.slice(0,e.indexOf(" ")+1)+"...,":e).join(`
`)}}});var Yd=class A{static fromBuffer(e,t){let r=ec(t),n=r.getCanonicalPath().toString(),i=new File(n,"w");return i.write(e.buffer),i.close(),tc(n,t),new A(n,r,t)}constructor(e,t,r){this.path=e,this.file=t,this._factory=r}load(){let{_factory:e}=this,{codeCacheDir:t}=e,r=e.use("dalvik.system.DexClassLoader"),n=e.use("java.io.File"),i=this.file;if(i===null&&(i=e.use("java.io.File").$new(this.path)),!i.exists())throw new Error("File not found");n.$new(t).mkdirs(),e.loader=r.$new(i.getCanonicalPath(),t,null,e.loader),I.preventDetachDueToClassLoader()}getClassNames(){let{_factory:e}=this,t=e.use("dalvik.system.DexFile"),r=ec(e),n=t.loadDex(this.path,r.getCanonicalPath(),0),i=[],a=n.entries();for(;a.hasMoreElements();)i.push(a.nextElement().toString());return i}};function ec(e){let{cacheDir:t,tempFileNaming:r}=e,n=e.use("java.io.File"),i=n.$new(t);return i.mkdirs(),n.createTempFile(r.prefix,r.suffix+".dex",i)}function tc(e,t){t.use("java.io.File").$new(e).setWritable(!1,!1)}function rc(){switch(M.state){case"empty":{M.state="pending";let e=M.factories[0],t=e.use("java.util.HashMap"),r=e.use("java.lang.Integer");M.loaders=t.$new(),M.Integer=r;let n=e.loader;return n!==null&&nc(e,n),M.state="ready",M}case"pending":do{Thread.sleep(.05)}while(M.state==="pending");return M;case"ready":return M}}function nc(e,t){let{factories:r,loaders:n,Integer:i}=M,a=i.$new(r.indexOf(e));n.put(t,a);for(let e=t.getParent();e!==null&&!n.containsKey(e);e=e.getParent())n.put(e,a)}function ic(e){let t=Ld.get(e);t===void 0&&(t=0),t++,Ld.set(e,t)}function ac(e){let t=Ld.get(e);if(t===void 0)throw new Error(`Thread ${e} is not ignored`);t--,t===0?Ld.delete(e):Ld.set(e,t)}function oc(e){return e.slice(e.lastIndexOf(".")+1)}function sc(r,n){let i=[],e=r.getArrayLength(n);for(let t=0;t!==e;t++){let e=r.getObjectArrayElement(n,t);try{i.push(r.getTypeName(e))}finally{r.deleteLocalRef(e)}}return i}function lc(e){let t=e.split(".");return t[t.length-1]+".java"}var dc=4,cc=Process.pointerSize,uc=class{ACC_PUBLIC=1;ACC_PRIVATE=2;ACC_PROTECTED=4;ACC_STATIC=8;ACC_FINAL=16;ACC_SYNCHRONIZED=32;ACC_BRIDGE=64;ACC_VARARGS=128;ACC_NATIVE=256;ACC_ABSTRACT=1024;ACC_STRICT=2048;ACC_SYNTHETIC=4096;constructor(){this.classFactory=null,this.ClassFactory=kd,this.vm=null,this.api=null,this._initialized=!1,this._apiError=null,this._wakeupHandler=null,this._pollListener=null,this._pendingMainOps=[],this._pendingVmOps=[],this._cachedIsAppProcess=null;try{this._tryInitialize()}catch{}}_tryInitialize(){if(this._initialized)return!0;if(this._apiError!==null)throw this._apiError;let e;try{e=Rs(),this.api=e}catch(e){throw this._apiError=e,e}if(e===null)return!1;let t=new Hr(e);return this.vm=t,zl(t),kd._initialize(t,e),this.classFactory=new kd,this._initialized=!0,!0}_dispose(){if(this.api===null)return;let{vm:e}=this;e.perform(e=>{kd._disposeAll(e),g.dispose(e)}),Script.nextTick(()=>{Hr.dispose(e)})}get available(){return this._tryInitialize()}get androidVersion(){return Xn()}synchronized(e,t){let{$h:r=e}=e;if(!(r instanceof NativePointer))throw new Error("Java.synchronized: the first argument `obj` must be either a pointer or a Java instance");let n=this.vm.getEnv();h("VM::MonitorEnter",n.monitorEnter(r));try{t()}finally{n.monitorExit(r)}}enumerateLoadedClasses(e){this._checkAvailable();let{flavor:t}=this.api;t==="jvm"?this._enumerateLoadedClassesJvm(e):t==="art"?this._enumerateLoadedClassesArt(e):this._enumerateLoadedClassesDalvik(e)}enumerateLoadedClassesSync(){let t=[];return this.enumerateLoadedClasses({onMatch(e){t.push(e)},onComplete(){}}),t}enumerateClassLoaders(e){this._checkAvailable();let{flavor:t}=this.api;if(t==="jvm")this._enumerateClassLoadersJvm(e);else if(t==="art")this._enumerateClassLoadersArt(e);else throw new Error("Enumerating class loaders is not supported on Dalvik")}enumerateClassLoadersSync(){let t=[];return this.enumerateClassLoaders({onMatch(e){t.push(e)},onComplete(){}}),t}_enumerateLoadedClassesJvm(r){let{api:e,vm:t}=this,{jvmti:n}=e,i=t.getEnv(),a=Memory.alloc(dc),o=Memory.alloc(cc);n.getLoadedClasses(a,o);let s=a.readS32(),l=o.readPointer(),d=[];for(let e=0;e!==s;e++)d.push(l.add(e*cc).readPointer());n.deallocate(l);try{for(let t of d){let e=i.getClassName(t);r.onMatch(e,t)}r.onComplete()}finally{d.forEach(e=>{i.deleteLocalRef(e)})}}_enumerateClassLoadersJvm(e){this.choose("java.lang.ClassLoader",e)}_enumerateLoadedClassesArt(n){let{vm:e,api:t}=this,i=e.getEnv(),a=t["art::JavaVMExt::AddGlobalRef"],{vm:o}=t;w(e,i,r=>{let e=ca(e=>{let t=a(o,r,e);try{let e=i.getClassName(t);n.onMatch(e,t)}finally{i.deleteGlobalRef(t)}return!0});t["art::ClassLinker::VisitClasses"](t.artClassLinker.address,e)}),n.onComplete()}_enumerateClassLoadersArt(r){let{classFactory:n,vm:e,api:i}=this,t=e.getEnv(),a=i["art::ClassLinker::VisitClassLoaders"];if(a===void 0)throw new Error("This API is only available on Android >= 7.0");let o=n.use("java.lang.ClassLoader"),s=[],l=i["art::JavaVMExt::AddGlobalRef"],{vm:d}=i;w(e,t,t=>{let e=ha(e=>(s.push(l(d,t,e)),!0));la(()=>{a(i.artClassLinker.address,e)})});try{s.forEach(e=>{let t=n.cast(e,o);r.onMatch(t)})}finally{s.forEach(e=>{t.deleteGlobalRef(e)})}r.onComplete()}_enumerateLoadedClassesDalvik(n){let{api:e}=this,i=ptr("0xcbcacccd"),t=172,a=8,r=e.gDvm.add(t).readPointer(),o=r.readS32(),s=r.add(12).readPointer(),l=o*a;for(let r=0;r<l;r+=a){let e=s.add(r).add(4).readPointer();if(e.isNull()||e.equals(i))continue;let t=e.add(24).readPointer().readUtf8String();if(t.startsWith("L")){let e=t.substring(1,t.length-1).replace(/\//g,".");n.onMatch(e)}}n.onComplete()}enumerateMethods(e){let{classFactory:r}=this,n=this.vm.getEnv(),i=r.use("java.lang.ClassLoader");return Us.enumerateMethods(e,this.api,n).map(e=>{let t=e.loader;return e.loader=t!==null?r.wrap(t,i,n):null,e})}scheduleOnMainThread(e){this.performNow(()=>{this._pendingMainOps.push(e);let{_wakeupHandler:n}=this;if(n===null){let{classFactory:e}=this,t=e.use("android.os.Handler"),r=e.use("android.os.Looper");n=t.$new(r.getMainLooper()),this._wakeupHandler=n}this._pollListener===null&&(this._pollListener=Interceptor.attach(Process.getModuleByName("libc.so").getExportByName("epoll_wait"),this._makePollHook()),Interceptor.flush()),n.sendEmptyMessage(1)})}_makePollHook(){let t=Process.id,{_pendingMainOps:r}=this;return function(){if(this.threadId!==t)return;let e;for(;(e=r.shift())!==void 0;)try{e()}catch(e){Script.nextTick(()=>{throw e})}}}perform(e){if(this._checkAvailable(),!this._isAppProcess()||this.classFactory.loader!==null)try{this.vm.perform(e)}catch(e){Script.nextTick(()=>{throw e})}else this._pendingVmOps.push(e),this._pendingVmOps.length===1&&this._performPendingVmOpsWhenReady()}performNow(e){return this._checkAvailable(),this.vm.perform(()=>{let{classFactory:t}=this;if(this._isAppProcess()&&t.loader===null){let e=t.use("android.app.ActivityThread").currentApplication();e!==null&&hc(t,e)}return e()})}_performPendingVmOpsWhenReady(){this.vm.perform(()=>{let{classFactory:n}=this,e=n.use("android.app.ActivityThread"),t=e.currentApplication();if(t!==null){hc(n,t),this._performPendingVmOps();return}let i=this,a=!1,o="early",r=e.handleBindApplication;r.implementation=function(e){if(e.instrumentationName.value!==null){o="late";let r=n.use("android.app.LoadedApk").makeApplication;r.implementation=function(e,t){return a||(a=!0,pc(n,this),i._performPendingVmOps()),r.apply(this,arguments)}}r.apply(this,arguments)};let s=e.getPackageInfo.overloads.map(e=>[e.argumentTypes.length,e]).sort(([e],[t])=>t-e).map(([e,t])=>t)[0];s.implementation=function(...e){let t=s.call(this,...e);return!a&&o==="early"&&(a=!0,pc(n,t),i._performPendingVmOps()),t}})}_performPendingVmOps(){let{vm:e,_pendingVmOps:t}=this,r;for(;(r=t.shift())!==void 0;)try{e.perform(r)}catch(e){Script.nextTick(()=>{throw e})}}use(e,t){return this.classFactory.use(e,t)}openClassFile(e){return this.classFactory.openClassFile(e)}choose(e,t){this.classFactory.choose(e,t)}retain(e){return this.classFactory.retain(e)}cast(e,t){return this.classFactory.cast(e,t)}array(e,t){return this.classFactory.array(e,t)}backtrace(e){return Da(this.vm,e)}isMainThread(){let e=this.classFactory.use("android.os.Looper"),t=e.getMainLooper(),r=e.myLooper();return r===null?!1:t.$isSameObject(r)}registerClass(e){return this.classFactory.registerClass(e)}deoptimizeEverything(){let{vm:e}=this;return po(e,e.getEnv())}deoptimizeBootImage(){let{vm:e}=this;return fo(e,e.getEnv())}deoptimizeMethod(e){let{vm:t}=this;return ho(t,t.getEnv(),e)}_checkAvailable(){if(!this.available)throw new Error("Java API not available")}_isAppProcess(){let a=this._cachedIsAppProcess;if(a===null){if(this.api.flavor==="jvm")return a=!1,this._cachedIsAppProcess=a,a;let e=new NativeFunction(Module.getGlobalExportByName("readlink"),"pointer",["pointer","pointer","pointer"],{exceptions:"propagate"}),t=Memory.allocUtf8String("/proc/self/exe"),r=1024,n=Memory.alloc(r),i=e(t,n,ptr(r)).toInt32();if(i!==-1){let e=n.readUtf8String(i);a=/^\/system\/bin\/app_process/.test(e)}else a=!0;this._cachedIsAppProcess=a}return a}};function hc(e,t){let r=e.use("android.os.Process");e.loader=t.getClassLoader(),r.myUid()===r.SYSTEM_UID.value?(e.cacheDir="/data/system",e.codeCacheDir="/data/dalvik-cache"):"getCodeCacheDir"in t?(e.cacheDir=t.getCacheDir().getCanonicalPath(),e.codeCacheDir=t.getCodeCacheDir().getCanonicalPath()):(e.cacheDir=t.getFilesDir().getCanonicalPath(),e.codeCacheDir=t.getCacheDir().getCanonicalPath())}function pc(e,t){let r=e.use("java.io.File");e.loader=t.getClassLoader();let n=r.$new(t.getDataDir()).getCanonicalPath();e.cacheDir=n,e.codeCacheDir=n+"/cache"}var fc=new uc;Script.bindWeak(fc,()=>{fc._dispose()});var _c=fc;Object.defineProperty(globalThis,"Java",{value:_c})})();